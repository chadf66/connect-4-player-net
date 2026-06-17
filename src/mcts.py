"""Network-guided PUCT Monte Carlo Tree Search (the AlphaZero variant).

The network supplies two things to the search: a *policy prior* (which moves look
worth exploring) and a *value estimate* (how good a leaf position is). The search
turns those into a much sharper move distribution by looking ahead — and, crucially,
by backing up the *true* game result whenever it reaches a terminal position. That
terminal signal is what lets training bootstrap from a randomly-initialised network.

Sign convention (the one detail that's easy to get wrong)
---------------------------------------------------------
Every node stores its statistics from the perspective of *the player to move at that
node*. A node's value `Q` answers "how good is this position for the side about to
move here?" Consequently:

- During **selection**, a parent choosing among its children negates the child's `Q`:
  what's good for the child's side-to-move (the opponent) is bad for the parent's.
- During **backup**, the leaf value is flipped at every step up the tree, because
  adjacent plies belong to opposing players (negamax).

For a terminal leaf there is no network call: the side to move has either just lost
(value −1) or drawn (value 0) — you can never be *winning* on the move after your
opponent completed four-in-a-row.

This implementation rebuilds the tree from scratch for every move and evaluates one
position per network call — chosen for clarity over speed (no subtree reuse, virtual
loss, or batched leaf evaluation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src.encode import encode_batch, legal_mask_batch
from src.game import COLS, Board
from src.model import ConnectFourNet, mask_illegal_logits


@dataclass
class MCTSConfig:
    num_simulations: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25


class Node:
    """One position in the search tree.

    `N`, `W`, `Q` are from the perspective of the player to move at `board`.
    `prior` is this node's probability under its parent's policy.
    """

    __slots__ = ("board", "prior", "N", "W", "children", "is_expanded")

    def __init__(self, board: Board, prior: float) -> None:
        self.board = board
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children: dict[int, Node] = {}
        self.is_expanded = False

    @property
    def Q(self) -> float:
        """Mean value from this node's side-to-move perspective (0 if unvisited)."""
        return self.W / self.N if self.N > 0 else 0.0


@torch.no_grad()
def _evaluate_batch(
    model: ConnectFourNet, boards: list[Board], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the network on a batch of positions in a single forward pass.

    Returns (priors, values): `priors` is (B, COLS) — each row a policy over legal
    moves (illegal columns 0); `values` is (B,) value-head estimates, each from the
    perspective of the player to move at that board.

    Batching is the whole point of the parallel self-play path: the network is tiny,
    so one forward over B positions costs barely more wall-clock than B=1, which
    amortises away the per-call GPU dispatch overhead that dominates batch-1 search.
    """
    states = encode_batch(boards).to(device)
    masks = legal_mask_batch(boards).to(device)
    logits, values = model(states)
    logits = mask_illegal_logits(logits, masks)
    priors = F.softmax(logits, dim=-1).cpu().numpy()
    return priors, values.cpu().numpy()


def _evaluate(
    model: ConnectFourNet, board: Board, device: torch.device
) -> tuple[np.ndarray, float]:
    """Single-position convenience wrapper around `_evaluate_batch`."""
    priors, values = _evaluate_batch(model, [board], device)
    return priors[0], float(values[0])


def _expand(node: Node, priors: np.ndarray) -> None:
    """Create a child for each legal move, seeded with the network's prior."""
    for col in node.board.legal_moves():
        node.children[col] = Node(node.board.play(col), prior=float(priors[col]))
    node.is_expanded = True


def _terminal_value(board: Board) -> float:
    """Value of a finished game from the perspective of the player to move.

    The side to move never completed the last connect-four (the previous player
    did), so a decided game is a loss (−1) for them; a full board is a draw (0).
    """
    return 0.0 if board.winner() is None else -1.0


def _select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    """Pick the child maximising PUCT, scored from the parent's perspective."""
    sqrt_total = math.sqrt(node.N)
    best_score = -float("inf")
    best = None
    for col, child in node.children.items():
        # -child.Q: the child's value is from the opponent's perspective.
        u = c_puct * child.prior * sqrt_total / (1 + child.N)
        score = -child.Q + u
        if score > best_score:
            best_score = score
            best = (col, child)
    assert best is not None  # an expanded non-terminal node always has children
    return best


def _select_leaf(root: Node, c_puct: float) -> list[Node]:
    """Descend from the root by PUCT until reaching a leaf (unexpanded or terminal).

    Returns the full root→leaf path (so backup can walk it).
    """
    node = root
    path = [node]
    while node.is_expanded and not node.board.terminal():
        _, node = _select_child(node, c_puct)
        path.append(node)
    return path


def _backup(path: list[Node], value: float) -> None:
    """Propagate `value` up the path, flipping sign at every ply (negamax).

    `value` is from the perspective of the player to move at the leaf.
    """
    for node in reversed(path):
        node.N += 1
        node.W += value
        value = -value


def _root_counts(root: Node) -> np.ndarray:
    """Per-column visit counts at the root (0 for illegal / never-visited)."""
    counts = np.zeros(COLS, dtype=np.float64)
    for col, child in root.children.items():
        counts[col] = child.N
    return counts


def _new_root(
    model: ConnectFourNet,
    board: Board,
    root_priors: np.ndarray,
    config: MCTSConfig,
    generator: np.random.Generator | None,
    add_noise: bool,
) -> Node:
    """Build and expand a root node from pre-computed priors."""
    root = Node(board, prior=1.0)
    if add_noise:
        root_priors = _add_dirichlet_noise(root_priors, board, config, generator)
    _expand(root, root_priors)
    return root


def run_mcts(
    model: ConnectFourNet,
    board: Board,
    device: torch.device,
    config: MCTSConfig,
    generator: np.random.Generator | None = None,
    add_noise: bool = True,
) -> np.ndarray:
    """Run MCTS from `board` and return per-column visit counts (length COLS).

    The clear, sequential reference implementation (one network call per leaf). The
    batched `run_mcts_batched` produces identical per-tree results; this version is
    kept for readability and for single-position use (e.g. `MCTSPlayer`).

    Illegal / never-visited columns have count 0. `add_noise` mixes Dirichlet noise
    into the root priors (self-play exploration; off for deterministic evaluation).
    """
    model.eval()
    root_priors, _ = _evaluate(model, board, device)
    root = _new_root(model, board, root_priors, config, generator, add_noise)

    for _ in range(config.num_simulations):
        path = _select_leaf(root, config.c_puct)
        leaf = path[-1]
        if leaf.board.terminal():
            value = _terminal_value(leaf.board)
        else:
            priors, value = _evaluate(model, leaf.board, device)
            _expand(leaf, priors)
        _backup(path, value)

    return _root_counts(root)


def run_mcts_batched(
    model: ConnectFourNet,
    boards: list[Board],
    device: torch.device,
    config: MCTSConfig,
    generator: np.random.Generator | None = None,
    add_noise: bool = True,
) -> list[np.ndarray]:
    """Run an independent MCTS for each board, batching all network evaluations.

    The searches advance in lockstep: at each simulation, every tree selects the one
    leaf it wants evaluated, and all those leaves are scored in a *single* forward
    pass. Each tree's search is identical to what `run_mcts` would produce for it
    alone — only the timing of evaluations is shared. Returns one visit-count array
    per input board, in order.
    """
    model.eval()

    # Batched root evaluation, then build+expand every root.
    root_priors, _ = _evaluate_batch(model, boards, device)
    roots = [
        _new_root(model, board, root_priors[i], config, generator, add_noise)
        for i, board in enumerate(boards)
    ]

    for _ in range(config.num_simulations):
        # Selection: one leaf per tree. Terminal leaves back up immediately (no eval);
        # the rest are gathered for a single batched forward pass.
        pending: list[tuple[list[Node], int]] = []  # (path, tree index)
        for i, root in enumerate(roots):
            path = _select_leaf(root, config.c_puct)
            if path[-1].board.terminal():
                _backup(path, _terminal_value(path[-1].board))
            else:
                pending.append((path, i))

        if pending:
            leaf_boards = [path[-1].board for path, _ in pending]
            priors, values = _evaluate_batch(model, leaf_boards, device)
            for k, (path, _) in enumerate(pending):
                _expand(path[-1], priors[k])
                _backup(path, float(values[k]))

    return [_root_counts(root) for root in roots]


def _add_dirichlet_noise(
    priors: np.ndarray,
    board: Board,
    config: MCTSConfig,
    generator: np.random.Generator | None,
) -> np.ndarray:
    """Mix symmetric Dirichlet noise into the priors of the legal root moves."""
    gen = generator or np.random.default_rng()
    legal = board.legal_moves()
    noise = gen.dirichlet([config.dirichlet_alpha] * len(legal))
    noisy = priors.copy()
    eps = config.dirichlet_epsilon
    for col, n in zip(legal, noise):
        noisy[col] = (1 - eps) * priors[col] + eps * n
    return noisy


def visit_distribution(counts: np.ndarray, temperature: float) -> np.ndarray:
    """Turn visit counts into a move probability distribution, π ∝ N^(1/τ).

    τ → 0 collapses to a one-hot on the most-visited move (greedy play).
    """
    if temperature <= 1e-6:
        probs = np.zeros_like(counts)
        probs[int(counts.argmax())] = 1.0
        return probs
    scaled = counts ** (1.0 / temperature)
    total = scaled.sum()
    if total == 0:
        # No visits recorded (shouldn't happen with ≥1 simulation) — fall back to uniform.
        return counts / counts.sum() if counts.sum() else counts
    return scaled / total


class MCTSPlayer:
    """A search-backed player (matches the `Player` protocol) for eval and play.

    Picks the most-visited root move (greedy), with no root noise — deterministic
    given the network.
    """

    def __init__(
        self,
        model: ConnectFourNet,
        device: torch.device,
        num_simulations: int = 800,
        c_puct: float = 1.5,
    ) -> None:
        self.model = model
        self.device = device
        self.config = MCTSConfig(num_simulations=num_simulations, c_puct=c_puct)

    def choose(self, board: Board) -> int:
        counts = run_mcts(self.model, board, self.device, self.config, add_noise=False)
        return int(counts.argmax())
