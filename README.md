# Connect Four Player Net

An educational project: train a neural network to play Connect Four from scratch
via AlphaZero-style self-play — network-guided MCTS in the loop from the very first
iteration — and measure it against a perfect-play solver.

The point is to **learn deep-learning fundamentals** by building the whole
self-play RL pipeline by hand — game engine, encoder, residual CNN, PUCT MCTS,
self-play, replay buffer, training loop, and an evaluation harness measured against
a perfect-play solver.

## Status

**Implemented and tested (41 tests passing).** The full AlphaZero recipe: a
residual policy+value CNN trained from random initialisation by self-play, with
PUCT MCTS driving every move. The search's visit-count distribution is the policy
target; game outcomes are the value target. Evaluation covers win-rate vs
random/heuristic baselines and move-match-rate against the Pascal Pons solver,
optionally with search (`--mcts`).

> **Design note.** The original plan had a Phase 1 (train a *raw* policy network
> with a REINFORCE loss) before adding MCTS in Phase 2. We skipped the raw-network
> phase entirely — there's no need to train a network without search — and went
> straight to the pure-AlphaZero approach: MCTS from iteration 0.

## How it works

1. **Self-play with MCTS.** The current network plays full games against itself.
   At each move, PUCT MCTS runs `--sims` simulations from the position (each
   simulation = one network forward pass at the leaf), producing visit counts over
   the 7 columns. The move is sampled from the visit distribution `π ∝ N^(1/τ)` —
   full temperature for the first few plies (diverse openings), then greedy.
2. **The search bootstraps from terminal truth.** At terminal leaves MCTS backs up
   the *real* game result (not a network estimate), so even a random network gets
   ground-truth signal near the end of games — and that signal propagates back
   through training over iterations.
3. **Labeling.** When a game ends, every recorded position is labeled with the
   outcome (`+1` / `0` / `-1`) from the perspective of the player who moved there.
4. **Replay buffer.** Positions `(state, π, outcome)` go into a sliding-window
   buffer. Batches are sampled uniformly with horizontal-mirror augmentation (the
   only symmetry Connect Four has — gravity breaks the rest).
5. **Training.** SGD on each batch:
   - *Policy loss* — cross-entropy between the network policy and the MCTS visit
     distribution `π` (a much richer target than the single move played).
   - *Value loss* — `MSE(V(s), outcome)`.
6. **Evaluation.** Periodic win-rate vs the baselines (cheap raw-policy probe); on
   demand, search-backed win-rate and solver move-match-rate.

### MCTS sign convention

The one correctness-critical detail in [src/mcts.py](src/mcts.py): every node's
statistics are stored from the perspective of *the player to move at that node*.
So selection negates the child's `Q` (what's good for the opponent is bad for me),
and backup flips the value sign at each ply up the tree (negamax). The mate-in-1
and must-block tests in [tests/test_mcts.py](tests/test_mcts.py) run with a *random*
network specifically to pin this down — if the search still finds a forced win/block
with uninformative priors, the terminal-value handling and signs are correct.

### Board & tensor conventions

- Board is 6 rows × 7 columns; row 0 is the bottom (pieces fall down). Players are
  `1` and `2`.
- The encoded input is `(3, 6, 7)`, oriented **from the current player's
  perspective**: channel 0 = my pieces, channel 1 = opponent's, channel 2 = empty.
  The network never has to learn "which color am I."

### Network

Residual CNN in the AlphaZero style ([src/model.py](src/model.py)):

- Stem: `Conv2d(3 → 128, 3×3)` → BN → ReLU
- 8 residual blocks (128 channels, 3×3)
- **Policy head**: `1×1 conv → 2 ch` → flatten → linear → 7 logits
- **Value head**: `1×1 conv → 1 ch` → flatten → 64 → `tanh` scalar in `(-1, 1)`

Illegal-move masking (`-inf` logits) lives in the sampling/loss code, not the
model — cleaner separation.

## Project layout

```
src/
├── game.py        # Board state, legal moves, gravity, win detection
├── encode.py      # Board ↔ tensor (current-player POV) + mirror augmentation
├── model.py       # Residual CNN with policy + value heads; device selection
├── mcts.py        # PUCT MCTS (network-guided); sequential + batched search
├── selfplay.py    # Parallel MCTS self-play; records (state, π, outcome)
├── replay.py      # Sliding-window replay buffer (+ mirror at sample time)
├── train.py       # Training loop: MCTS self-play → buffer → SGD; auto-resume
├── opponents.py   # Random + heuristic baseline players
├── solver.py      # Pascal Pons perfect-play solver wrapper (eval only)
├── evaluate.py    # Win-rate matches + solver move-match-rate; CLI
├── play.py        # Play against the trained network in the terminal
├── app.py         # Streamlit dashboard (dev): board + AI "thinking" internals
└── four_in_a_row.py   # Streamlit app (public): clean, deployable, no internals
tests/             # Game rules, encoding/mirror, solver wrapper, MCTS (41 tests)
third_party/
└── connect4-solver/   # Pascal Pons C++ solver (vendored; c4solver binary built)
checkpoints/       # Saved weights as iter_NNNN.pt (+ frozen test_positions.pkl)
```

## Setup

The environment is managed by [uv](https://docs.astral.sh/uv/). Python 3.13.

```bash
uv sync          # install torch, numpy, tqdm (+ pytest dev group)
```

On Apple Silicon, training uses the MPS backend automatically (CUDA if present,
else CPU) — see `select_device()` in [src/model.py](src/model.py#L85).

### Building the solver (for solver-based evaluation)

The C++ source is vendored under [third_party/connect4-solver/](third_party/connect4-solver/);
the compiled binary is gitignored. Build it once:

```bash
cd third_party/connect4-solver && make
```

The solver is used **only for evaluation**, never as a training target — using it
as a target would be supervised learning and defeat the purpose.

## Usage

All commands run from the project root via `uv run`.

### Tests

```bash
uv run pytest tests/ -v
```

### Smoke training run (a few minutes; use low `--sims` to keep it fast)

```bash
uv run python -m src.train --iterations 3 --games-per-iter 8 --steps-per-iter 50 --sims 32
```

Sanity checks: loss is finite and trends down (RL is noisy, not monotone), no
NaN/Inf, games terminate. (`train.py` auto-resumes from the latest checkpoint in
`--checkpoints-dir`, so clear old `iter_*.pt` before re-running a fresh smoke test.)

### Full training run (overnight)

```bash
uv run python -m src.train --iterations 100 --games-per-iter 50 --steps-per-iter 500 --sims 400
```

Checkpoints are written to `checkpoints/iter_NNNN.pt` every `--save-every`
iterations. Each bundles model weights, optimizer state, **and the replay buffer**,
so a re-run auto-resumes from the latest checkpoint in `--checkpoints-dir` —
overnight stop/start loses no buffered self-play.

**Speed.** Self-play games are played in parallel pools (up to `max_parallel`,
default 64) that share batched network evaluations — see [src/selfplay.py](src/selfplay.py).
On an M-series MPS this measures ~**1.9 s/game at 400 sims** (vs ~27.6 s/game for
one-game-at-a-time search — a ~14× speedup, since the tiny net makes a 64-wide
forward nearly as cheap as a single one). At the defaults above that's roughly
**~2 min/iteration**, so a 100-iteration run is on the order of **3–4 hours**.
Raise `--sims` for stronger search (cost scales ≈ linearly) or lower it to iterate
faster. How many iterations are needed to reach the solver-match target is itself
unknown — watch the eval metrics below and stop when they plateau.

**What to watch.** Each iteration prints a line with loss (total / policy / value),
mean value-head output, average game length, and timing. Every `--eval-every`
iterations it also prints win-rate vs random and vs the heuristic *and* the **solver
move-match-rate** on a small frozen suite — the headline "is it actually good?"
signal. All of this is also appended to `checkpoints/metrics.jsonl` (one JSON object
per iteration) so a long run is plottable afterward, not just terminal scrollback.

Two caveats on reading these: (1) **loss is a weak progress signal** — the targets
move as the network and search improve, so it won't trend cleanly to zero; lean on
win-rate and solver-match instead. (2) The in-loop eval plays the **raw network**
(policy argmax, no search), which understates the MCTS-backed playing strength —
use `evaluate --mcts` for the true number.

Key `train.py` flags (defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--iterations` | 100 | self-play + train cycles |
| `--games-per-iter` | 50 | MCTS self-play games per iteration |
| `--steps-per-iter` | 500 | SGD steps per iteration |
| `--sims` | 400 | MCTS simulations per move |
| `--c-puct` | 1.5 | PUCT exploration constant |
| `--batch-size` | 256 | |
| `--lr` | 1e-3 | AdamW learning rate |
| `--weight-decay` | 1e-4 | |
| `--buffer-capacity` | 100000 | sliding-window size |
| `--eval-every` | 10 | iterations between baseline evals |
| `--eval-games` | 60 | games per in-loop eval |
| `--solver-eval-positions` | 200 | frozen positions for in-loop solver match-rate |
| `--save-every` | 10 | iterations between checkpoints |
| `--checkpoints-dir` | `checkpoints` | |

### Evaluation

Pass `--mcts --sims N` to evaluate with search (the network's true playing
strength); omit it for a cheap raw-policy-argmax probe.

```bash
# Win-rate vs baselines (matches played as both colors to remove first-move bias)
uv run python -m src.evaluate --checkpoint checkpoints/iter_0100.pt --vs random    --games 200
uv run python -m src.evaluate --checkpoint checkpoints/iter_0100.pt --vs heuristic --games 200 --mcts --sims 800

# Move-match-rate vs the perfect-play solver over a frozen 1000-position suite
uv run python -m src.evaluate --checkpoint checkpoints/iter_0100.pt --vs solver --positions 1000 --mcts --sims 800
```

The test-position suite is generated once via random rollouts to varying depths
and cached at `checkpoints/test_positions.pkl`, so the metric is comparable across
checkpoints.

### Play against it

```bash
uv run python -m src.play                              # latest checkpoint, you first
uv run python -m src.play --sims 800 --first ai        # stronger AI, AI moves first
uv run python -m src.play --checkpoint checkpoints/iter_0100.pt
```

The network plays via MCTS (its real strength). The board prints with `X` = player 1,
`O` = player 2, column indices along the bottom; type a column number to drop a piece
(`q` quits). Single-move search is ~1–2 s at 400–800 sims.

Or play in the browser with the Streamlit dashboard (click a column; a side panel shows
the AI's value estimate and the MCTS visit counts behind its last move):

```bash
uv run streamlit run src/app.py
```

It defaults to **CPU** inference so it stays responsive even while a training run uses
the GPU; switch the device to MPS in the sidebar for faster moves when the GPU is free.

**Public version (deployable):** `src/four_in_a_row.py` is a clean, shareable build —
no checkpoint/device pickers or AI-internals panel, just the game plus difficulty,
who-moves-first, and a session scoreboard. It runs on CPU against a bundled
**weights-only** model (`four_in_a_row.pt`, ~9 MB — vs the 152 MB training
checkpoints), so it deploys anywhere (e.g. Streamlit Community Cloud):

```bash
uv run streamlit run src/four_in_a_row.py
```

To refresh the bundled model from a newer checkpoint:
`python -c "import torch; torch.save(torch.load('checkpoints/iter_NNNN.pt', map_location='cpu', weights_only=True)['model'], 'four_in_a_row.pt')"`

## Targets

Done-when (rough targets from a full overnight run, not guarantees):

- vs random: **> 95%** win rate
- vs heuristic: **> 70%** win rate
- solver move-match-rate with ~800 sims: **> 90%**
- As P1, should learn to open in the center column (the solved winning opening)
  and follow standard opening theory.

## Results

Trained **110 iterations** from scratch (50 games/iter, 400 sims, ~2h on M-series MPS;
training loss fell to ~0.45, games lengthened to ~38 of 42 plies — long, well-defended
play). The shipped model is `checkpoints/iter_0110.pt` (the app and terminal player
default to the latest checkpoint). It opens center, blocks threats, and beats casual
human opponents comfortably.

Evaluated by **solver move-match-rate** on a frozen 200-position suite — both the raw
policy (argmax, no search) and the MCTS-backed player (800 sims), the latter across
checkpoints:

| iter | raw policy | MCTS @ 800 sims |
|---|---|---|
| 30 | 46.0% | **68.0%** |
| 70 | 54.0% | **68.0%** |
| 100 | 55.5% | **70.5%** |
| 110 | 48.5% | **67.5%** |

**Key finding — the searched player plateaued at ~68% by iteration ~30 and stayed flat
to 110**, even though training loss and the raw policy kept moving. The lesson: once
MCTS is doing the work, **loss and raw-policy metrics are misleading proxies for
playing strength** — search *compensates* for a weaker early policy, so the player you
actually face can plateau long before the training curves do. More iterations of this
recipe bought the searched player nothing.

Two things bound the ~68%: the **strict metric** (only the solver's *fastest*-winning
move counts as a match, so it understates real strength — the player makes a
winning/non-losing move far more often), and likely **search depth** (800 sims doesn't
always reach the solver's exact best line on tactical positions). It is *not* bounded
by needing more training. Pushing higher would mean **deeper search** at play/eval time
and/or a **different training recipe** (more exploration: less temperature annealing,
more games per iteration) — see below — not simply more iterations.

Checkpoints bundle model + optimizer + replay buffer, so a future run can resume from
the latest `iter_NNNN.pt` by re-running `train.py` with a larger `--iterations`.

## Possible extensions

Deliberately left out, in rough priority order:

- **More MCTS speed**: subtree reuse across moves, and leaf parallelization within a
  single tree (virtual loss). Cross-game batched evaluation is already done (the big
  win); these are the remaining levers. The search still rebuilds the tree each move
  — chosen for clarity over throughput.
- A lightweight web app to play against the trained network (a separate project).
- Tournament ELO tracking across checkpoints; hyperparameter sweeps.

## Acknowledgements

Perfect-play evaluation uses [Pascal Pons's Connect Four solver](https://github.com/PascalPons/connect4)
(vendored under `third_party/connect4-solver/`, see its LICENSE).
</content>
</invoke>
# connect-4-player-net
# connect-4-player-network
