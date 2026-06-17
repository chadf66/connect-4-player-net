"""Streamlit dashboard to play Connect Four against the trained network.

    uv run streamlit run src/app.py

Click a column to drop your piece; the network replies via MCTS. A side panel shows
the AI's "thinking": its value estimate for the position and the visit-count
distribution behind its last move.

Inference defaults to CPU so the app stays responsive even while a training run is
using the GPU (single-move MCTS on this small net is ~1-2s on CPU). Switch the device
to MPS in the sidebar for faster moves when nothing else needs the GPU.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# `streamlit run src/app.py` puts src/ (not the project root) on sys.path, which
# breaks the `from src.… import` package imports. Ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import torch

from src.encode import encode
from src.game import COLS, P1, P2, ROWS, new_game
from src.mcts import MCTSConfig, run_mcts
from src.model import ConnectFourNet

CHECKPOINTS_DIR = Path("checkpoints")
DISC = {0: "⚪", P1: "🔴", P2: "🟡"}  # for text captions
CELL_COLOR = {0: "#fdfdfd", P1: "#e23b2e", P2: "#f4c430"}  # empty hole, player 1, player 2
BOARD_BLUE = "#2d7ff0"
DIFFICULTY = {"Easy": 50, "Medium": 300, "Hard": 800}  # label -> MCTS sims/move


def winning_cells(board) -> set:
    """The cells forming the winning four-in-a-row (empty set if no winner).

    Scans the four directions through the last-played cell for a run of ≥4 of the
    winner's discs.
    """
    if board.winner() is None or not board.history:
        return set()
    col = board.history[-1]
    row = int((board.grid[:, col] != 0).sum()) - 1
    player = int(board.grid[row, col])
    grid = board.grid
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        line = [(row, col)]
        for sign in (1, -1):
            rr, cc = row + sign * dr, col + sign * dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and grid[rr, cc] == player:
                line.append((rr, cc))
                rr += sign * dr
                cc += sign * dc
        if len(line) >= 4:
            return set(line)
    return set()


def render_board_html(board, last_move=None, highlight=None) -> str:
    """Render the board as a blue panel of circular holes/discs (top row first).

    The inner grid carries a fixed 7/6 aspect ratio with explicit rows, so its
    height is derived from its width — the blue panel then wraps all six rows
    instead of the bottom row overflowing.

    `last_move` as (row, col), if given, gets a drop-in animation. The grid clips
    overflow, so a disc starting well above its cell appears to fall from the top of
    the board — lower cells visibly fall farther, like a real piece dropping.
    `highlight` is a set of (row, col) cells to ring with a glow (the winning four).
    """
    highlight = highlight or set()
    inset = "inset 0 -3px 5px rgba(0,0,0,0.28)"
    hole = CELL_COLOR[0]
    cells = []
    for r in range(ROWS - 1, -1, -1):
        for c in range(COLS):
            val = int(board.grid[r, c])
            if val == 0:
                cells.append(f'<div style="border-radius:50%;background:{hole};box-shadow:{inset};"></div>')
            else:
                anim = "animation:drop 0.45s cubic-bezier(0.35,0,0.9,1);" if last_move == (r, c) else ""
                glow = ",0 0 0 3px #fff,0 0 16px 5px #ffd54a" if (r, c) in highlight else ""
                # The white hole stays put; the coloured disc is a separate overlay that
                # drops into it — so a falling disc reveals an open hole, not blue board.
                cells.append(
                    f'<div style="position:relative;border-radius:50%;background:{hole};box-shadow:{inset};">'
                    f'<div style="position:absolute;inset:0;border-radius:50%;'
                    f'background:{CELL_COLOR[val]};box-shadow:{inset}{glow};{anim}"></div></div>'
                )
    return (
        "<style>@keyframes drop{from{transform:translateY(-600%);}to{transform:translateY(0);}}</style>"
        f'<div style="background:{BOARD_BLUE};padding:1.8%;border-radius:16px;'
        'box-shadow:0 4px 12px rgba(0,0,0,0.2);">'
        '<div style="display:grid;aspect-ratio:7/6;overflow:hidden;'
        'grid-template-columns:repeat(7,1fr);grid-template-rows:repeat(6,1fr);gap:3%;">'
        + "".join(cells)
        + "</div></div>"
    )


@st.cache_resource
def load_model(checkpoint_path: str, device_str: str):
    """Load a checkpoint (cached so it isn't re-read on every Streamlit rerun)."""
    device = torch.device(device_str)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ConnectFourNet()
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.to(device).eval()
    return model, device


def ai_move(model, device, board, sims: int) -> tuple[int, np.ndarray, float]:
    """Return (chosen column, visit counts, value estimate) for the AI to move."""
    counts = run_mcts(model, board, device, MCTSConfig(num_simulations=sims), add_noise=False)
    with torch.no_grad():
        _, value = model(encode(board).unsqueeze(0).to(device))
    return int(counts.argmax()), counts, float(value.item())


def start_new_game() -> None:
    """Reset to a fresh game. If the AI moves first, the main loop makes its move
    on the next cycle (so its opening drop animates like any other move)."""
    st.session_state.board = new_game()
    st.session_state.ai_counts = None
    st.session_state.ai_value = None
    st.session_state.counted = False  # this game's result not yet added to the scoreboard


def main() -> None:
    st.set_page_config(page_title="Connect Four AI", page_icon="🔴")
    st.title("🔴 Connect Four — play the network")

    checkpoints = sorted(CHECKPOINTS_DIR.glob("iter_*.pt"))
    if not checkpoints:
        st.error("No checkpoints found in `checkpoints/`. Train a model first.")
        return
    if "score" not in st.session_state:
        st.session_state.score = {"you": 0, "ai": 0, "draw": 0}

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Settings")
        ckpt = st.selectbox("Checkpoint", checkpoints, index=len(checkpoints) - 1, format_func=lambda p: p.name)
        device_str = st.selectbox("Device", ["cpu", "mps"], index=0,
                                  help="CPU keeps the app responsive while a training run uses the GPU.")
        diff = st.radio("Difficulty", list(DIFFICULTY), index=1, horizontal=True,
                        help="More MCTS simulations per move = stronger (and slower).")
        sims = DIFFICULTY[diff]
        first = st.radio("Who moves first?", ["You", "AI"], horizontal=True)

    # Load the chosen checkpoint, tolerating a partially-written file (training may be
    # saving it right now) by falling back to the newest readable one.
    try:
        model, device = load_model(str(ckpt), device_str)
    except (EOFError, RuntimeError, zipfile.BadZipFile):
        st.warning(f"Couldn't read {ckpt.name} (it may be mid-write during training); "
                   "using the newest readable checkpoint.")
        model = None
        for alt in reversed(checkpoints):
            if alt == ckpt:
                continue
            try:
                model, device = load_model(str(alt), device_str)
                ckpt = alt
                break
            except (EOFError, RuntimeError, zipfile.BadZipFile):
                continue
        if model is None:
            st.error("No readable checkpoint found.")
            return
    ai_player = P2 if first == "You" else P1

    # (Re)start a game when needed: first load, or the first-move choice changed.
    if "board" not in st.session_state or st.session_state.get("ai_player") != ai_player:
        st.session_state.ai_player = ai_player
        start_new_game()

    if st.sidebar.button("New game", use_container_width=True):
        st.session_state.ai_player = ai_player
        start_new_game()
        st.rerun()

    board = st.session_state.board
    human_player = P1 if ai_player == P2 else P2
    ai_turn = (not board.terminal()) and board.current_player == ai_player
    st.caption(f"You are {DISC[human_player]} · AI is {DISC[ai_player]} · "
               f"{diff} ({sims} sims) on {device_str.upper()}")

    # --- Drop buttons: a human move only; disabled while it isn't your turn. ---
    btn_cols = st.columns(COLS)
    for c in range(COLS):
        if btn_cols[c].button("⬇", key=f"drop{c}",
                              disabled=ai_turn or board.terminal() or not board.is_legal(c),
                              use_container_width=True):
            st.session_state.board = board.play(c)
            st.rerun()

    # --- Board: the most recent move animates dropping in; the winning four glows. ---
    last_move = None
    if board.history:
        lc = board.history[-1]
        lr = int((board.grid[:, lc] != 0).sum()) - 1  # topmost filled cell in that column
        last_move = (lr, lc)
    st.markdown(render_board_html(board, last_move, winning_cells(board)), unsafe_allow_html=True)

    # --- Result banner + scoreboard tally (counted once per finished game) ---
    if board.terminal():
        winner = board.winner()
        if not st.session_state.get("counted", False):
            key = "draw" if winner is None else ("you" if winner == human_player else "ai")
            st.session_state.score[key] += 1
            st.session_state.counted = True
        if winner is None:
            st.info("Draw.")
        elif winner == human_player:
            st.success("You win! 🎉")
        else:
            st.error("The network wins.")

    # --- Session scoreboard (sidebar) ---
    s = st.session_state.score
    st.sidebar.divider()
    sc1, sc2, sc3 = st.sidebar.columns(3)
    sc1.metric("You", s["you"])
    sc2.metric("AI", s["ai"])
    sc3.metric("Draws", s["draw"])

    # --- "Dashboard": what the AI was thinking on its last move ---
    if st.session_state.ai_value is not None:
        st.divider()
        st.subheader("AI's last move")
        val = st.session_state.ai_value
        st.metric("Value estimate (AI's POV)", f"{val:+.2f}",
                  help="The value head's score for the position after the AI moved, "
                       "from the AI's perspective: +1 = winning, -1 = losing.")
        counts = st.session_state.ai_counts
        st.caption("MCTS visit counts per column (where the search spent its time):")
        st.bar_chart({"visits": {str(c): float(counts[c]) for c in range(COLS)}})

    # --- AI move: computed *after* the board (with your disc) is already on screen,
    #     so your move shows first; the AI's disc then drops in on the next rerun. ---
    if ai_turn:
        with st.spinner("AI is thinking…"):
            ai_col, counts, value = ai_move(model, device, board, sims)
        st.session_state.board = board.play(ai_col)
        st.session_state.ai_counts, st.session_state.ai_value = counts, value
        st.rerun()


if __name__ == "__main__":
    main()
