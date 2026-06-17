"""Four in a Row — a clean, public-facing web app to play the trained network.

    uv run streamlit run src/four_in_a_row.py

A stripped-down sibling of `app.py`: no checkpoint picker, device selection, or
"what the AI is thinking" panel — just the game, plus difficulty, who-moves-first,
and a session scoreboard. Inference runs on CPU against a bundled, weights-only model
file (~9 MB), so it deploys anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/four_in_a_row.py` puts src/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch

from src.game import COLS, P1, P2, ROWS, new_game
from src.mcts import MCTSPlayer
from src.model import ConnectFourNet

MODEL_PATH = Path(__file__).resolve().parent.parent / "four_in_a_row.pt"
DEVICE = torch.device("cpu")
DISC = {0: "⚪", P1: "🔴", P2: "🟡"}
CELL_COLOR = {0: "#fdfdfd", P1: "#e23b2e", P2: "#f4c430"}  # empty hole, player 1, player 2
SCORE_COLOR = {P1: "#e23b2e", P2: "#d4a017"}  # scoreboard number text (gold darkened for contrast)
BOARD_BLUE = "#2d7ff0"
DIFFICULTY = {"Easy": 50, "Medium": 300, "Hard": 800}  # label -> search effort


def winning_cells(board) -> set:
    """The cells forming the winning four-in-a-row (empty set if no winner)."""
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
    """Render the board: a blue panel of circular holes/discs. The last move drops
    in (animated, clipped to fall from the top); the winning four gets a gold glow."""
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
def load_model():
    """Load the bundled weights-only model (cached across reruns)."""
    model = ConnectFourNet()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model


def scorecard_html(score, human_player, ai_player) -> str:
    """A compact scoreboard coloured by each side's actual disc (depends on who's first)."""
    items = [(f"{DISC[human_player]} You", score["you"], SCORE_COLOR[human_player]),
             (f"{DISC[ai_player]} Computer", score["ai"], SCORE_COLOR[ai_player]),
             ("Draws", score["draw"], "#9aa0a6")]
    cells = "".join(
        '<div style="flex:1;text-align:center;">'
        f'<div style="font-size:0.7rem;letter-spacing:0.03em;color:#8a8f98;white-space:nowrap;">{label}</div>'
        f'<div style="font-size:1.9rem;font-weight:800;line-height:1.2;color:{color};">{val}</div>'
        '</div>'
        for label, val, color in items
    )
    return f'<div style="display:flex;gap:0.4rem;margin-bottom:1.5rem;">{cells}</div>'


def start_new_game() -> None:
    st.session_state.board = new_game()
    st.session_state.counted = False  # result not yet added to the scoreboard


def main() -> None:
    st.set_page_config(page_title="Four in a Row", page_icon="🔴")
    st.title("🔴 Four in a Row 🟡")
    st.markdown(
        "<p style='margin-top:-0.6rem;color:#6b7280;font-size:1.05rem;'>"
        "Can you beat an AI that taught itself to play?</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<style>"
        # Cap the main column so the board fits the viewport height (board height =
        # width*6/7); buttons share this width, so they stay aligned. 72vh width caps
        # the board near ~62vh tall, leaving room for the title/buttons above it.
        ".block-container,[data-testid='stMainBlockContainer']"
        "{max-width:min(560px,72vh)!important;}"
        # Bigger, chunkier "New game" button in the sidebar.
        "section[data-testid='stSidebar'] div[data-testid='stButton'] button p"
        "{font-size:1.15rem;font-weight:700;}"
        "section[data-testid='stSidebar'] div[data-testid='stButton'] button"
        "{padding:0.55rem 0;}"
        "</style>",
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("Model file `four_in_a_row.pt` not found.")
        return
    if "score" not in st.session_state:
        st.session_state.score = {"you": 0, "ai": 0, "draw": 0}

    # --- Sidebar: game preferences ---
    with st.sidebar:
        st.header("Game")
        diff = st.segmented_control("Difficulty", list(DIFFICULTY), default="Medium",
                                    selection_mode="single") or "Medium"
        first = st.segmented_control("Who moves first?", ["You", "Computer"], default="You",
                                     selection_mode="single") or "You"
        new_clicked = st.button("New game", use_container_width=True, type="primary")

    sims = DIFFICULTY[diff]
    ai_player = P2 if first == "You" else P1
    human_player = P1 if ai_player == P2 else P2

    model = load_model()

    if new_clicked or "board" not in st.session_state or st.session_state.get("ai_player") != ai_player:
        st.session_state.ai_player = ai_player
        start_new_game()
        if new_clicked:
            st.rerun()

    board = st.session_state.board
    ai_turn = (not board.terminal()) and board.current_player == ai_player
    st.caption(f"You are {DISC[human_player]} · Computer is {DISC[ai_player]} · first to four in a row wins")

    # --- Drop buttons (your move only; disabled while it isn't your turn) ---
    btn_cols = st.columns(COLS)
    for c in range(COLS):
        if btn_cols[c].button("⬇", key=f"drop{c}",
                              disabled=ai_turn or board.terminal() or not board.is_legal(c),
                              use_container_width=True):
            st.session_state.board = board.play(c)
            st.rerun()

    # --- Board (last move animates; winning four glows) ---
    last_move = None
    if board.history:
        lc = board.history[-1]
        lr = int((board.grid[:, lc] != 0).sum()) - 1
        last_move = (lr, lc)
    st.markdown(render_board_html(board, last_move, winning_cells(board)), unsafe_allow_html=True)

    # --- Result banner + scoreboard tally (once per finished game) ---
    if board.terminal():
        winner = board.winner()
        if not st.session_state.get("counted", False):
            key = "draw" if winner is None else ("you" if winner == human_player else "ai")
            st.session_state.score[key] += 1
            st.session_state.counted = True
        if winner is None:
            st.info("It's a draw.")
        elif winner == human_player:
            st.success("You win! 🎉")
        else:
            st.error("The computer wins.")

    # --- Session scoreboard (sidebar) ---
    st.sidebar.divider()
    st.sidebar.markdown(scorecard_html(st.session_state.score, human_player, ai_player),
                        unsafe_allow_html=True)

    # --- How to play (sidebar, keeps the main column free for the board) ---
    with st.sidebar.expander("How to play"):
        st.markdown(
            "- **Goal:** be the first to get **four of your discs in a row** — "
            "horizontally, vertically, or diagonally.\n"
            "- **Your move:** click a column's **⬇** button to drop a disc; it falls to the "
            "lowest open slot.\n"
            "- The **computer replies automatically** after each of your moves.\n"
            "- Pick **who moves first** and the **difficulty** above; **New game** starts over.\n"
            "- Your opponent is a neural network that learned entirely by playing against "
            "itself — no human games — guided by Monte Carlo Tree Search."
        )

    # --- Computer's move: made after your disc is on screen, so moves are sequenced ---
    if ai_turn:
        with st.spinner("Thinking…"):
            move = MCTSPlayer(model, DEVICE, num_simulations=sims).choose(board)
        st.session_state.board = board.play(move)
        st.rerun()


if __name__ == "__main__":
    main()
