import torch

from src.encode import (
    encode,
    encode_batch,
    legal_mask,
    legal_mask_batch,
    mirror_move,
    mirror_policy,
    mirror_state,
)
from src.game import COLS, P1, P2, ROWS, new_game


def test_encode_shape_and_dtype():
    state = encode(new_game())
    assert state.shape == (3, ROWS, COLS)
    assert state.dtype == torch.float32


def test_encode_empty_board():
    state = encode(new_game())
    assert state[0].sum().item() == 0  # no current-player pieces
    assert state[1].sum().item() == 0  # no opponent pieces
    assert state[2].sum().item() == ROWS * COLS  # everything empty


def test_encode_current_player_perspective_after_one_move():
    board = new_game().play(3)  # P1 plays col 3 -> now P2 to move
    state = encode(board)
    # From P2's perspective, the piece at (0,3) is the OPPONENT's piece.
    assert state[0, 0, 3] == 0
    assert state[1, 0, 3] == 1
    assert state[2, 0, 3] == 0


def test_encode_perspective_flips_each_move():
    board = new_game().play(3).play(4)  # P1 col 3, P2 col 4; now P1 to move
    state = encode(board)
    # From P1's perspective again: (0,3) is me, (0,4) is opp.
    assert state[0, 0, 3] == 1
    assert state[1, 0, 4] == 1


def test_encode_channels_sum_to_one_per_cell():
    board = new_game().play(0).play(1).play(2).play(3)
    state = encode(board)
    # Every cell belongs to exactly one of the three channels.
    assert torch.allclose(state.sum(dim=0), torch.ones(ROWS, COLS))


def test_encode_batch():
    boards = [new_game(), new_game().play(3), new_game().play(3).play(4)]
    states = encode_batch(boards)
    assert states.shape == (3, 3, ROWS, COLS)


def test_mirror_state_round_trip():
    state = encode(new_game().play(0).play(6).play(1))
    twice = mirror_state(mirror_state(state))
    assert torch.equal(state, twice)


def test_mirror_state_swaps_columns():
    board = new_game().play(0)  # P1 at (0,0), now P2 to move
    state = encode(board)
    mirrored = mirror_state(state)
    # Original: opp piece at col 0. Mirrored: opp piece at col COLS-1.
    assert state[1, 0, 0] == 1
    assert mirrored[1, 0, COLS - 1] == 1
    assert mirrored[1, 0, 0] == 0


def test_mirror_policy():
    p = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    mp = mirror_policy(p)
    assert torch.allclose(mp, torch.tensor([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))


def test_mirror_move():
    assert mirror_move(0) == COLS - 1
    assert mirror_move(COLS - 1) == 0
    assert mirror_move(3) == 3  # center is symmetric


def test_mirror_consistency_between_state_and_move():
    # If we mirror the encoded board and also mirror the move that was played,
    # the resulting (state, move) pair should be a valid Connect Four position
    # that's just the left-right reflection of the original.
    board = new_game().play(1)  # P1 at (0,1)
    state = encode(board)
    mirrored_state = mirror_state(state)
    move = 1
    mirrored = mirror_move(move)
    # In the mirrored encoding, the opponent (from current POV) piece is at the mirrored column.
    assert mirrored_state[1, 0, mirrored] == 1


def test_legal_mask_full_board():
    mask = legal_mask(new_game())
    assert mask.all()
    assert mask.shape == (COLS,)


def test_legal_mask_full_column():
    board = new_game()
    for _ in range(ROWS):
        board = board.play(3)
    mask = legal_mask(board)
    assert not mask[3]
    assert mask[0] and mask[1] and mask[2]  # other columns still open


def test_legal_mask_batch():
    boards = [new_game(), new_game().play(3)]
    masks = legal_mask_batch(boards)
    assert masks.shape == (2, COLS)
    assert masks.all()  # nothing's full yet
