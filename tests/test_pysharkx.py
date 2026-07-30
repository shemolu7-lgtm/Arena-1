import time

import chess

from pysharkx import (
    INF,
    MATE_THRESHOLD,
    MATE_VALUE,
    Engine,
    _position_key,
    _score_from_tt,
    _score_to_tt,
    static_exchange_eval,
)


def test_see_is_non_mutating_and_handles_special_captures() -> None:
    cases = [
        # Ordinary exchange with an optional rook recapture sequence.
        ("3r2k1/8/8/3p4/4P3/8/8/3R2K1 w - - 0 1", "e4d5", 100),
        # En passant must remove the pawn behind the destination square.
        ("6k1/8/8/3pP3/8/8/8/6K1 w - d6 0 1", "e5d6", 100),
        # Capture plus queen-promotion material gain.
        ("k6r/6P1/8/8/8/8/8/K7 w - - 0 1", "g7h8q", 1300),
    ]

    for fen, uci, expected in cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        original_fen = board.fen()
        original_stack = list(board.move_stack)

        assert move in board.legal_moves
        assert static_exchange_eval(board, move) == expected
        assert board.fen() == original_fen
        assert board.move_stack == original_stack


def test_see_ignores_a_pinned_recapturing_pawn() -> None:
    board = chess.Board("4k3/4p3/3p4/2P5/8/8/8/4R1K1 w - - 0 1")
    move = chess.Move.from_uci("c5d6")

    # e7xd6 would expose the king on e8 to the rook, so White wins the pawn.
    assert move in board.legal_moves
    assert static_exchange_eval(board, move) == 100


def test_quiescence_scores_checkmate_and_stalemate_as_terminals() -> None:
    engine = Engine(use_book=False)
    engine.stop_time = float("inf")

    mate_in_one = chess.Board("7k/6p1/5KQ1/8/8/8/8/8 w - - 0 1")
    assert engine.quiescence(mate_in_one, -INF, INF, 0) == MATE_VALUE - 1

    stalemate = chess.Board("k7/8/1QK5/8/8/8/8/8 b - - 0 1")
    assert stalemate.is_stalemate()
    assert engine.quiescence(stalemate, -INF, INF, 0) == 0

    mate_at_fifty_move_boundary = chess.Board(
        "7k/6Q1/5K2/8/8/8/8/8 b - - 100 1"
    )
    assert mate_at_fifty_move_boundary.is_checkmate()
    assert (
        engine.quiescence(mate_at_fifty_move_boundary, -INF, INF, 1)
        == -MATE_VALUE + 1
    )


def test_king_shield_evaluation_is_color_symmetric() -> None:
    engine = Engine(use_book=False)
    white_to_move = chess.Board(
        "3q2k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"
    )
    black_to_move = chess.Board(
        "3q2k1/5ppp/8/8/8/8/5PPP/3Q2K1 b - - 0 1"
    )

    # Equal shield structures should leave only the side-to-move tempo bonus.
    assert engine.evaluate(white_to_move) == 9
    assert engine.evaluate(black_to_move) == 9


def test_tt_mate_scores_are_normalized_for_the_probe_ply() -> None:
    for score in (MATE_VALUE - 7, -MATE_VALUE + 7):
        stored = _score_to_tt(score, 7)
        restored = _score_from_tt(stored, 3)
        assert abs(stored) == MATE_VALUE
        assert abs(restored) >= MATE_THRESHOLD
        assert restored == (MATE_VALUE - 3 if score > 0 else -MATE_VALUE + 3)

    assert _score_from_tt(_score_to_tt(123, 12), 2) == 123


def test_position_key_includes_fifty_move_state() -> None:
    fresh = chess.Board("8/8/8/8/8/4k3/8/4K2R w - - 0 1")
    near_draw = chess.Board("8/8/8/8/8/4k3/8/4K2R w - - 99 1")

    assert _position_key(fresh) != _position_key(near_draw)


def test_transposition_table_generations_stay_inside_capacity() -> None:
    engine = Engine(tt_size_mb=1, use_book=False)
    for key in range(engine.tt_max_entries * 2):
        engine._store_tt(key, 1, key % 100, 0, None, 0)

    assert len(engine.tt_current) <= engine.tt_generation_entries
    assert len(engine.tt_previous) <= engine.tt_generation_entries
    assert len(engine.tt_current) + len(engine.tt_previous) <= engine.tt_max_entries


def test_timed_search_returns_a_legal_move_without_mutating_board() -> None:
    board = chess.Board()
    original_fen = board.fen()
    engine = Engine(use_book=False)

    started = time.monotonic()
    move, info = engine.search(board, move_time=0.02)
    elapsed = time.monotonic() - started

    assert move in board.legal_moves
    assert board.fen() == original_fen
    assert info["nodes"] > 0
    # Allow slow/loaded CI substantial scheduling headroom while guarding
    # against the former unbounded wall-clock deadline behavior.
    assert elapsed < 1.0
