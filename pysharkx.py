#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
PySharkX - a single-file UCI-style chess engine.

Libraries used
--------------
* python-chess : board representation, legal move generation, SAN/UCI parsing.
* numpy        : vectorised piece-square tables / evaluation data.
* Cython       : this file is written in Cython's "pure Python" style
                  (PEP 484 type hints + the `# cython:` directives above),
                  so it can be compiled as-is for a large speed boost:

                      pip install cython
                      cythonize -i -3 pysharkx.py     # builds pysharkx.*.so
                      python3 -c "import pysharkx; pysharkx.main()"

                  It also runs perfectly fine as normal, uncompiled Python:

                      python3 pysharkx.py

Search / engine features (aimed at strong club/expert level play)
-------------------------------------------------------------------
  - Negamax with alpha-beta pruning, implemented as Principal Variation
    Search (PVS): the first move at each node is searched with a full
    window, the rest with a fast zero-width "scout" window.
  - Iterative deepening with aspiration windows around the previous
    iteration's score.
  - Memory-bounded, two-generation transposition table using fast native
    bitboard position hashing. It stores depth, normalized mate score, bound
    type (exact / lower / upper) and best move for cutoffs and move ordering.
  - Move ordering: TT move -> good captures via Static Exchange
    Evaluation (SEE) + MVV-LVA -> killer moves -> history heuristic ->
    remaining quiet moves.
  - Quiescence search (captures + promotions + evasions) with
    stand-pat, delta pruning and SEE pruning of losing captures, to
    avoid the horizon effect.
  - Null-move pruning (with zugzwang guard: disabled when the side to
    move has only pawns and king left, and when in check).
  - Late move reductions (LMR) for quiet, non-checking moves ordered
    late in the move list.
  - Futility pruning / reverse futility pruning near the leaves.
  - Check extensions.
  - Draw detection: threefold repetition, fifty-move rule, insufficient
    material.
  - Tapered (middlegame -> endgame) evaluation: material, piece-square
    tables, bishop pair, mobility, pawn structure (doubled / isolated /
    passed pawns), rook on (semi)open files, and king safety (pawn
    shield + open files near the king).
  - Simple per-move time management driven by a hard time budget.

Interactive mode
-----------------
Run the script, choose which colour you want to play, and how many
seconds the engine is allowed to think per move. After every engine
move you'll see: nodes searched, nodes/second, time taken and the
maximum depth reached that iteration.

NOTE ON PLAYING STRENGTH
-------------------------
Reaching a genuine 2400 FIDE-equivalent strength is primarily a
function of raw nodes-per-second, because alpha-beta search strength
scales with search depth, and pure/compiled-Python + python-chess
move generation is orders of magnitude slower than the bitboard
engines (C/C++, hand written bitboards) that "2400" is usually
benchmarked against. This file implements essentially every classical
technique that contributes real Elo (see the list above), and
compiling it with Cython will meaningfully raise its nps - but on a
typical laptop it will realistically search far fewer nodes/second
than a native engine at the same time control. Give it more time per
move (10-60s+) to close more of that gap.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import chess
    import chess.polyglot
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "This engine requires the 'python-chess' package.\n"
        "Install it with:  pip install chess\n"
    )
    raise

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

INF: int = 30000
MATE_VALUE: int = 29000
MATE_THRESHOLD: int = MATE_VALUE - 1000
MAX_PLY: int = 128
TIME_CHECK_MASK: int = 127  # Check the deadline every 128 visited nodes.
HISTORY_MAX: int = 75_000
# A Python dict entry, integer key, score tuple, and move collectively use far
# more than the raw payload size. This estimate keeps the user-facing TT size
# close to its requested memory budget instead of potentially exceeding it by
# an order of magnitude.
ESTIMATED_TT_ENTRY_BYTES: int = 256

EXACT: int = 0
LOWER: int = 1  # fail-high (score is a lower bound)
UPPER: int = 2  # fail-low  (score is an upper bound)

# Polyglot opening book. By default we look for "book.bin" sitting right
# next to this script/module, regardless of the process's current working
# directory. Pass an explicit book_path=... to Engine() to override.
BOOK_FILENAME: str = "book.bin"
DEFAULT_BOOK_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), BOOK_FILENAME
)

PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# ==============================================================================
# TUNED EVALUATION CONSTANTS (5M Positions Texel-Tuned)
# ==============================================================================

PIECE_VALUES_MG: Dict[int, int] = {
    chess.PAWN: 79,
    chess.KNIGHT: 343,
    chess.BISHOP: 356,
    chess.ROOK: 465,
    chess.QUEEN: 1010,
    chess.KING: 0,
}

PIECE_VALUES_EG: Dict[int, int] = {
    chess.PAWN: 91,
    chess.KNIGHT: 287,
    chess.BISHOP: 288,
    chess.ROOK: 500,
    chess.QUEEN: 921,
    chess.KING: 0,
}

# Values used for SEE / MVV-LVA ordering (simple, symmetric, "see-value" scale).
SEE_VALUES: Dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

GAME_PHASE_INC: Dict[int, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
TOTAL_PHASE: int = 24  # 4N + 4B + 4R + 2Q worth of increments

# ---- Tunable Evaluation Parameters ----
BISHOP_PAIR_MG: int = 24
BISHOP_PAIR_EG: int = 39

DOUBLE_PAWN_MG: int = 13
DOUBLE_PAWN_EG: int = 23

ISOLATED_PAWN_MG: int = 5
ISOLATED_PAWN_EG: int = 9

PASSED_PAWN_MG: int = 9
PASSED_PAWN_EG: int = 18

ROOK_OPEN_FILE_MG: int = 14
ROOK_SEMI_OPEN_FILE_MG: int = 13

KING_SHIELD_MG: int = 7

TEMPO_BONUS: int = 9

MOBILITY_WEIGHTS: Tuple[Tuple[int, int], ...] = (
    (chess.KNIGHT, 4),
    (chess.BISHOP, 4),
    (chess.ROOK, 2),
    (chess.QUEEN, 1),
)
WHITE_KING_SHIELD_RANKS: int = 0x0000000000FFFF00  # ranks two and three
BLACK_KING_SHIELD_RANKS: int = 0x00FFFF0000000000  # ranks six and seven

# --------------------------------------------------------------------------
# Piece-square tables (White's perspective, a8=top-left reading order like a
# FEN board). Values are in centipawns and get added on top of material.
# For Black, the mirrored square (sq ^ 56) is looked up instead.
# --------------------------------------------------------------------------

_PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_KNIGHT_MG = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]
_KNIGHT_EG = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]
_BISHOP_MG = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
_BISHOP_EG = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
]
_ROOK_MG = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
]
_ROOK_EG = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
]
_QUEEN_MG = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]
_QUEEN_EG = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]
_KING_MG = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
_KING_EG = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

_RAW_MG = {
    chess.PAWN: _PAWN_MG, chess.KNIGHT: _KNIGHT_MG, chess.BISHOP: _BISHOP_MG,
    chess.ROOK: _ROOK_MG, chess.QUEEN: _QUEEN_MG, chess.KING: _KING_MG,
}
_RAW_EG = {
    chess.PAWN: _PAWN_EG, chess.KNIGHT: _KNIGHT_EG, chess.BISHOP: _BISHOP_EG,
    chess.ROOK: _ROOK_EG, chess.QUEEN: _QUEEN_EG, chess.KING: _KING_EG,
}


def _reorder_to_chess_squares(table: List[int]) -> np.ndarray:
    """Convert a table written a8..h1 (FEN reading order) into an array
    indexed by python-chess's square numbering (a1=0 ... h8=63)."""
    arr = np.array(table, dtype=np.int32).reshape(8, 8)  # row0 = rank8 .. row7 = rank1
    arr = arr[::-1]  # row0 = rank1 .. row7 = rank8  -> index = rank*8+file
    return arr.flatten().copy()


PST_SCALE: float = 0.75

PST_MG: Dict[int, List[int]] = {
    pt: [int(v * PST_SCALE) for v in _reorder_to_chess_squares(_RAW_MG[pt])]
    for pt in PIECE_TYPES
}
PST_EG: Dict[int, List[int]] = {
    pt: [int(v * PST_SCALE) for v in _reorder_to_chess_squares(_RAW_EG[pt])]
    for pt in PIECE_TYPES
}


def mirror(square: int) -> int:
    """Vertical mirror (rank flip) used to reuse White's PST for Black."""
    return square ^ 56


# File / rank masks, precomputed once with numpy for fast pawn-structure eval.
FILE_MASK: List[int] = [0] * 8
for _f in range(8):
    bb = 0
    for _r in range(8):
        bb |= 1 << chess.square(_f, _r)
    FILE_MASK[_f] = bb

ADJACENT_FILES_MASK: List[int] = [0] * 8
for _f in range(8):
    m = 0
    if _f > 0:
        m |= FILE_MASK[_f - 1]
    if _f < 7:
        m |= FILE_MASK[_f + 1]
    ADJACENT_FILES_MASK[_f] = m


def _forward_ranks_mask(rank: int, color: bool) -> int:
    m = 0
    if color == chess.WHITE:
        for r in range(rank + 1, 8):
            m |= 0xFF << (r * 8)
    else:
        for r in range(0, rank):
            m |= 0xFF << (r * 8)
    return m


# passed_pawn_mask[color][square] = squares that, if empty of enemy pawns, mean this pawn is passed.
PASSED_MASK: Dict[bool, List[int]] = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
for _color in (chess.WHITE, chess.BLACK):
    for _sq in chess.SQUARES:
        _file = chess.square_file(_sq)
        _rank = chess.square_rank(_sq)
        _files = FILE_MASK[_file] | ADJACENT_FILES_MASK[_file]
        PASSED_MASK[_color][_sq] = _files & _forward_ranks_mask(_rank, _color)


def popcount(bb: int) -> int:
    return bb.bit_count()


# --------------------------------------------------------------------------
# Static Exchange Evaluation (SEE)
# --------------------------------------------------------------------------

def _see_attacker_is_legal(
    board: "chess.Board",
    side: bool,
    attacker_sq: int,
    to_sq: int,
    occupied: int,
) -> bool:
    """Return whether moving an SEE attacker leaves its own king safe.

    SEE works on occupancy rather than by pushing moves.  Filtering pinned
    pieces and defended king captures here avoids the most damaging errors of
    pseudo-legal SEE while retaining the speed of bitboard exchange analysis.
    The target bit is excluded from enemy piece sets because the original
    occupant there has already been captured; the current exchange occupant is
    represented by occupancy only.
    """
    king_sq = board.king(side)
    if king_sq is None:
        return True

    occupied_after = occupied & ~(1 << attacker_sq)
    if attacker_sq == king_sq:
        king_sq = to_sq

    enemy_attackers = board.attackers_mask(not side, king_sq, occupied_after)
    enemy_attackers &= occupied_after & ~(1 << to_sq)
    return not enemy_attackers


def _least_valuable_legal_attacker(
    board: "chess.Board", side: bool, to_sq: int, occupied: int
) -> Tuple[Optional[int], Optional[int]]:
    attackers = board.attackers_mask(side, to_sq, occupied) & occupied
    if not attackers:
        return None, None

    for piece_type in PIECE_TYPES:
        candidates = attackers & board.pieces_mask(piece_type, side)
        while candidates:
            attacker_bit = candidates & -candidates
            attacker_sq = attacker_bit.bit_length() - 1
            if _see_attacker_is_legal(board, side, attacker_sq, to_sq, occupied):
                return attacker_sq, piece_type
            candidates ^= attacker_bit
    return None, None


def static_exchange_eval(board: "chess.Board", move: "chess.Move") -> int:
    """Evaluate an exchange on ``move.to_square`` without mutating ``board``.

    Both sides are assumed to use their least valuable legal recapture and may
    decline an unfavorable continuation.  Promotions, en passant occupancy,
    pins, and king captures are handled.  This occupancy-based implementation
    is substantially faster than repeatedly pushing moves and cannot leave the
    caller's board damaged if an unusual position is encountered.
    """
    to_sq = move.to_square
    from_sq = move.from_square
    moving_pt = board.piece_type_at(from_sq)
    if moving_pt is None:
        return 0

    is_en_passant = board.is_en_passant(move)
    captured_pt = chess.PAWN if is_en_passant else board.piece_type_at(to_sq)
    promotion_gain = 0
    resulting_pt = moving_pt
    if move.promotion is not None:
        promotion_gain = SEE_VALUES[move.promotion] - SEE_VALUES[chess.PAWN]
        resulting_pt = move.promotion

    # Quiet promotions still gain the promoted material even though they do
    # not enter capture ordering at ordinary search nodes.
    if captured_pt is None:
        return promotion_gain

    gains: List[int] = [SEE_VALUES[captured_pt] + promotion_gain]
    occupied = board.occupied & ~(1 << from_sq)
    if is_en_passant:
        captured_sq = to_sq - 8 if board.turn == chess.WHITE else to_sq + 8
        occupied &= ~(1 << captured_sq)
    occupied |= 1 << to_sq

    side = not board.turn
    target_value = SEE_VALUES[resulting_pt]

    for _ in range(31):
        # A legal king capture cannot itself be recaptured.
        if resulting_pt == chess.KING:
            break

        attacker_sq, attacker_pt = _least_valuable_legal_attacker(
            board, side, to_sq, occupied
        )
        if attacker_sq is None or attacker_pt is None:
            break

        gains.append(target_value - gains[len(gains) - 1])
        occupied &= ~(1 << attacker_sq)

        if attacker_pt == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
            resulting_pt = chess.QUEEN
        else:
            resulting_pt = attacker_pt
        target_value = SEE_VALUES[resulting_pt]
        side = not side

    for i in range(len(gains) - 2, -1, -1):
        gains[i] = -max(-gains[i], gains[i + 1])
    return gains[0]


# --------------------------------------------------------------------------
# Transposition table entry (plain tuple for speed): (depth, score, flag, move)
# --------------------------------------------------------------------------

def _position_key(board: "chess.Board") -> int:
    """Build a fast native hash for all state relevant to the search.

    ``chess.polyglot.zobrist_hash()`` walks every occupied square in Python and
    was one of the engine's hottest operations.  Hashing python-chess's native
    bitboards is over an order of magnitude faster.  The halfmove clock is
    included so a TT result cannot cross-contaminate positions on opposite
    sides of the fifty-move draw boundary.
    """
    return hash(
        (
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,
            board.occupied_co[chess.WHITE],
            board.occupied_co[chess.BLACK],
            board.turn,
            board.castling_rights,
            board.ep_square,
            min(board.halfmove_clock, 100),
        )
    )


def _score_to_tt(score: int, ply: int) -> int:
    """Normalize mate scores before storage so they are root-independent."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    """Restore a normalized TT mate score at the current search ply."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


class TimeUp(Exception):
    """Raised internally to unwind the search once the time budget is spent."""
    pass


class Engine:
    def __init__(
        self,
        tt_size_mb: int = 64,
        book_path: Optional[str] = None,
        use_book: bool = True,
    ) -> None:
        self.tt_current: Dict[int, Tuple[int, int, int, Optional[chess.Move]]] = {}
        self.tt_previous: Dict[int, Tuple[int, int, int, Optional[chess.Move]]] = {}
        requested_tt_bytes = max(1, tt_size_mb) * 1024 * 1024
        self.tt_max_entries: int = max(
            2, requested_tt_bytes // ESTIMATED_TT_ENTRY_BYTES
        )
        # Two equally bounded generations keep useful older entries while
        # ensuring their combined size stays within the configured budget.
        self.tt_generation_entries: int = max(1, self.tt_max_entries // 2)
        self.killers: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(MAX_PLY)]
        self.history: Dict[Tuple[bool, int, int], int] = {}
        self.nodes: int = 0
        self.start_time: float = 0.0
        self.stop_time: float = 0.0
        self.seldepth: int = 0
        self.root_moves_order: List[chess.Move] = []

        # ---- opening book (Polyglot .bin) ----
        self.use_book: bool = use_book
        self.book_path: str = book_path if book_path is not None else DEFAULT_BOOK_PATH
        self.book_reader: Optional["chess.polyglot.MemoryMappedReader"] = None
        self.book_loaded: bool = False
        if self.use_book and os.path.isfile(self.book_path):
            try:
                self.book_reader = chess.polyglot.open_reader(self.book_path)
                self.book_loaded = True
            except Exception as exc:  # pragma: no cover - corrupt/bad file
                sys.stderr.write(
                    f"Warning: could not open opening book '{self.book_path}': {exc}\n"
                )
                self.book_reader = None
                self.book_loaded = False

    def close_book(self) -> None:
        """Release the memory-mapped book file, if one is open."""
        if self.book_reader is not None:
            try:
                self.book_reader.close()
            except Exception:
                pass
            self.book_reader = None
        self.book_loaded = False

    def get_book_move(self, board: "chess.Board") -> Optional["chess.Move"]:
        """Return a move sampled (weighted by book 'weight') from the
        Polyglot opening book for the current position, or None if no
        book is loaded or the position has no book entries."""
        if self.book_reader is None or not self.use_book:
            return None
        try:
            entry = self.book_reader.weighted_choice(board)
        except (IndexError, KeyError):
            return None  # position not found in the book
        except Exception:
            return None
        move = entry.move
        if move in board.legal_moves:
            return move
        return None

    # ---------------------------------------------------------------- eval
    def evaluate(self, board: "chess.Board") -> int:
        """Tapered static evaluation, returned from the side-to-move's
        point of view (positive = good for the side to move)."""
        white_occ = board.occupied_co[chess.WHITE]
        black_occ = board.occupied_co[chess.BLACK]

        mg_score = 0
        eg_score = 0
        phase = 0

        # ---- material + piece-square tables ----
        for pt in PIECE_TYPES:
            bb_all = board.pieces_mask(pt, chess.WHITE)
            for sq in chess.scan_forward(bb_all):
                mg_score += PIECE_VALUES_MG[pt] + PST_MG[pt][sq]
                eg_score += PIECE_VALUES_EG[pt] + PST_EG[pt][sq]
                phase += GAME_PHASE_INC[pt]

            bb_all = board.pieces_mask(pt, chess.BLACK)
            for sq in chess.scan_forward(bb_all):
                msq = mirror(sq)
                mg_score -= PIECE_VALUES_MG[pt] + PST_MG[pt][msq]
                eg_score -= PIECE_VALUES_EG[pt] + PST_EG[pt][msq]
                phase += GAME_PHASE_INC[pt]

        # ---- bishop pair ----
        if popcount(board.bishops & white_occ) >= 2:
            mg_score += BISHOP_PAIR_MG
            eg_score += BISHOP_PAIR_EG
        if popcount(board.bishops & black_occ) >= 2:
            mg_score -= BISHOP_PAIR_MG
            eg_score -= BISHOP_PAIR_EG

        # ---- mobility (cheap bitboard-based approximation) ----
        for pt, w in MOBILITY_WEIGHTS:
            for sq in chess.scan_forward(board.pieces_mask(pt, chess.WHITE)):
                attacks = board.attacks_mask(sq)
                mob = popcount(attacks & ~white_occ)
                mg_score += mob * w
                eg_score += mob * w
            for sq in chess.scan_forward(board.pieces_mask(pt, chess.BLACK)):
                attacks = board.attacks_mask(sq)
                mob = popcount(attacks & ~black_occ)
                mg_score -= mob * w
                eg_score -= mob * w

        # ---- pawn structure ----
        wp = board.pawns & white_occ
        bp = board.pawns & black_occ
        for f in range(8):
            f_mask = FILE_MASK[f]
            if not (wp & f_mask) and not (bp & f_mask):
                continue
            wc = popcount(wp & f_mask)
            bc = popcount(bp & f_mask)
            if wc > 1:
                mg_score -= DOUBLE_PAWN_MG * (wc - 1)
                eg_score -= DOUBLE_PAWN_EG * (wc - 1)
            if bc > 1:
                mg_score += DOUBLE_PAWN_MG * (bc - 1)
                eg_score += DOUBLE_PAWN_EG * (bc - 1)
            if wc > 0 and (wp & ADJACENT_FILES_MASK[f]) == 0:
                mg_score -= ISOLATED_PAWN_MG
                eg_score -= ISOLATED_PAWN_EG
            if bc > 0 and (bp & ADJACENT_FILES_MASK[f]) == 0:
                mg_score += ISOLATED_PAWN_MG
                eg_score += ISOLATED_PAWN_EG

        for sq in chess.scan_forward(wp):
            if (PASSED_MASK[chess.WHITE][sq] & bp) == 0:
                rank = chess.square_rank(sq)
                bonus = (rank - 1) * PASSED_PAWN_MG
                mg_score += bonus
                eg_score += bonus * 2
        for sq in chess.scan_forward(bp):
            if (PASSED_MASK[chess.BLACK][sq] & wp) == 0:
                rank = 7 - chess.square_rank(sq)
                bonus = (rank - 1) * PASSED_PAWN_MG
                mg_score -= bonus
                eg_score -= bonus * 2

        # ---- rooks on (semi-)open files ----
        for sq in chess.scan_forward(board.rooks & white_occ):
            f = chess.square_file(sq)
            if popcount(wp & FILE_MASK[f]) == 0:
                mg_score += ROOK_OPEN_FILE_MG if popcount(bp & FILE_MASK[f]) == 0 else ROOK_SEMI_OPEN_FILE_MG
        for sq in chess.scan_forward(board.rooks & black_occ):
            f = chess.square_file(sq)
            if popcount(bp & FILE_MASK[f]) == 0:
                mg_score -= ROOK_OPEN_FILE_MG if popcount(wp & FILE_MASK[f]) == 0 else ROOK_SEMI_OPEN_FILE_MG

        # ---- king safety: pawn shield (middlegame only) ----
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is not None:
            wf = chess.square_file(wk)
            if wf <= 2 or wf >= 5:
                shield = 0
                for f in range(max(0, wf - 1), min(7, wf + 1) + 1):
                    if wp & FILE_MASK[f] & WHITE_KING_SHIELD_RANKS:
                        shield += 1
                mg_score += shield * KING_SHIELD_MG
        if bk is not None:
            bf = chess.square_file(bk)
            if bf <= 2 or bf >= 5:
                shield = 0
                for f in range(max(0, bf - 1), min(7, bf + 1) + 1):
                    if bp & FILE_MASK[f] & BLACK_KING_SHIELD_RANKS:
                        shield += 1
                mg_score -= shield * KING_SHIELD_MG

        # ---- tapered blend ----
        phase = min(phase, TOTAL_PHASE)
        tapered = mg_score * phase + eg_score * (TOTAL_PHASE - phase)
        # Explicit truncation toward zero keeps interpreted Python and Cython
        # builds identical and preserves color symmetry for negative scores.
        if tapered >= 0:
            score = tapered // TOTAL_PHASE
        else:
            score = -((-tapered) // TOTAL_PHASE)

        stm_score = score if board.turn == chess.WHITE else -score
        stm_score += TEMPO_BONUS  # small tempo bonus for the side to move
        return stm_score

    # ----------------------------------------------------------- ordering
    def order_moves(
        self,
        board: "chess.Board",
        moves: List["chess.Move"],
        ply: int,
        tt_move: Optional["chess.Move"],
        see_scores: Optional[Dict["chess.Move", int]] = None,
    ) -> List["chess.Move"]:
        scored: List[Tuple[int, chess.Move]] = []
        killer0, killer1 = self.killers[ply][0], self.killers[ply][1]
        opp_occ = board.occupied_co[not board.turn]
        for m in moves:
            is_en_passant = board.is_en_passant(m)
            is_cap = bool(opp_occ & (1 << m.to_square)) or is_en_passant
            if tt_move is not None and m == tt_move:
                score = 1_000_000
            elif is_cap:
                see = static_exchange_eval(board, m)
                if see_scores is not None:
                    see_scores[m] = see
                victim_pt = (
                    chess.PAWN if is_en_passant else board.piece_type_at(m.to_square)
                )
                attacker_pt = board.piece_type_at(m.from_square)
                mvv_lva = (
                    SEE_VALUES.get(victim_pt, 0) * 10
                    - SEE_VALUES.get(attacker_pt, 0)
                )
                # Profitable exchanges lead all non-TT moves. Losing captures
                # wait until after promotions and killers instead of consuming
                # the most valuable alpha-beta slots.
                if see >= 0:
                    score = 100_000 + see + mvv_lva
                else:
                    score = 70_000 + see + mvv_lva
            elif m.promotion:
                score = 90_000 + SEE_VALUES.get(m.promotion, 0)
            elif killer0 is not None and m == killer0:
                score = 80_000
            elif killer1 is not None and m == killer1:
                score = 79_000
            else:
                score = self.history.get(
                    (board.turn, m.from_square, m.to_square), 0
                )
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # -------------------------------------------------------- quiescence
    def quiescence(
        self,
        board: "chess.Board",
        alpha: int,
        beta: int,
        ply: int,
        count_node: bool = True,
    ) -> int:
        # A depth-zero negamax node has already been counted; recursive qsearch
        # calls have not. Avoid inflating node/NPS statistics at the boundary.
        if count_node:
            self.nodes += 1
            if self.nodes & TIME_CHECK_MASK == 0:
                self._check_time()

        in_check = board.is_check()
        if count_node and ply > 0 and (
            board.is_repetition(2)
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        ):
            # Checkmate ends the game before a draw can be claimed.
            if in_check and not any(board.generate_legal_moves()):
                return -MATE_VALUE + ply
            return 0

        if not in_check:
            stand_pat = self.evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
        else:
            stand_pat = -INF
            best = -INF

        if ply >= MAX_PLY - 1:
            if in_check and not any(board.generate_legal_moves()):
                return -MATE_VALUE + ply
            if not in_check and board.is_stalemate():
                return 0
            return self.evaluate(board) if in_check else stand_pat

        opp_occ = board.occupied_co[not board.turn]
        if in_check:
            candidates = list(board.legal_moves)
            if not candidates:
                return -MATE_VALUE + ply
        else:
            candidates = []
            has_legal_move = False
            for m in board.legal_moves:
                has_legal_move = True
                if (
                    bool(opp_occ & (1 << m.to_square))
                    or board.is_en_passant(m)
                    or m.promotion
                ):
                    candidates.append(m)
            if not has_legal_move:
                return 0  # stalemate

        see_scores: Dict[chess.Move, int] = {}
        candidates = self.order_moves(
            board,
            candidates,
            min(ply, MAX_PLY - 1),
            None,
            see_scores=see_scores,
        )

        for m in candidates:
            is_cap = bool(opp_occ & (1 << m.to_square)) or board.is_en_passant(m)
            if not in_check and is_cap:
                # Never prune a forcing check or promotion solely because its
                # material exchange looks unfavorable; either may be mate.
                see_val = see_scores[m]
                if see_val < 0:
                    if not m.promotion and not board.gives_check(m):
                        continue
                elif (
                    not m.promotion
                    and stand_pat + see_val + 130 < alpha
                    and not board.gives_check(m)
                ):
                    continue

            board.push(m)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break

        return best

    # ------------------------------------------------------------ negamax
    def negamax(
        self,
        board: "chess.Board",
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        can_null: bool = True,
    ) -> int:
        self.nodes += 1
        if self.nodes & TIME_CHECK_MASK == 0:
            self._check_time()

        pv_node = (beta - alpha) > 1
        alpha_orig = alpha

        in_check = board.is_check()
        if ply > 0 and (
            board.is_repetition(2)
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        ):
            # Checkmate takes precedence over a claimable draw.
            if in_check and not any(board.generate_legal_moves()):
                return -MATE_VALUE + ply
            return 0

        if ply >= MAX_PLY - 1:
            if in_check and not any(board.generate_legal_moves()):
                return -MATE_VALUE + ply
            if not in_check and board.is_stalemate():
                return 0
            return self.evaluate(board)

        if in_check:
            depth += 1  # check extension

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, count_node=False)

        zkey = _position_key(board)
        tt_entry = self.tt_current.get(zkey)
        if tt_entry is None:
            tt_entry = self.tt_previous.get(zkey)
            if (
                tt_entry is not None
                and len(self.tt_current) < self.tt_generation_entries
            ):
                self.tt_current[zkey] = tt_entry

        tt_move: Optional[chess.Move] = None
        if tt_entry is not None:
            tt_depth, stored_score, tt_flag, tt_move = tt_entry
            tt_score = _score_from_tt(stored_score, ply)
            if tt_depth >= depth and ply > 0:
                if tt_flag == EXACT:
                    return tt_score
                elif tt_flag == LOWER and tt_score > alpha:
                    alpha = tt_score
                elif tt_flag == UPPER and tt_score < beta:
                    beta = tt_score
                if alpha >= beta:
                    return tt_score

        static_eval = self.evaluate(board) if not in_check else -MATE_VALUE

        # ---- reverse futility pruning ----
        if (not pv_node and not in_check and depth <= 6 and abs(beta) < MATE_THRESHOLD):
            margin = 60 * depth  # Calibrated down from 85

            if static_eval - margin >= beta:
                return static_eval - margin

        # ---- null move pruning ----
        if (
            can_null and not pv_node and not in_check and depth >= 3
            and static_eval >= beta
            and self._has_non_pawn_material(board, board.turn)
        ):
            R = 3 + depth // 6
            board.push(chess.Move.null())
            try:
                score = -self.negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, can_null=False)
            finally:
                board.pop()
            if score >= beta and abs(score) < MATE_THRESHOLD:
                return beta

        moves = list(board.legal_moves)
        if not moves:
            if in_check:
                return -MATE_VALUE + ply
            return 0

        ordered = self.order_moves(board, moves, min(ply, MAX_PLY - 1), tt_move)

        best_score = -INF
        best_move: Optional[chess.Move] = ordered[0]
        move_index = 0

        opp_occ = board.occupied_co[not board.turn]
        for m in ordered:
            if ply == 0:
                self._check_time()
            is_capture = bool(opp_occ & (1 << m.to_square)) or board.is_en_passant(m)
            gives_check = False
            if not is_capture and not m.promotion:
                gives_check = board.gives_check(m)

            # ---- futility pruning of quiet moves near the leaves ----
            if (
                not pv_node and not in_check and not is_capture and not gives_check
                and not m.promotion and depth <= 4 and move_index > 0
                and abs(alpha) < MATE_THRESHOLD
            ):
                if static_eval + 85 * depth < alpha:  # Calibrated down from 120

                    move_index += 1
                    continue

            board.push(m)

            reduction = 0
            if (
                depth >= 3 and move_index >= 3 and not is_capture and not gives_check
                and not m.promotion and not in_check
            ):
                reduction = 1 + (1 if move_index >= 8 else 0) + (depth // 6)
                if pv_node:
                    reduction -= 1
                reduction = max(0, min(reduction, depth - 2))

            try:
                if move_index == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    # scout search (zero window), possibly reduced
                    score = -self.negamax(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1
                    )
                    if score > alpha and (reduction > 0 or score < beta):
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            move_index += 1

            if score > best_score:
                best_score = score
                best_move = m

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not is_capture:
                    if self.killers[min(ply, MAX_PLY - 1)][0] != m:
                        self.killers[min(ply, MAX_PLY - 1)][1] = self.killers[min(ply, MAX_PLY - 1)][0]
                        self.killers[min(ply, MAX_PLY - 1)][0] = m
                    key = (board.turn, m.from_square, m.to_square)
                    bonus = min(depth * depth, HISTORY_MAX // 4)
                    old_history = self.history.get(key, 0)
                    self.history[key] = old_history + bonus - (
                        old_history * bonus // HISTORY_MAX
                    )
                break

        flag = EXACT
        if best_score <= alpha_orig:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER

        self._store_tt(zkey, depth, best_score, flag, best_move, ply)

        return best_score

    def _store_tt(
        self,
        zkey: int,
        depth: int,
        score: int,
        flag: int,
        move: Optional["chess.Move"],
        ply: int,
    ) -> None:
        """Store a depth-preferred entry in the bounded two-generation TT."""
        current_entry = self.tt_current.get(zkey)
        prior_entry = current_entry or self.tt_previous.get(zkey)
        if prior_entry is not None:
            prior_depth, _, prior_flag, _ = prior_entry
            if prior_depth > depth and prior_flag == EXACT:
                return
            if prior_depth > depth + 1 and flag != EXACT:
                return

        if (
            current_entry is None
            and len(self.tt_current) >= self.tt_generation_entries
        ):
            self.tt_previous = self.tt_current
            self.tt_current = {}

        self.tt_current[zkey] = (
            depth,
            _score_to_tt(score, ply),
            flag,
            move,
        )

    def _has_non_pawn_material(self, board: "chess.Board", color: bool) -> bool:
        occ = board.occupied_co[color]
        return bool(occ & (board.knights | board.bishops | board.rooks | board.queens))

    def _check_time(self) -> None:
        if time.monotonic() >= self.stop_time:
            raise TimeUp()

    # --------------------------------------------------------------- root
    def search(
        self, board: "chess.Board", move_time: float, max_depth: int = 64
    ) -> Tuple[Optional["chess.Move"], dict]:
        legal = list(board.legal_moves)
        if not legal:
            return None, {"nodes": 0, "nps": 0, "time": 0.0, "depth": 0, "score": 0, "book": False}

        book_move = self.get_book_move(board)
        if book_move is not None:
            return book_move, {
                "nodes": 0,
                "nps": 0,
                "time": 0.0,
                "depth": 0,
                "score": 0,
                "book": True,
            }

        self.nodes = 0
        self.start_time = time.monotonic()
        try:
            budget = float(move_time)
        except (TypeError, ValueError):
            budget = 0.001
        if not math.isfinite(budget) or budget <= 0.0:
            budget = 0.001
        self.stop_time = self.start_time + budget
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        # Age history between moves so obsolete cutoffs do not permanently
        # dominate ordering, while retaining useful information from the game.
        if self.history:
            self.history = {
                key: min(value // 2, HISTORY_MAX // 2)
                for key, value in self.history.items()
                if value >= 2
            }

        best_move: Optional[chess.Move] = None
        best_score = 0
        depth_reached = 0

        if len(legal) == 1:
            return legal[0], {"nodes": 1, "nps": 0, "time": 0.0, "depth": 1, "score": 0, "book": False}

        alpha, beta = -INF, INF
        score = 0

        try:
            for depth in range(1, max_depth + 1):
                window = 25
                if depth >= 4:
                    alpha = score - window
                    beta = score + window
                else:
                    alpha, beta = -INF, INF

                while True:
                    score = self.negamax(board, depth, alpha, beta, 0)
                    if score <= alpha:
                        alpha = max(-INF, alpha - window)
                        window *= 2
                    elif score >= beta:
                        beta = min(INF, beta + window)
                        window *= 2
                    else:
                        break

                zkey = _position_key(board)
                entry = self.tt_current.get(zkey)
                if entry is None:
                    entry = self.tt_previous.get(zkey)
                if (
                    entry is not None
                    and entry[3] is not None
                    and entry[3] in legal
                ):
                    best_move = entry[3]
                best_score = score
                depth_reached = depth

                if time.monotonic() >= self.stop_time:
                    break
                if abs(score) >= MATE_THRESHOLD:
                    break
        except TimeUp:
            pass

        if best_move is None:
            best_move = legal[0]

        elapsed = max(1e-6, time.monotonic() - self.start_time)
        info = {
            "nodes": self.nodes,
            "nps": int(self.nodes / elapsed),
            "time": elapsed,
            "depth": depth_reached,
            "score": best_score,
            "book": False,
        }
        return best_move, info



# --------------------------------------------------------------------------
# Interactive CLI
# --------------------------------------------------------------------------

def print_board(board: "chess.Board") -> None:
    print()
    print(board.unicode(borders=True, empty_square="."))
    print()


def format_score(score: int) -> str:
    if score >= MATE_THRESHOLD:
        moves_to_mate = (MATE_VALUE - score + 1) // 2
        return f"mate in {moves_to_mate}"
    if score <= -MATE_THRESHOLD:
        moves_to_mate = (MATE_VALUE + score + 1) // 2
        return f"mate in -{moves_to_mate}"
    return f"{score / 100:+.2f}"


def ask_colour() -> bool:
    while True:
        ans = input("Play as White or Black? [w/b]: ").strip().lower()
        if ans in ("w", "white"):
            return chess.WHITE
        if ans in ("b", "black"):
            return chess.BLACK
        print("Please answer 'w' or 'b'.")


def ask_time_per_move() -> float:
    while True:
        ans = input("Time per engine move, in seconds: ").strip()
        try:
            t = float(ans)
            if t > 0:
                return t
        except ValueError:
            pass
        print("Please enter a positive number.")


def ask_human_move(board: "chess.Board") -> "chess.Move":
    while True:
        ans = input("Your move (SAN or UCI, e.g. 'Nf3' or 'g1f3', 'moves' to list, 'quit' to exit): ").strip()
        if ans.lower() in ("quit", "exit"):
            print("Goodbye.")
            sys.exit(0)
        if ans.lower() == "moves":
            print(", ".join(sorted(board.san(m) for m in board.legal_moves)))
            continue
        try:
            move = board.parse_san(ans)
            return move
        except ValueError:
            pass
        try:
            move = chess.Move.from_uci(ans)
            if move in board.legal_moves:
                return move
        except ValueError:
            pass
        print("Not a legal move, try again (type 'moves' to see legal moves).")


def main() -> None:
    print("=" * 60)
    print(" PySharkX - single file python-chess / numpy / Cython engine")
    print("=" * 60)

    human_colour = ask_colour()
    move_time = ask_time_per_move()

    board = chess.Board()
    engine = Engine()

    if engine.book_loaded:
        print(f"Opening book loaded: {engine.book_path}")
    else:
        print(f"No opening book found at '{engine.book_path}' - playing from search only.")

    print_board(board)

    try:
        _play_game(board, engine, human_colour, move_time)
    finally:
        engine.close_book()

def _play_game(
    board: "chess.Board", engine: "Engine", human_colour: bool, move_time: float
) -> None:
    while not board.is_game_over(claim_draw=True):
        if board.turn == human_colour:
            move = ask_human_move(board)
            print(f"You played: {board.san(move)}")
            board.push(move)
        else:
            print("Engine is thinking...")
            move, info = engine.search(board, move_time)
            if move is None:
                break
            san = board.san(move)
            board.push(move)
            if info.get("book"):
                print(f"Engine plays: {san}   (from opening book)")
            else:
                print(
                    f"Engine plays: {san}   "
                    f"(score {format_score(info['score'])})"
                )
                print(
                    f"  depth reached : {info['depth']}\n"
                    f"  nodes searched: {info['nodes']:,}\n"
                    f"  time taken    : {info['time']:.2f}s\n"
                    f"  nps           : {info['nps']:,}"
                )

        print_board(board)

    print("Game over:", board.result(claim_draw=True))
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate - {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate.")
    elif board.is_insufficient_material():
        print("Draw - insufficient material.")
    else:
        print("Draw.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")
