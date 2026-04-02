"""
bot.py – Chess AI using minimax with alpha-beta pruning.

The bot evaluates positions using material values combined with
piece-square tables sourced from the Chess Programming Wiki.
"""

import math
import chess
import threading

# ---------------------------------------------------------------------------
# Material values (centipawns)
# ---------------------------------------------------------------------------
PIECE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:  20_000,
}

# ---------------------------------------------------------------------------
# Piece-square tables (from White's perspective, index 0 = a8, 63 = h1)
# so that table[0] is the back-rank for the opponent and table[56] is a1.
# Applied as: white score += table[(7-rank)*8 + file]
#             black score += table[rank*8 + file]
# ---------------------------------------------------------------------------
_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

PST = {
    chess.PAWN:   _PAWN,
    chess.KNIGHT: _KNIGHT,
    chess.BISHOP: _BISHOP,
    chess.ROOK:   _ROOK,
    chess.QUEEN:  _QUEEN,
    chess.KING:   _KING_MG,   # switched to _KING_EG in endgame
}

# Endgame threshold (total non-king material on both sides, centipawns)
_EG_THRESHOLD = 1300


def _is_endgame(board: chess.Board) -> bool:
    total = sum(
        PIECE_VALUE[p.piece_type]
        for sq in chess.SQUARES
        if (p := board.piece_at(sq)) and p.piece_type != chess.KING
    )
    return total <= _EG_THRESHOLD


def _pst_index(sq: int, color: chess.Color) -> int:
    """Return the piece-square table index for a square and color."""
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    if color == chess.WHITE:
        return (7 - rank) * 8 + file
    return rank * 8 + file


def _evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns (positive = White advantage)."""
    if board.is_checkmate():
        return -100_000 if board.turn == chess.WHITE else 100_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    eg = _is_endgame(board)
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        val = PIECE_VALUE[p.piece_type]
        if p.piece_type == chess.KING:
            table = _KING_EG if eg else _KING_MG
        else:
            table = PST[p.piece_type]
        pos = table[_pst_index(sq, p.color)]
        if p.color == chess.WHITE:
            score += val + pos
        else:
            score -= val + pos
    return score


def _order_moves(board: chess.Board, moves):
    """Sort moves: captures (MVV-LVA) first, then promotions, then quiet."""
    def key(move):
        s = 0
        if board.is_capture(move):
            victim   = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                s += 10 * PIECE_VALUE[victim.piece_type] - PIECE_VALUE[attacker.piece_type]
        if move.promotion:
            s += PIECE_VALUE.get(move.promotion, 0)
        return s
    return sorted(moves, key=key, reverse=True)


class Bot:
    """Minimax bot with alpha-beta pruning."""

    def __init__(self, depth: int = 3):
        self.depth = depth

    # ------------------------------------------------------------------
    def get_move(self, board: chess.Board,
                 stop: threading.Event | None = None) -> chess.Move | None:
        """Return the best move for the side to move."""
        maximizing = board.turn == chess.WHITE
        best_move  = None
        best_score = -math.inf if maximizing else math.inf

        for move in _order_moves(board, list(board.legal_moves)):
            if stop and stop.is_set():
                break
            board.push(move)
            score = self._minimax(board, self.depth - 1,
                                  -math.inf, math.inf,
                                  not maximizing, stop)
            board.pop()
            if maximizing and score > best_score:
                best_score, best_move = score, move
            elif not maximizing and score < best_score:
                best_score, best_move = score, move

        return best_move

    # ------------------------------------------------------------------
    def _minimax(self, board: chess.Board, depth: int,
                 alpha: float, beta: float,
                 maximizing: bool,
                 stop: threading.Event | None) -> float:
        if stop and stop.is_set():
            return 0
        if depth == 0 or board.is_game_over():
            return _evaluate(board)

        moves = _order_moves(board, list(board.legal_moves))
        if maximizing:
            val = -math.inf
            for m in moves:
                board.push(m)
                val = max(val, self._minimax(board, depth - 1,
                                             alpha, beta, False, stop))
                board.pop()
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = math.inf
            for m in moves:
                board.push(m)
                val = min(val, self._minimax(board, depth - 1,
                                             alpha, beta, True, stop))
                board.pop()
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val
