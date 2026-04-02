"""
game.py – Chess game with pygame.

Features
--------
* Full chess rules via python-chess (castling, en passant, promotion, …)
* Mouse drag-and-drop piece movement
* Smooth piece movement animation (ease-in-out cubic)
* Premove: queue a move while the bot is thinking (shown in blue)
* Last-move highlight (yellow tint)
* Valid-move dots and capture rings
* King-in-check red glow
* AI bot (minimax, depth 3) runs in a background thread
* Undo (← / Z) and Redo (→ / Y) – always land on White's turn
* Board flip (F key or button)
* New game (R key or button)
* On-screen buttons for all keyboard actions
* Pawn-promotion dialog
"""

import sys
import math
import time
import threading
from pathlib import Path

import pygame
import chess

from bot import Bot

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
WINDOW_W  = 720
WINDOW_H  = 820
BOARD_PX  = 640          # board pixel size
SQ        = BOARD_PX // 8  # 80 px per square
BX        = (WINDOW_W - BOARD_PX) // 2   # board left edge  (40)
BY        = (WINDOW_H - BOARD_PX) // 2 - 20  # board top edge (70)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_BG          = ( 40,  40,  40)
C_LIGHT       = (240, 217, 181)
C_DARK        = (181, 136,  99)
C_LM_LIGHT    = (205, 210, 106)   # last-move highlight, light square
C_LM_DARK     = (170, 162,  58)   # last-move highlight, dark square
C_SEL_LIGHT   = (130, 151, 105)   # selected, light square
C_SEL_DARK    = (100, 111,  64)   # selected, dark square
C_PRE_LIGHT   = (160, 180, 220)   # premove, light square
C_PRE_DARK    = (100, 120, 180)   # premove, dark square
C_CHECK       = (220,  50,  50)
C_W_FILL      = (240, 240, 220)   # white piece fill
C_B_FILL      = ( 20,  20,  20)   # black piece fill
C_W_OUTLINE   = ( 20,  20,  20)   # white piece outline
C_B_OUTLINE   = (230, 230, 210)   # black piece outline
C_STATUS_BG   = ( 60,  60,  60)
C_BTN         = ( 80,  80,  80)
C_BTN_HOV     = (110, 110, 110)
C_TXT         = (220, 220, 220)
C_TXT_DIM     = (140, 140, 140)

# ---------------------------------------------------------------------------
# Piece Unicode symbols
# ---------------------------------------------------------------------------
SYMBOLS = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}

FALLBACK_LETTER = {
    chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
    chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P",
}

ANIM_SECS = 0.20   # animation duration
PIECE_TEXTURE_PADDING = 8
TEXTURE_DIR_VARIANTS = ("lichess texture", "lichess_texture", "lichess-texture")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ease(t: float) -> float:
    """Ease-in-out cubic."""
    t = max(0.0, min(1.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _sq_center(sq: int, flipped: bool) -> tuple[int, int]:
    """Screen pixel centre of a board square."""
    col = chess.square_file(sq)
    row = chess.square_rank(sq)
    dc  = (7 - col) if flipped else col
    dr  = row        if flipped else (7 - row)
    return BX + dc * SQ + SQ // 2, BY + dr * SQ + SQ // 2


def _draw_col_row(sq: int, flipped: bool) -> tuple[int, int]:
    """(draw_col, draw_row) for top-left corner rectangle calculations."""
    col = chess.square_file(sq)
    row = chess.square_rank(sq)
    dc  = (7 - col) if flipped else col
    dr  = row        if flipped else (7 - row)
    return dc, dr


def _is_light(dc: int, dr: int) -> bool:
    return (dc + dr) % 2 == 0


# ---------------------------------------------------------------------------
# PieceAnimation
# ---------------------------------------------------------------------------
class PieceAnimation:
    """Smoothly interpolates a piece from one square to another."""

    def __init__(self, piece_type: int, piece_color: bool,
                 from_sq: int, to_sq: int):
        self.piece_type  = piece_type
        self.piece_color = piece_color
        self.from_sq     = from_sq
        self.to_sq       = to_sq
        self._t0         = time.monotonic()

    @property
    def done(self) -> bool:
        return time.monotonic() - self._t0 >= ANIM_SECS

    def current_pos(self, flipped: bool) -> tuple[float, float]:
        t = _ease((time.monotonic() - self._t0) / ANIM_SECS)
        fx, fy = _sq_center(self.from_sq, flipped)
        tx, ty = _sq_center(self.to_sq,   flipped)
        return fx + (tx - fx) * t, fy + (ty - fy) * t


# ---------------------------------------------------------------------------
# ChessGame
# ---------------------------------------------------------------------------
class ChessGame:
    """Main game class – owns the pygame window and the game loop."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Jeu d'échecs")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock  = pygame.time.Clock()

        self._init_fonts()
        self._init_piece_textures()

        # ── game state ──────────────────────────────────────────────────
        self.board      = chess.Board()
        # history: list of (board_copy, move_that_led_here)
        self.history    = [(self.board.copy(), None)]
        self.hist_idx   = 0
        self.last_move  = None   # chess.Move | None
        self.flipped    = False  # board orientation

        # ── selection / drag ────────────────────────────────────────────
        self.sel_sq     = None   # currently selected square
        self.sel_moves  = []     # legal destination squares for sel_sq
        self.dragging   = False
        self.drag_sq    = None   # square piece was grabbed from
        self.drag_piece = None   # chess.Piece
        self.drag_pos   = (0, 0) # current cursor position

        # ── premove ─────────────────────────────────────────────────────
        # tuple (from_sq, to_sq, promo_piece_type | None) or None
        self.premove    = None

        # ── animation ───────────────────────────────────────────────────
        self.anim: PieceAnimation | None = None

        # ── bot ─────────────────────────────────────────────────────────
        self.bot        = Bot(depth=3)
        self.bot_busy   = False
        self.bot_result = None        # chess.Move | None  (set by thread)
        self.bot_stop   = threading.Event()

        # ── promotion dialog ─────────────────────────────────────────────
        # tuple (from_sq, to_sq) when waiting for promotion choice
        self.promo_pending = None

        # ── UI ──────────────────────────────────────────────────────────
        self.buttons: dict[str, pygame.Rect] = {}

        self._update_status()

    # ====================================================================
    # Font initialisation
    # ====================================================================
    def _init_fonts(self):
        unicode_ok = False
        for name in ("dejavusans", "freesans", "liberationsans",
                     "notosans", "arial", "unifont", None):
            try:
                f = pygame.font.SysFont(name, 58, bold=True)
                if f.render("♔", True, (0, 0, 0)).get_width() > 8:
                    self._pf   = f
                    unicode_ok = True
                    break
            except Exception:
                pass
        if not unicode_ok:
            self._pf = pygame.font.SysFont(None, 58, bold=True)
        self._use_unicode = unicode_ok

        self._uf  = pygame.font.SysFont("segoeui,arial,sans", 22)
        self._usf = pygame.font.SysFont("segoeui,arial,sans", 16)
        self._cf  = pygame.font.SysFont("segoeui,arial,sans", 13)

    def _init_piece_textures(self):
        self._piece_textures: dict[tuple[int, bool], pygame.Surface] = {}
        base_dir = Path(__file__).resolve().parent
        tex_dir = None
        for dirname in TEXTURE_DIR_VARIANTS:
            candidate = base_dir / dirname
            if candidate.is_dir():
                tex_dir = candidate
                break
        if tex_dir is None:
            return

        piece_codes = {
            chess.KING: "K",
            chess.QUEEN: "Q",
            chess.ROOK: "R",
            chess.BISHOP: "B",
            chess.KNIGHT: "N",
            chess.PAWN: "P",
        }

        target_size = (SQ - PIECE_TEXTURE_PADDING, SQ - PIECE_TEXTURE_PADDING)
        for color, prefix in ((chess.WHITE, "w"), (chess.BLACK, "b")):
            for pt, code in piece_codes.items():
                base = f"{prefix}{code}"
                candidates = (tex_dir / f"{base}.svg",
                              tex_dir / f"{base}.png",
                              tex_dir / f"{base}.webp")
                for path in candidates:
                    if not path.exists():
                        continue
                    try:
                        surf = pygame.image.load(str(path)).convert_alpha()
                        self._piece_textures[(pt, color)] = pygame.transform.smoothscale(
                            surf, target_size
                        )
                        break
                    except (pygame.error, OSError, ValueError):
                        continue

    # ====================================================================
    # Coordinate utilities
    # ====================================================================
    def _screen_to_sq(self, x: int, y: int) -> int | None:
        col = (x - BX) // SQ
        row = (y - BY) // SQ
        if not (0 <= col <= 7 and 0 <= row <= 7):
            return None
        sq_col = (7 - col) if self.flipped else col
        sq_row = row        if self.flipped else (7 - row)
        return chess.square(sq_col, sq_row)

    # ====================================================================
    # Game logic helpers
    # ====================================================================
    def _player_turn(self) -> bool:
        return self.board.turn == chess.WHITE

    def _can_select(self, sq: int) -> bool:
        p = self.board.piece_at(sq)
        return p is not None and p.color == chess.WHITE and self._player_turn()

    def _legal_dests(self, sq: int) -> list[int]:
        return [m.to_square for m in self.board.legal_moves
                if m.from_square == sq]

    def _update_status(self):
        b = self.board
        if b.is_checkmate():
            winner = "Blanc" if b.turn == chess.BLACK else "Noir"
            self.status = f"Échec et mat ! {winner} gagne !"
        elif b.is_stalemate():
            self.status = "Pat — match nul"
        elif b.is_insufficient_material():
            self.status = "Matériel insuffisant — match nul"
        elif b.is_check():
            checked_player = "Blanc" if b.turn == chess.WHITE else "Noir"
            self.status = f"Échec au roi {checked_player} !"
        elif self._player_turn():
            suffix = "  (prémove en attente)" if self.premove else ""
            self.status = f"Votre tour (Blanc){suffix}"
        else:
            self.status = "Le bot réfléchit…"

    # ====================================================================
    # Move execution
    # ====================================================================
    def _execute_move(self, move: chess.Move, animate: bool = True):
        from_sq = move.from_square
        to_sq   = move.to_square
        piece   = self.board.piece_at(from_sq)

        self.board.push(move)
        self.last_move = move

        # Truncate redo branch
        if self.hist_idx < len(self.history) - 1:
            self.history = self.history[:self.hist_idx + 1]
        self.history.append((self.board.copy(), move))
        self.hist_idx = len(self.history) - 1

        if animate and piece:
            self.anim = PieceAnimation(piece.piece_type, piece.color,
                                       from_sq, to_sq)

        self._clear_selection()
        self._update_status()

    def _player_move(self, move: chess.Move):
        self._execute_move(move)
        if not self.board.is_game_over():
            self._start_bot()

    def _try_player_move(self, from_sq: int, to_sq: int,
                         promo: int | None = None) -> bool:
        """Attempt a player move; shows promotion dialog if needed.
        Returns True if a move was made or the dialog was opened."""
        piece = self.board.piece_at(from_sq)
        if piece is None:
            return False

        # Pawn promotion
        if piece.piece_type == chess.PAWN and promo is None:
            if ((piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or
                    (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0)):
                test = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                if test in self.board.legal_moves:
                    self.promo_pending = (from_sq, to_sq)
                    self._clear_selection()
                    return True

        move = chess.Move(from_sq, to_sq, promotion=promo)
        if move in self.board.legal_moves:
            self._player_move(move)
            return True
        return False

    # ====================================================================
    # Bot
    # ====================================================================
    def _start_bot(self):
        if self.bot_busy or self.board.is_game_over():
            return
        self.bot_busy   = True
        self.bot_result = None
        self.bot_stop.clear()
        threading.Thread(target=self._bot_thread, daemon=True).start()

    def _bot_thread(self):
        board_copy = self.board.copy()
        move = self.bot.get_move(board_copy, self.bot_stop)
        if not self.bot_stop.is_set():
            self.bot_result = move
        self.bot_busy = False

    # ====================================================================
    # Premove
    # ====================================================================
    def _try_premove(self):
        if self.premove is None:
            return
        fs, ts, pr = self.premove
        for m in self.board.legal_moves:
            if m.from_square == fs and m.to_square == ts:
                if pr is None or m.promotion == pr:
                    self.premove = None
                    self._execute_move(m)
                    if not self.board.is_game_over():
                        self._start_bot()
                    return
        # Premove is no longer legal
        self.premove = None
        self._update_status()

    # ====================================================================
    # Undo / Redo
    # ====================================================================
    def _cancel_bot(self):
        if self.bot_busy:
            self.bot_stop.set()
            self.bot_busy   = False
            self.bot_result = None

    def undo(self):
        self._cancel_bot()
        target = self.hist_idx
        # Step back; skip bot's move to land on player's turn
        if target > 0:
            target -= 1
        if target > 0 and self.history[target][0].turn != chess.WHITE:
            target -= 1
        if target == self.hist_idx:
            return
        self._goto_history(target)

    def redo(self):
        target = self.hist_idx
        if target < len(self.history) - 1:
            target += 1
        if (target < len(self.history) - 1 and
                self.history[target][0].turn != chess.WHITE):
            target += 1
        if target == self.hist_idx:
            return
        self._goto_history(target)

    def _goto_history(self, idx: int):
        self.hist_idx  = idx
        board, mv      = self.history[idx]
        self.board     = board.copy()
        self.last_move = mv
        self._clear_ui()
        self._update_status()

    def reset(self):
        self._cancel_bot()
        self.board     = chess.Board()
        self.history   = [(self.board.copy(), None)]
        self.hist_idx  = 0
        self.last_move = None
        self._clear_ui()
        self._update_status()

    # ====================================================================
    # UI state helpers
    # ====================================================================
    def _clear_selection(self):
        self.sel_sq    = None
        self.sel_moves = []
        self.dragging  = False
        self.drag_piece = None
        self.drag_sq   = None

    def _clear_ui(self):
        self._clear_selection()
        self.anim         = None
        self.premove      = None
        self.promo_pending = None

    # ====================================================================
    # Input handlers
    # ====================================================================
    def on_keydown(self, key: int):
        if key in (pygame.K_LEFT, pygame.K_z):
            self.undo()
        elif key in (pygame.K_RIGHT, pygame.K_y):
            self.redo()
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_f:
            self.flipped = not self.flipped

    def on_mousedown(self, pos: tuple[int, int], btn: int):
        if btn != 1:
            return

        # ── on-screen buttons ───────────────────────────────────────────
        for bid, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if bid == "undo":
                    self.undo()
                elif bid == "redo":
                    self.redo()
                elif bid == "reset":
                    self.reset()
                elif bid == "flip":
                    self.flipped = not self.flipped
                return

        # ── promotion dialog ─────────────────────────────────────────────
        if self.promo_pending is not None:
            self._handle_promo_click(pos)
            return

        sq = self._screen_to_sq(*pos)
        if sq is None:
            self._clear_selection()
            return

        if self._player_turn():
            if self.sel_sq is not None and sq in self.sel_moves:
                # Move the selected piece
                self._try_player_move(self.sel_sq, sq)
            elif self._can_select(sq):
                # Select piece + start drag
                self.sel_sq    = sq
                self.sel_moves = self._legal_dests(sq)
                self.dragging  = True
                self.drag_sq   = sq
                self.drag_piece = self.board.piece_at(sq)
                self.drag_pos  = pos
            else:
                self._clear_selection()
        else:
            # It's the bot's turn – handle premove
            p = self.board.piece_at(sq)
            if p and p.color == chess.WHITE:
                # Toggle premove source off if same square
                if self.premove and self.premove[0] == sq:
                    self.premove = None
                    self._update_status()
                    return
                self.sel_sq    = sq
                self.sel_moves = []
                self.dragging  = True
                self.drag_sq   = sq
                self.drag_piece = p
                self.drag_pos  = pos

    def on_mouseup(self, pos: tuple[int, int], btn: int):
        if btn != 1:
            return
        sq = self._screen_to_sq(*pos)

        if self.dragging and self.drag_sq is not None:
            if sq is not None and sq != self.drag_sq:
                if self._player_turn():
                    if sq in self.sel_moves:
                        self._try_player_move(self.drag_sq, sq)
                    else:
                        self._clear_selection()
                else:
                    # Register premove
                    p = self.board.piece_at(self.drag_sq)
                    if p and p.color == chess.WHITE:
                        self.premove  = (self.drag_sq, sq, None)
                        self.sel_sq   = None
                        self.sel_moves = []
                        self._update_status()
            self.dragging  = False
            self.drag_piece = None
            self.drag_sq   = None

    def on_mousemove(self, pos: tuple[int, int]):
        if self.dragging:
            self.drag_pos = pos

    def _handle_promo_click(self, pos: tuple[int, int]):
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        fs, ts = self.promo_pending
        dw = 4 * SQ
        dx = WINDOW_W // 2 - dw // 2
        dy = WINDOW_H // 2 - SQ // 2
        for i, pt in enumerate(pieces):
            r = pygame.Rect(dx + i * SQ, dy, SQ, SQ)
            if r.collidepoint(pos):
                m = chess.Move(fs, ts, promotion=pt)
                if m in self.board.legal_moves:
                    self.promo_pending = None
                    self._player_move(m)
                return
        self.promo_pending = None   # cancelled by clicking outside

    # ====================================================================
    # Drawing
    # ====================================================================
    def _draw_board(self):
        for dr in range(8):
            for dc in range(8):
                color = C_LIGHT if _is_light(dc, dr) else C_DARK
                pygame.draw.rect(self.screen, color,
                                 (BX + dc * SQ, BY + dr * SQ, SQ, SQ))

        # Rank numbers (left edge)
        for dr in range(8):
            rank  = dr + 1 if self.flipped else 8 - dr
            color = C_DARK if _is_light(0, dr) else C_LIGHT
            t = self._cf.render(str(rank), True, color)
            self.screen.blit(t, (BX + 2, BY + dr * SQ + 2))

        # File letters (bottom edge)
        for dc in range(8):
            fi   = (7 - dc) if self.flipped else dc
            file = chr(ord("a") + fi)
            color = C_LIGHT if _is_light(dc, 7) else C_DARK
            t = self._cf.render(file, True, color)
            self.screen.blit(t, (BX + dc * SQ + SQ - 13,
                                  BY + BOARD_PX - 16))

    def _hl_square(self, sq: int, color: tuple):
        dc, dr = _draw_col_row(sq, self.flipped)
        pygame.draw.rect(self.screen, color,
                         (BX + dc * SQ, BY + dr * SQ, SQ, SQ))

    def _draw_highlights(self):
        # Last move
        if self.last_move:
            for sq in (self.last_move.from_square, self.last_move.to_square):
                dc, dr = _draw_col_row(sq, self.flipped)
                c = C_LM_LIGHT if _is_light(dc, dr) else C_LM_DARK
                self._hl_square(sq, c)

        # Selected square
        if self.sel_sq is not None:
            dc, dr = _draw_col_row(self.sel_sq, self.flipped)
            c = C_SEL_LIGHT if _is_light(dc, dr) else C_SEL_DARK
            self._hl_square(self.sel_sq, c)

        # Premove squares
        if self.premove:
            for sq in (self.premove[0], self.premove[1]):
                dc, dr = _draw_col_row(sq, self.flipped)
                c = C_PRE_LIGHT if _is_light(dc, dr) else C_PRE_DARK
                self._hl_square(sq, c)

        # King in check – red glow
        if self.board.is_check():
            ksq = self.board.king(self.board.turn)
            dc, dr = _draw_col_row(ksq, self.flipped)
            s = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
            pygame.draw.circle(s, (*C_CHECK, 160),
                               (SQ // 2, SQ // 2), SQ // 2)
            self.screen.blit(s, (BX + dc * SQ, BY + dr * SQ))

        # Valid-move dots / capture rings
        if self.sel_sq is not None and self._player_turn():
            for dest in self.sel_moves:
                dc, dr = _draw_col_row(dest, self.flipped)
                s = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
                if self.board.piece_at(dest):
                    pygame.draw.circle(s, (0, 0, 0, 75),
                                       (SQ // 2, SQ // 2), SQ // 2 - 3, 7)
                else:
                    pygame.draw.circle(s, (0, 0, 0, 75),
                                       (SQ // 2, SQ // 2), SQ // 7)
                self.screen.blit(s, (BX + dc * SQ, BY + dr * SQ))

    def _draw_piece(self, pt: int, pc: bool, cx: int, cy: int):
        """Render a chess piece centred at (cx, cy)."""
        tex = self._piece_textures.get((pt, pc))
        if tex is not None:
            self.screen.blit(tex, tex.get_rect(center=(cx, cy)))
            return

        fill    = C_W_FILL    if pc == chess.WHITE else C_B_FILL
        outline = C_W_OUTLINE if pc == chess.WHITE else C_B_OUTLINE

        if self._use_unicode:
            sym  = SYMBOLS[(pt, pc)]
            surf = self._pf.render(sym, True, outline)
            r    = surf.get_rect(center=(cx, cy))
            for dx, dy in ((-2, -2), (2, -2), (-2, 2), (2, 2),
                           (0, -2), (0, 2), (-2, 0), (2, 0)):
                self.screen.blit(surf, (r.x + dx, r.y + dy))
            surf2 = self._pf.render(sym, True, fill)
            self.screen.blit(surf2, surf2.get_rect(center=(cx, cy)))
        else:
            # Fallback: coloured circle + letter
            rad = SQ // 2 - 6
            pygame.draw.circle(self.screen, fill,    (cx, cy), rad)
            pygame.draw.circle(self.screen, outline, (cx, cy), rad, 3)
            t = self._uf.render(FALLBACK_LETTER[pt], True, outline)
            self.screen.blit(t, t.get_rect(center=(cx, cy)))

    def _draw_pieces(self):
        anim_to = (self.anim.to_sq
                   if (self.anim and not self.anim.done) else None)

        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p is None:
                continue
            if sq == anim_to:
                continue          # animated piece drawn separately
            if self.dragging and sq == self.drag_sq:
                continue          # dragged piece drawn at cursor
            cx, cy = _sq_center(sq, self.flipped)
            self._draw_piece(p.piece_type, p.color, cx, cy)

        # Animated piece
        if self.anim:
            if not self.anim.done:
                ax, ay = self.anim.current_pos(self.flipped)
                self._draw_piece(self.anim.piece_type, self.anim.piece_color,
                                 int(ax), int(ay))
            else:
                self.anim = None

        # Piece following the cursor during drag
        if self.dragging and self.drag_piece:
            self._draw_piece(self.drag_piece.piece_type, self.drag_piece.color,
                             *self.drag_pos)

    def _draw_promo_dialog(self):
        if self.promo_pending is None:
            return
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        fs, ts  = self.promo_pending
        color   = self.board.turn
        dw = 4 * SQ
        dx = WINDOW_W // 2 - dw // 2
        dy = WINDOW_H // 2 - SQ // 2

        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, (200, 180, 140),
                         (dx - 6, dy - 6, dw + 12, SQ + 12),
                         border_radius=10)
        for i, pt in enumerate(pieces):
            bg = C_LIGHT if i % 2 == 0 else C_DARK
            pygame.draw.rect(self.screen, bg,
                             (dx + i * SQ, dy, SQ, SQ))
            self._draw_piece(pt, color,
                             dx + i * SQ + SQ // 2, dy + SQ // 2)

    def _draw_ui(self):
        # Status bar
        sy = BY + BOARD_PX + 10
        pygame.draw.rect(self.screen, C_STATUS_BG,
                         (BX, sy, BOARD_PX, 38), border_radius=6)
        t  = self._uf.render(self.status, True, C_TXT)
        self.screen.blit(t, t.get_rect(center=(WINDOW_W // 2, sy + 19)))

        # Buttons
        by = sy + 48
        specs = [
            ("undo",  "← Annuler",        BX,       120),
            ("redo",  "Refaire →",         BX + 130, 120),
            ("reset", "↺ Nouvelle partie", BX + 260, 165),
            ("flip",  "⇅ Retourner",      BX + 435, 120),
        ]
        mp = pygame.mouse.get_pos()
        self.buttons = {}
        for bid, label, bx_btn, bw in specs:
            r   = pygame.Rect(bx_btn, by, bw, 34)
            self.buttons[bid] = r
            hov = r.collidepoint(mp)
            pygame.draw.rect(self.screen, C_BTN_HOV if hov else C_BTN,
                             r, border_radius=5)
            ts = self._usf.render(label, True, C_TXT)
            self.screen.blit(ts, ts.get_rect(center=r.center))

        # Keyboard shortcut hint
        hint = ("← / Z : Annuler   → / Y : Refaire   "
                "R : Nouvelle partie   F : Retourner")
        ht = self._usf.render(hint, True, C_TXT_DIM)
        self.screen.blit(ht, ht.get_rect(center=(WINDOW_W // 2, by + 52)))

    def draw(self):
        self.screen.fill(C_BG)
        self._draw_board()
        self._draw_highlights()
        self._draw_pieces()
        self._draw_promo_dialog()
        self._draw_ui()
        pygame.display.flip()

    # ====================================================================
    # Main loop
    # ====================================================================
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_mousedown(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.on_mouseup(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self.on_mousemove(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.on_keydown(event.key)

            # Pick up bot result (set by background thread)
            if self.bot_result is not None and not self.bot_busy:
                mv              = self.bot_result
                self.bot_result = None
                self._execute_move(mv)
                if self._player_turn() and not self.board.is_game_over():
                    self._try_premove()

            self.draw()
            self.clock.tick(60)
