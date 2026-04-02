"""
main.py – Entry point for the chess simulation.

Usage
-----
    python main.py

Controls
--------
    Mouse          : click or drag pieces to move them
    Mouse (bot turn): click/drag to queue a premove (shown in blue)
    ← or Z         : undo last pair of moves (player + bot)
    → or Y         : redo
    R              : new game
    F              : flip board orientation
"""

from game import ChessGame

if __name__ == "__main__":
    ChessGame().run()
