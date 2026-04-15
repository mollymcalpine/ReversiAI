"""
Board representation and low-level operations.

The Board class is a pure data structure.
It knows how to:
- Store piece positions.
- Clone itself.
- Detect which pieces would flip for a candidate move.
- Apply a move.
- Report scores.

It does NOT know aboyt turns, whose turn it is, or win conditions.
That belongs in Game.
"""

import numpy as np
from game.constants import BOARD_SIZE, BLACK, WHITE, EMPTY, DIRECTIONS

class Board:
    """An 8x8 Reversi board."""

    def __init__(self):
        self._grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8)
        self._place_initial_pieces()

    #-----------------------------------------------------------------------
    # Setup:
    #-----------------------------------------------------------------------

    def _place_initial_pieces(self) -> None:
        """Places the initial four pieces in the center of the board."""
        mid = BOARD_SIZE // 2
        self._grid[mid - 1][mid - 1] = WHITE
        self._grid[mid - 1][mid]     = BLACK
        self._grid[mid][mid - 1]     = BLACK
        self._grid[mid][mid]         = WHITE

    #-----------------------------------------------------------------------
    # Read Access:
    #-----------------------------------------------------------------------

    def get(self, row: int, col: int) -> int:
        """Returns the piece at the given position (BLACK, WHITE, or EMPTY)"""
        return int(self._grid[row][col])
    
    def is_empty(self, row: int, col: int) -> bool:
        """Returns True if the given position is empty"""
        return self._grid[row][col] == EMPTY
    
    def in_bounds(self, row: int, col: int) -> bool:
        """Returns True if the given position is within the board boundaries"""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
    
    def score(self, player: int) -> int:
        """Returns the score for the given player"""
        return int(np.sum(self._grid == player))
    
    def scores(self) -> dict[int, int]:
        """Returns a dictionary mapping each player to their score"""
        return {BLACK: self.score(BLACK), WHITE: self.score(WHITE)}
    
    def as_array(self) -> np.ndarray:
        """Return a read-only view of the internal grid"""
        view = self._grid.view()
        view.flags.writeable = False
        return view
    
    #-----------------------------------------------------------------------
    # Move Logic:
    #-----------------------------------------------------------------------

    def get_flips(self, row: int, col: int, player: int) -> list[tuple[int, int]]:
        """
        Return all positions that would be flipped if 'player' places a piece at (row, col).
        Returns an empty list if the move is illegal.
        """
        if not self.in_bounds(row, col) or not self.is_empty(row, col):
            return []

        opponent = -player
        all_flips: list[tuple[int, int]] = []

        for dr, dc in DIRECTIONS:
            candidates: list[tuple[int, int]] = []
            r, c = row + dr, col + dc

            while self.in_bounds(r, c) and self._grid[r][c] == opponent:
                candidates.append((r, c))
                r += dr
                c += dc

            if candidates and self.in_bounds(r, c) and self._grid[r][c] == player:
                all_flips.extend(candidates)

        return all_flips

    def is_valid_move(self, row: int, col: int, player: int) -> bool:
        """Returns true if placing a piece at (row, col) would be a legal move for 'player'"""
        return bool(self.get_flips(row, col, player))
    
    def get_valid_moves(self, player: int) -> list[tuple[int, int]]:
        """Returns a list of all valid moves for 'player'"""
        return [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.is_valid_move(r, c, player)
        ]
    
    def apply_move(self, row: int, col: int, player: int) -> None:
        """
        Place a piece and flip captured pieces in-place.
        Raises ValueError if the move is illegal.
        """
        flips = self.get_flips(row, col, player)
        if not flips:
            raise ValueError(f"Illegal move: ({row}, {col}) for player {player}")

        self._grid[row][col] = player
        for r, c in flips:
            self._grid[r][c] = player

    #-----------------------------------------------------------------------
    # Cloning:
    #-----------------------------------------------------------------------

    def copy(self) -> "Board":
        """Returns a deep copy of this board"""
        clone = Board.__new__(Board)
        clone._grid = self._grid.copy()
        return clone
    
    #-----------------------------------------------------------------------
    # Debug:
    #-----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Returns a string representation of the board for debugging"""
        symbols = {BLACK: "B", WHITE: "W", EMPTY: "."}
        rows = ["  " + " ".join(str(c) for c in range(BOARD_SIZE))]
        for r in range(BOARD_SIZE):
            row_str = " ".join(symbols[int(self._grid[r][c])] for c in range(BOARD_SIZE))
            rows.append(f"{r} {row_str}")
        return "\n".join(rows)