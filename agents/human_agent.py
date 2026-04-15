"""
A human agent.

Rather than blocking on input, the HumanAgent holds a pending move 
that the UI sets when the player clicks a cell.

The game loop calls 'choose_move' only after the UI has confirmed a selection, 
so this never actually blocks.
"""

from agents.base_agent import Agent
from game.board import Board

class HumanAgent(Agent):
    """Represents a human player, and the UI injects the chosen move"""

    def __init__(self, player: int, name: str = "Human") -> None:
        super().__init__(player, name)
        self._pending: tuple[int, int] | None = None

    def set_move(self, row: int, col: int) -> None:
        """Called by the UI when the player clicks a valid cell"""
        self._pending = (row, col)
    
    def has_pending_move(self) -> bool:
        return self._pending is not None
    
    def choose_move(self, board: Board, valid_moves: list[tuple[int, int]]) -> tuple[int, int]:
        """
        Return the pending move that was set by the UI.
        Raises RuntimeError if called before a move has been set.
        """
        if self._pending is None:
            raise RuntimeError("choose_move called before a move was set")
        move, self._pending = self._pending, None
        return move
