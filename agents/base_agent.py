"""
Abstract base class for all Reversi agents.

Every agent (human, random, greedy, minimax) implements the same 'choose_move' interface.
This lets the game loop treat all player types uniformly, without any special-casing.
"""

from abc import ABC, abstractmethod
from game.board import Board

class Agent(ABC):
    """
    Base class for all agents (human or AI).
    
    Subclasses must implement 'choose_move', which receives the current board
    state and list of legal moves, and returns one (row, col) pair.
    """

    def __init__(self, player: int, name: str) -> None:
        self.player = player
        self.name = name

    @abstractmethod
    def choose_move(self, board: Board, valid_moves: list[tuple[int, int]]) -> tuple[int, int]:
        """
        Select a move from 'valid_moves'.
        
        Args:
            board: Current board state (do not modify it).
            valid_moves: Non-empty list of legal (row, col) moves.
            
        Returns one (row, col) from valid_moves.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
