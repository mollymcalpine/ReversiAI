"""
Greedy agent that maximizes the number of pieces flipped in one move.

Ties are broken randomly so that the agent isn't deterministic on equal moves.
"""

import random
from agents.base_agent import Agent
from game.board import Board

class GreedyAgent(Agent):
    def __init__(self, player: int, name: str = "Greedy") -> None:
        super().__init__(player, name)

    def choose_move(self, board: Board, valid_moves: list[tuple[int, int]]) -> tuple[int, int]:
        best_moves: list[tuple[int, int]] = []
        best_count = -1

        for row, col in valid_moves:
            flips = len(board.get_flips(row, col, self.player))
            if flips > best_count:
                best_count = flips
                best_moves = [(row, col)]
            elif flips == best_count:
                best_moves.append((row, col))

        return random.choice(best_moves)
    
    