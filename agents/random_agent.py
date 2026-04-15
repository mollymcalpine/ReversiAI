"""
A random agent that picks a uniformly random legal move.

Useful as a baseline for testing other agents against. 
"""

import random
from agents.base_agent import Agent
from game.board import Board

class RandomAgent(Agent):
    def __init__(self, player: int, name: str = "Random") -> None:
        super().__init__(player, name)

    def choose_move(self, board: Board, valid_moves: list[tuple[int, int]]) -> tuple[int, int]:
        return random.choice(valid_moves)
