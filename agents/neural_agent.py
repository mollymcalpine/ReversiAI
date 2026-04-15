"""
Neural network agent.

Loads the trained ReversiNet model, and uses it to evaluate board positions.
For each legal move, it:
1. Applies the move to a copy of the board.
2. Passes the resulting position through the network.
3. Picks the move with the highest score (from the current player's perspective).

The network always outputs a score from Black's perspective (+1 = Black winning, -1 = White winning).
If it's White's turn, we negate the score so that White also picks the move that is best for itself.
"""

import os
import sys
import numpy as np

# Allow running standalone for testing:
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from agents.base_agent import Agent
from game.board import Board
from game.constants import BLACK, WHITE, BOARD_SIZE

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "reversi_net.pt")

#-----------------------------------------------------------------------
# Network Definition (must match train.py exactly):
#-----------------------------------------------------------------------

class ReversiNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
    
#-----------------------------------------------------------------------
# Board Encoding (must match generate_data.py exactly):
#-----------------------------------------------------------------------

def encode_board(board: Board) -> np.ndarray:
    grid = board.as_array()
    encoded = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    encoded[0] = (grid == 1).astype(np.float32) # Black.
    encoded[1] = (grid == -1).astype(np.float32) # White
    encoded[2] = (grid == 0).astype(np.float32) # Empty
    return encoded

#-----------------------------------------------------------------------
# Agent:
#-----------------------------------------------------------------------

class NeuralAgent(Agent):
    """
    Evaluates each legal move using a trained CNN, and picks the best one.
    Falls back to a random choice if the model file is not found.
    """

    def __init__(self, player: int, name: str = "Neural", model_path: str = MODEL_PATH) -> None:
        super().__init__(player, name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)

    def _load_model(self, path: str) -> ReversiNet | None:
        if not os.path.exists(path):
            print(f"[NeuralAgent] Warning: model not found at {path}")
            print(f"[NeuralAgent] Run 'python scripts/generate_data.py', then 'python scripts/train.py'")
            return None
        
        model = ReversiNet().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        print(f"[NeuralAgent] Loaded model from {path}")
        return model
    
    def choose_move(self, board: Board, valid_moves: list[tuple[int, int]]) -> tuple[int, int]:
        if self.model is None:
            # Fallback (pick a random move):
            import random
            return random.choice(valid_moves)
        
        best_move = valid_moves[0]
        best_score = float("-inf")

        # Batch all candidate positions for a single forward pass:
        candidate_boards = []
        for row, col in valid_moves:
            child = board.copy()
            child.apply_move(row, col, self.player)
            candidate_boards.append(encode_board(child))

        batch = torch.tensor(np.array(candidate_boards), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores = self.model(batch).squeeze(1).cpu().numpy() # Shape: (num_moves,)

        # Network scores from Black's perspective (negate if we are White):
        if self.player == WHITE:
            scores = -scores

        best_idx = int(np.argmax(scores))
        best_move = valid_moves[best_idx]

        return best_move
