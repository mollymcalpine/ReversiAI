"""
A self play data generator.

Plays NUM_GAMES games of Minimax vs. Minimax, and records every board state seen during each game.
The board states are labelled with the final outcome:
- +1 = Black won
- -1 = White won
-  0 = Draw

Each board state is encoded as a (3, 8, 8) tensor, with 3 channels:
- Channel 0: cells occupied by Black (where 1 = black, 0 = elsewhere).
- Channel 1: cells occupied by White (where 1 = white, 0 = elsewhere).
- Channel 2: uoccupied cells (where 1 = empty, 0 = elsewhere).

Output: 
- data/board.npy: shape (N, 3, 8, 8), float32
- data/outcomes.npy: shape (N,), float32

Run this from the ReversiAi/ root directory:
    python scripts/generate_data.py
"""

import sys
import os
import numpy as np
import time

# Allow imports from the project root:
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game import Game
from game.board import Board
from game.constants import BLACK, WHITE, BOARD_SIZE
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent

#-----------------------------------------------------------------------
# Configuration:
#-----------------------------------------------------------------------

NUM_GAMES = 200
SEARCH_DEPTH = 2 # Depth 3 keeps generation fast, while still playing well.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#-----------------------------------------------------------------------
# Board Encoding:
#-----------------------------------------------------------------------

def encode_board(board: Board) -> np.ndarray:
    """
    Encode a board as a (3, 8, 8) float32 array:
    - Channel 0: Black pieces.
    - Channel 1: White pieces.
    - Channel 2: Empty squares.
    """
    grid = board.as_array()
    encoded = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    encoded[0] = (grid == 1).astype(np.float32) # Black
    encoded[1] = (grid == -1).astype(np.float32) # White
    encoded[2] = (grid == 0).astype(np.float32) # Empty
    return encoded

#-----------------------------------------------------------------------
# Game Simulation:
#-----------------------------------------------------------------------

def play_game(black_agent, white_agent) -> tuple[list[np.ndarray], float]:
    """
    Play one complete game.
    Returns (board_states, outcome), where outcome is +1, -1, or 0 (from Black's perspective).
    """
    game = Game()
    agents = {BLACK: black_agent, WHITE: white_agent}
    states = []

    while not game.is_over:
        states.append(encode_board(game.board))
        agent = agents[game.current_player]
        move = agent.choose_move(game.board, game.valid_moves)
        game.apply_move(*move)

    scores = game.scores()
    if scores[BLACK] > scores[WHITE]:
        outcome = 1.0
    elif scores[BLACK] < scores[WHITE]:
        outcome = -1.0
    else:
        outcome = 0.0

    return states, outcome

#-----------------------------------------------------------------------
# Main:
#-----------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_boards: list[np.ndarray] = []
    all_outcomes: list[float] = []

    print(f"Generating {NUM_GAMES:,} games (Minimax has depth {SEARCH_DEPTH})...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    start_time = time.time()

    for i in range(NUM_GAMES):
        # Alternating agents for better data:
        if i % 9 == 0:
            print(f"Running game {i+1} (Minimax (black) vs. Random (white))... \n")
            black_agent = MinimaxAgent(BLACK, depth=SEARCH_DEPTH)
            white_agent = RandomAgent(WHITE)
        elif i % 9 == 1:
            print(f"Running game {i+1} (Random (black) vs. Minimax (white))... \n")
            black_agent = RandomAgent(BLACK)
            white_agent = MinimaxAgent(WHITE, depth=SEARCH_DEPTH)
        elif i % 9 == 2:
            print(f"Running game {i+1} (Random vs. Random)... \n")
            black_agent = RandomAgent(BLACK)
            white_agent = RandomAgent(WHITE)
        elif i % 9 == 3:
            print(f"Running game {i+1} (Random (black) vs. Greedy (white))... \n")
            black_agent = RandomAgent(BLACK)
            white_agent = GreedyAgent(WHITE)
        elif i % 9 == 4:
            print(f"Running game {i+1} (Greedy (black) vs. Random (white))... \n")
            black_agent = GreedyAgent(BLACK)
            white_agent = RandomAgent(WHITE)
        elif i % 9 == 5:
            print(f"Running game {i+1} (Greedy vs. Greedy)... \n")
            black_agent = GreedyAgent(BLACK)
            white_agent = GreedyAgent(WHITE)
        elif i % 9 == 6:
            print(f"Running game {i+1} (Greedy (black) vs. Minimax (white))... \n")
            black_agent = GreedyAgent(BLACK)
            white_agent = MinimaxAgent(WHITE, depth=SEARCH_DEPTH)
        elif i % 9 == 7:
            print(f"Running game {i+1} (Minimax (black) vs. Greedy (white))... \n")
            black_agent = MinimaxAgent(BLACK, depth=SEARCH_DEPTH)
            white_agent = GreedyAgent(WHITE)
        else:
            print(f"Running game {i+1} (Minimax vs. Minimax)... \n")
            black_agent = MinimaxAgent(BLACK, depth=SEARCH_DEPTH)
            white_agent = MinimaxAgent(WHITE, depth=SEARCH_DEPTH)
        
        states, outcome = play_game(black_agent, white_agent)
        print(f"Game {i+1}: outcome={outcome}") # !!!
        all_boards.extend(states)
        all_outcomes.extend([outcome] * len(states))

        if (i + 1) % 2 == 0:
            elapsed = time.time() - start_time
            games_done = i + 1
            games_left = NUM_GAMES - games_done
            avg_per_game = elapsed / games_done
            eta_seconds = avg_per_game * games_left
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)
            pct = games_done / NUM_GAMES * 100
            # Print the number of games ran so far, and the ETA:
            print(f"  {games_done:>5,} / {NUM_GAMES:,}   ({pct:.0f}%)  -  ETA: {eta_min}m {eta_sec}s")
    
    boards_arr = np.array(all_boards, dtype=np.float32)
    outcomes_arr = np.array(all_outcomes, dtype=np.float32)

    boards_path = os.path.join(OUTPUT_DIR, "boards.npy")
    outcomes_path = os.path.join(OUTPUT_DIR, "outcomes.npy")

    np.save(boards_path, boards_arr)
    np.save(outcomes_path, outcomes_arr)

    print(f"\nDone.")
    print(f"  Positions saved : {len(all_boards):,}")
    print(f"  boards.npy      : {boards_path}")
    print(f"  outcomes.npy    : {outcomes_path}")

    # Quick sanity check on label distribution:
    wins = int(np.sum(outcomes_arr == 1.0))
    losses = int(np.sum(outcomes_arr == -1.0))
    draws = int(np.sum(outcomes_arr == 0.0))
    total = len(outcomes_arr)
    print(f"\n Label distribution (from Black's perspective):")
    print(f"  Black wins : {wins:>7,}  ({wins/total*100:.1f}%)")
    print(f"  White wins : {losses:>7,}  ({losses/total*100:.1f}%)")
    print(f"  Draws      : {draws:>7,}  ({draws/total*100:.1f}%)")

if __name__ == "__main__":
    main()
    