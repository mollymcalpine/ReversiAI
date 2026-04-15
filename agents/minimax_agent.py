"""
Minimax agent with alpha-beta pruning.

Heuristic:
- A pure piece-count heuristic loses badly to positional play because corners and edges
  are disporportionately valuable.
- This agent uses a weighted position table that strongly rewards corners, and penalizes
  the cells adjacent to them (which hands corners to the opponent).
  
The search also blends:
- Positional score (weighted piece placement).
- Mobility (number of legal moves available).
- Piece parity (raw piece count, weighted more heavily late in the game)

Depth:
- Default depth is imported from constants.
- The agent automatically deepens near the end of the game when the tree is small.
"""

from game.board import Board
from game.constants import BOARD_SIZE, DEFAULT_MINIMAX_DEPTH
from agents.base_agent import Agent

# Positional weights (higher is better for the maximizing player).
# Corners are the most valuable, and cells diagonally adjacent to corners are traps.
_WEIGHTS = [
    [ 100, -20, 10,  5,  5, 10, -20, 100],
    [ -20, -50, -2, -2, -2, -2, -50, -20],
    [  10,  -2,  5,  1,  1,  5,  -2,  10],
    [   5,  -2,  1,  0,  0,  1,  -2,   5],
    [   5,  -2,  1,  0,  0,  1,  -2,   5],
    [  10,  -2,  5,  1,  1,  5,  -2,  10],
    [ -20, -50, -2, -2, -2, -2, -50, -20],
    [ 100, -20, 10,  5,  5, 10, -20, 100],
]

def _positional_score(board: Board, player: int) -> int:
    opponent = -player
    total = 0
    grid = board.as_array()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r][c] == player:
                total += _WEIGHTS[r][c]
            elif grid[r][c] == opponent:
                total -= _WEIGHTS[r][c]
    return total

def _mobility_score(board: Board, player: int) -> int:
    my_moves = len(board.get_valid_moves(player))
    opp_moves = len(board.get_valid_moves(-player))
    if my_moves + opp_moves == 0:
        return 0
    return 100 * (my_moves - opp_moves) // (my_moves + opp_moves)

def _parity_score(board: Board, player: int) -> int:
    scores = board.scores()
    my_pieces = scores[player]
    opp_pieces = scores[-player]
    if my_pieces + opp_pieces == 0:
        return 0
    return 100 * (my_pieces - opp_pieces) // (my_pieces + opp_pieces)

def _evaluate(board: Board, player: int) -> float:
    pieces_played = sum(board.scores().values())

    # Weight piece parity more heavily as the board fills up:
    parity_weight = pieces_played / (BOARD_SIZE * BOARD_SIZE)

    return (_positional_score(board, player) * 1.0 +
            _mobility_score(board, player) * 0.5 +
            _parity_score(board, player)   * parity_weight * 2.0
    )

class MinimaxAgent(Agent):
    def __init__(self, player: int, name: str = "Minimax", depth: int = DEFAULT_MINIMAX_DEPTH) -> None:
        super().__init__(player, name)
        self.depth = depth
    
    def choose_move(self, board: Board, valid_moves: list[tuple[int, int]]) -> tuple[int, int]:
        # Deepen automatically when only a few empty cells remain:
        empty_cells = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board.is_empty(r, c))
        if empty_cells <= 12:
            depth = max(self.depth, 10)
        else:
            depth = self.depth

        best_move = valid_moves[0]
        best_score = float("-inf")

        for row, col in valid_moves:
            child = board.copy()
            child.apply_move(row, col, self.player)
            score = self._minimax(child, depth - 1, float("-inf"), float("inf"), -self.player)
            if score > best_score:
                best_score = score
                best_move = (row, col)
            
        return best_move

    #-----------------------------------------------------------------------
    # Core Search:
    #-----------------------------------------------------------------------

    def _minimax(self, board, depth, alpha, beta, current_player: int):
        moves = board.get_valid_moves(current_player)

        if depth == 0 or not moves:
            if not moves:
                other_moves = board.get_valid_moves(-current_player)
                if other_moves:
                    return self._minimax(board, depth, alpha, beta, -current_player)
            return _evaluate(board, self.player)
        
        if current_player == self.player: # Maximizing.
            value = float("-inf")
            for row, col in moves:
                child = board.copy()
                child.apply_move(row, col, current_player)
                value = max(value, self._minimax(child, depth - 1, alpha, beta, -current_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else: # Minimizing.
            value = float("inf")
            for row, col in moves:
                child = board.copy()
                child.apply_move(row, col, current_player)
                value = min(value, self._minimax(child, depth - 1, alpha, beta, -current_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
        
    """
    def _minimax(
            self, 
            board: Board, 
            depth: int,
            alpha: float,
            beta: float,
            is_maximizing: bool,
    ) -> float:
        if is_maximizing:
            current_player = self.player 
        else:
            current_player = -self.player
        
        moves = board.get_valid_moves(current_player)

        if depth == 0 or not moves:
            # Check if the other player can move (forced pass):
            if not moves:
                other_moves = board.get_valid_moves(-current_player)
                if other_moves:
                    # Pass (flip the maximizing flag):
                    return self._minimax(board, depth, alpha, beta, not is_maximizing)
            return _evaluate(board, self.player)
        
        if is_maximizing:
            value = float("-inf")
            for row, col in moves:
                child = board.copy()
                child.apply_move(row, col, current_player)
                value = max(value, self._minimax(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break # Beta cut-off.
            return value
        else:
            value = float("inf")
            for row, col in moves:
                child = board.copy()
                child.apply_move(row, col, current_player)
                value = min(value, self._minimax(child, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break # Alpha cut-off.
            return value
    """
