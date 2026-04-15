"""
Game state machine.

Responsibilities:
- Track whose turn it is.
- Advance turns (including pass-turn logic).
- Detect game-over conditions.
- Expose a clean interface to the UI and agents.

The Game class owns a Board, and mutates it via apply_move().
"""

from enum import Enum, auto
from game.board import Board
from game.constants import BLACK, WHITE, PLAYER_NAMES

class GameStatus(Enum):
    IN_PROGRESS = auto()
    BLACK_WINS = auto()
    WHITE_WINS = auto()
    DRAW = auto()


class Game:
    """Manages a full game of Reversi from start to finish"""

    def __init__(self) -> None:
        self.board = Board()
        self.current_player = BLACK # Black always goes first.
        self.status = GameStatus.IN_PROGRESS
        self.history: list[tuple[int, int, int]] = [] # (player, row, col) per move; (-1, -1) = pass.

    #-----------------------------------------------------------------------
    # Queries:
    #-----------------------------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.status != GameStatus.IN_PROGRESS
    
    @property
    def valid_moves(self) -> list[tuple[int, int]]:
        return self.board.get_valid_moves(self.current_player)
    
    @property
    def opponent(self) -> int:
        return -self.current_player
    
    def scores(self) -> dict[int, int]:
        return self.board.scores()
    
    def winner_name(self) -> str | None:
        match self.status:
            case GameStatus.BLACK_WINS: return PLAYER_NAMES[BLACK]
            case GameStatus.WHITE_WINS: return PLAYER_NAMES[WHITE]
            case GameStatus.DRAW:       return "Draw"
            case _:                     return None

    #-----------------------------------------------------------------------
    # Mutating Actions:
    #-----------------------------------------------------------------------  

    def apply_move(self, row: int, col: int) -> None:
        """
        Apply a move for the current player and advance the game state.
        Raises ValueError if the move is invalid.
        """
        if self.is_over:
            raise RuntimeError("Cannot make a move: game is already over.")
        
        if not self.board.is_valid_move(row, col, self.current_player):
            raise ValueError(f"({row}, {col}) is not a valid move for {PLAYER_NAMES[self.current_player]}")
        
        self.board.apply_move(row, col, self.current_player)
        self.history.append((self.current_player, row, col))
        self._advance_turn()

    def _advance_turn(self) -> None:
        """
        Switch to the opponent if they have moves.
        If only the current player has moves, keep the current player.
        If neither player has moves, end the game.
        """
        opponent_moves = self.board.get_valid_moves(self.opponent)
        
        if opponent_moves:
            self.current_player = self.opponent
            return
        
        # Opponent must pass (check if current player still has moves):
        current_moves = self.board.get_valid_moves(self.current_player)
        if current_moves:
            # Current player gets another turn (opponent has no moves):
            self.history.append((self.opponent, -1, -1)) # Record the forced pass.
            return
        
        # Neither player has moves (game over):
        self._resolve_winner()

    def _resolve_winner(self) -> None:
        """Count pieces and set the game status to BLACK_WINS, WHITE_WINS, or DRAW"""
        scores = self.board.scores()
        if scores[BLACK] > scores[WHITE]:
            self.status = GameStatus.BLACK_WINS
        elif scores[WHITE] > scores[BLACK]:
            self.status = GameStatus.WHITE_WINS
        else:
            self.status = GameStatus.DRAW

    #-----------------------------------------------------------------------
    # Debug:
    #-----------------------------------------------------------------------

    def __repr__(self) -> str:
        scores = self.scores()
        return (
            f"Game(turn={PLAYER_NAMES[self.current_player]}, "
            f"B={scores[BLACK]}, W={scores[WHITE]}, "
            f"status={self.status.name})"
        )
