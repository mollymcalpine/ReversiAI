"""
Renders the board, pieces, hints, and HUD using pygame.

The renderer is purely visual.
It reads the game state and draws it.
It does not modify game state or handle events.
"""

import pygame
from game.board import Board
from game.constants import (
    BOARD_SIZE, CELL_SIZE, PIECE_RADIUS,
    BOARD_OFFSET_X, BOARD_OFFSET_Y,
    BLACK, WHITE, EMPTY, 
    COLOR_BG, COLOR_BOARD, COLOR_BOARD_DARK, COLOR_GRID,
    COLOR_BLACK_PIECE, COLOR_WHITE_PIECE, 
    COLOR_HINT, COLOR_HINT_OUTLINE, 
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_ACCENT, COLOR_PANEL_BG,
    WINDOW_WIDTH, WINDOW_HEIGHT, HUD_HEIGHT,
    FONT_SIZE_LG, FONT_SIZE_MD, FONT_SIZE_SM,
    PLAYER_NAMES,
)

class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.font_lg = pygame.font.SysFont("Arial", FONT_SIZE_LG, bold=True)
        self.font_md = pygame.font.SysFont("Arial", FONT_SIZE_MD)
        self.font_sm = pygame.font.SysFont("Arial", FONT_SIZE_SM)

    #-----------------------------------------------------------------------
    # Public Entry Point:
    #-----------------------------------------------------------------------

    def draw(
            self,
            board: Board,
            current_player: int,
            valid_moves: list[tuple[int, int]],
            scores: dict[int, int],
            status_text: str,
            player_labels: dict[int, str],
    ) -> None:
        self.screen.fill(COLOR_BG)
        self._draw_board(board)
        self._draw_hints(valid_moves)
        self._draw_pieces(board)
        self._draw_hud(current_player, scores, status_text, player_labels)
        pygame.display.flip()

    #-----------------------------------------------------------------------
    # Board:
    #-----------------------------------------------------------------------

    def _draw_board(self, board: Board) -> None:
        """Draw the checkerboard pattern and grid lines"""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = BOARD_OFFSET_X + col * CELL_SIZE
                y = BOARD_OFFSET_Y + row * CELL_SIZE
                if ((row + col) % 2 == 0):
                    color = COLOR_BOARD
                else:
                    color = COLOR_BOARD_DARK
                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, COLOR_GRID, (x, y, CELL_SIZE, CELL_SIZE), 1)
        
        # Start points:
        for r, c in [(2, 2), (2, 5), (5, 2), (5, 5)]:
            cx = BOARD_OFFSET_X + c * CELL_SIZE + CELL_SIZE // 2
            cy = BOARD_OFFSET_Y + r * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, COLOR_GRID, (cx, cy), 4)
    
    def _draw_hints(self, valid_moves: list[tuple[int, int]]) -> None:
        """Draw circle hints on valid moves"""
        for row, col in valid_moves:
            cx = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
            cy = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, COLOR_HINT_OUTLINE, (cx, cy), 14)
            pygame.draw.circle(self.screen, COLOR_HINT, (cx, cy), 12)

    def _draw_pieces(self, board: Board) -> None:
        """Draw pieces on the board according to the board state"""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board.get(row, col)
                if piece == EMPTY:
                    continue
                cx = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
                cy = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
                if piece == BLACK:
                    color = COLOR_BLACK_PIECE
                else: 
                    color = COLOR_WHITE_PIECE
                
                # Shadow:
                pygame.draw.circle(self.screen, (0, 0, 0), (cx + 2, cy + 2), PIECE_RADIUS)

                # Piece:
                pygame.draw.circle(self.screen, color, (cx, cy), PIECE_RADIUS)

                # Highlight sheen:
                if piece == BLACK:
                    highlight = (80, 80, 80)
                else:
                    highlight = (255, 255, 255)
                pygame.draw.circle(self.screen, highlight, (cx - 8, cy - 8), PIECE_RADIUS // 4)

        
    #-----------------------------------------------------------------------
    # HUD (Heads-Up Display):
    #-----------------------------------------------------------------------

    def _draw_hud(
            self, 
            current_player: int, 
            scores: dict[int, int], 
            status_text: str, 
            player_labels: dict[int, str],
    ) -> None:
        hud_y = WINDOW_HEIGHT - HUD_HEIGHT
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, (0, hud_y, WINDOW_WIDTH, HUD_HEIGHT))
        pygame.draw.line(self.screen, COLOR_ACCENT, (0, hud_y), (WINDOW_WIDTH, hud_y), 1)

        # Black score (left):
        self._draw_score_block(
            x=40, y=hud_y + 14,
            player=BLACK,
            score=scores[BLACK],
            label=player_labels[BLACK],
            is_active=(current_player == BLACK),
        )

        # White score (right):
        self._draw_score_block(
            x=WINDOW_WIDTH - 200, y=hud_y + 14,
            player=WHITE,
            score=scores[WHITE],
            label=player_labels[WHITE],
            is_active=(current_player == WHITE),
        )

        # Status / turn text (center):
        surf = self.font_md.render(status_text, True, COLOR_TEXT)
        rect = surf.get_rect(center=(WINDOW_WIDTH // 2, hud_y + HUD_HEIGHT // 2))
        self.screen.blit(surf, rect)
    
    def _draw_score_block(
            self,
            x: int, 
            y: int,
            player: int,
            score: int,
            label: str,
            is_active: bool,
    ) -> None:
        if player == BLACK:
            piece_color = COLOR_BLACK_PIECE
        else:
            piece_color = COLOR_WHITE_PIECE
        pygame.draw.circle(self.screen, piece_color, (x + 16, y + 20), 14)
    
        if is_active:
            text_color = COLOR_TEXT
        else:
            text_color = COLOR_TEXT_MUTED
        name_surf = self.font_sm.render(label, True, text_color)
        score_surf = self.font_lg.render(str(score), True, text_color)

        self.screen.blit(name_surf, (x + 38, y + 4))
        self.screen.blit(score_surf, (x + 38, y + 22))

        if is_active:
            indicator_surf = self.font_sm.render("▶ Your Turn", True, COLOR_ACCENT)
            self.screen.blit(indicator_surf, (x + 38, y + 46))

    #-----------------------------------------------------------------------
    # Overlays:
    #-----------------------------------------------------------------------

    def draw_game_over(self, winner_text: str, scores: dict[int, int]) -> None:
        """Semi-transparent overlay shown when the game ends"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        title_surf = self.font_lg.render(winner_text, True, COLOR_TEXT)
        score_surf = self.font_md.render(f"Black {scores[BLACK]} - {scores[WHITE]} White", True, COLOR_TEXT_MUTED)
        hint_surf = self.font_sm.render("Press R to restart  ·  Q to quit", True, COLOR_TEXT_MUTED)

        cx = WINDOW_WIDTH // 2
        cy = WINDOW_HEIGHT // 2
        self.screen.blit(title_surf, title_surf.get_rect(center=(cx, cy - 40)))
        self.screen.blit(score_surf, score_surf.get_rect(center=(cx, cy + 4)))
        self.screen.blit(hint_surf, hint_surf.get_rect(center=(cx, cy + 44)))

        pygame.display.flip()






