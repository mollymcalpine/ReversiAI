"""
Menu screen to let the player choose who controls Black and White before starting the game.

Returns a (black_type, white_type) tuple of strings like ("Human", "Minimax"), which main.py
uses to instantiate agents.
"""

import pygame
from game.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    COLOR_BG, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_ACCENT, COLOR_PANEL_BG,
    COLOR_BOARD, FONT_SIZE_LG, FONT_SIZE_MD, FONT_SIZE_SM,
    BLACK, WHITE,
)

AGENT_OPTIONS = ["Human", "Random", "Greedy", "Minimax", "Neural"]

class Menu:
    """Handles the pre-game menu where players choose their agents"""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_lg = pygame.font.SysFont("Arial", FONT_SIZE_LG, bold=True)
        self.font_md = pygame.font.SysFont("Arial", FONT_SIZE_MD)
        self.font_sm = pygame.font.SysFont("Arial", FONT_SIZE_SM)

        self.black_idx = 0 # index into AGENT_OPTIONS
        self.white_idx = 3 # default White to Minimax

    #-----------------------------------------------------------------------
    # Public:
    #-----------------------------------------------------------------------

    def run(self) -> tuple[str, str] | None:
        """
        Block until the player confirms or quits.
        Returns (black_type, white_type) strings, or None if the player quits.
        """
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    result = self._handle_key(event.key)
                    if result is not None:
                        return result
                if event.type == pygame.MOUSEBUTTONDOWN:
                    result = self._handle_click(pygame.mouse.get_pos())
                    if result is not None:
                        return result
            
            self._draw()
            clock.tick(60)
    
    #-----------------------------------------------------------------------
    # Input Handling:
    #-----------------------------------------------------------------------

    def _handle_key(self, key: int) -> tuple[str, str] | None:
        if key == pygame.K_RETURN or key == pygame.K_SPACE:
            return self._selection()
        if key == pygame.K_q or key == pygame.K_ESCAPE:
            return None
        # Arrow keys cycle through options for each side:
        if key == pygame.K_LEFT:
            self.black_idx = (self.black_idx - 1) % len(AGENT_OPTIONS)
        if key == pygame.K_RIGHT:
            self.black_idx = (self.black_idx + 1) % len(AGENT_OPTIONS)
        if key == pygame.K_a:
            self.white_idx = (self.white_idx - 1) % len(AGENT_OPTIONS)
        if key == pygame.K_d:
            self.white_idx = (self.white_idx + 1) % len(AGENT_OPTIONS)
        return None
    
    def _handle_click(self, pos: tuple[int, int]) -> tuple[str, str] | None:
        mx, my = pos
        for i, (x, y, w, h, side) in enumerate(self._option_rects()):
            if x <= mx <= x + w and y <= my <= y + h:
                option_idx = i % len(AGENT_OPTIONS)
                if side == "black":
                    self.black_idx = option_idx
                else:
                    self.white_idx = option_idx
        
        # Start button:
        sx, sy, sw, sh = self._start_rect()
        if sx <= mx <= sx + sw and sy <= my <= sy + sh:
            return self._selection()
        return None
    
    def _selection(self) -> tuple[str, str]:
        return AGENT_OPTIONS[self.black_idx].lower(), AGENT_OPTIONS[self.white_idx].lower()
    
    #-----------------------------------------------------------------------
    # Drawing:
    #-----------------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(COLOR_BG)

        # Title:
        title = self.font_title.render("Reversi", True, COLOR_TEXT)
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 90)))

        subtitle = self.font_sm.render("A Python implementation", True, COLOR_TEXT_MUTED)
        self.screen.blit(subtitle, subtitle.get_rect(center=(WINDOW_WIDTH // 2, 130)))

        # Player selection panels:
        self._draw_selector(
            cx=WINDOW_WIDTH // 4,
            cy=260,
            label="Black ● (goes first)",
            selected_idx=self.black_idx,
            side="black",
        )
        self._draw_selector(
            cx=3 * WINDOW_WIDTH // 4,
            cy=260,
            label="White ○",
            selected_idx=self.white_idx,
            side="white",
        )

        # Controls hint:
        hint = self.font_sm.render("← → to change Black, A D to change White, Enter to start", True, COLOR_TEXT_MUTED)
        self.screen.blit(hint, hint.get_rect(center=(WINDOW_WIDTH // 2, 570)))

        # Start button:
        sx, sy, sw, sh = self._start_rect()
        pygame.draw.rect(self.screen, COLOR_BOARD, (sx, sy, sw, sh), border_radius=10)
        start_surf = self.font_lg.render("Start Game", True, COLOR_TEXT)
        self.screen.blit(start_surf, start_surf.get_rect(center=(sx + sw // 2, sy + sh // 2)))

        pygame.display.flip()

    def _draw_selector(self, cx: int, cy: int, label: str, selected_idx: int, side: str) -> None:
        label_surf = self.font_md.render(label, True, COLOR_TEXT_MUTED)
        self.screen.blit(label_surf, label_surf.get_rect(center=(cx, cy - 60)))

        for i, option in enumerate(AGENT_OPTIONS):
            x = cx - 70
            y = cy - 30 + i * 44
            is_selected = i == selected_idx
            bg = COLOR_BOARD if is_selected else COLOR_PANEL_BG
            color = COLOR_TEXT if is_selected else COLOR_TEXT_MUTED
            pygame.draw.rect(self.screen, bg, (x, y, 140, 36), border_radius=6)
            if is_selected:
                pygame.draw.rect(self.screen, COLOR_ACCENT, (x, y, 140, 36), 2, border_radius=6)
            surf = self.font_md.render(option, True, color)
            self.screen.blit(surf, surf.get_rect(center=(x + 70, y + 18)))

    def _option_rects(self) -> list[tuple[int, int, int, int, str]]:
        """Return (x, y, w, h, side) for every clickable option box"""
        rects = []
        for side, cx in [("black", WINDOW_WIDTH // 4), ("white", 3 * WINDOW_WIDTH // 4)]:
            cy = 260
            for i in range(len(AGENT_OPTIONS)):
                rects.append((cx - 70, cy - 30 + i * 44, 140, 36, side))
        return rects
    
    def _start_rect(self) -> tuple[int, int, int, int]:
        w, h = 220, 52
        x = WINDOW_WIDTH // 2 - w // 2
        y = WINDOW_HEIGHT // 2 + 140
        return x, y, w, h