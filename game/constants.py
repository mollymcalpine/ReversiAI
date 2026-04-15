"""
Game-wide constants.
All magic numbers and configuration lives here.
"""

# Board:
BOARD_SIZE = 8

# Players:
BLACK = 1
WHITE = -1
EMPTY = 0

PLAYER_NAMES = {BLACK: "Black", WHITE: "White"}
PLAYER_SYMBOLS = {BLACK: "●", WHITE: "○"}

# Directions for flip detection (row_delta, col_delta):
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           (0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

# Display:
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 720
CELL_SIZE = 70
BOARD_OFFSET_X = (WINDOW_WIDTH - BOARD_SIZE * CELL_SIZE) // 2
BOARD_OFFSET_Y = (WINDOW_HEIGHT - BOARD_SIZE * CELL_SIZE) // 2
PIECE_RADIUS = CELL_SIZE // 2 - 6

# Colors (R, G, B):
COLOR_BG = (30, 30, 30)
COLOR_BOARD = (34, 85, 34)
COLOR_BOARD_DARK = (28, 72, 28)
COLOR_GRID = (20, 60, 20)
COLOR_BLACK_PIECE = (20, 20, 20)
COLOR_WHITE_PIECE = (235, 235, 235)
COLOR_HINT = (80, 160, 80)
COLOR_HINT_OUTLINE = (60, 130, 60)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_MUTED = (140, 140, 140)
COLOR_ACCENT = (100, 180, 100)
COLOR_PANEL_BG = (20, 20, 20)

# UI:
HUD_HEIGHT = 80
FONT_SIZE_LG = 28
FONT_SIZE_MD = 20
FONT_SIZE_SM = 16

# Minimax:
DEFAULT_MINIMAX_DEPTH = 5

# Frame rate:
FPS = 60