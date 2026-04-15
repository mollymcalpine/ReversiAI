"""
Entry point.

Responsibilities:
  	1. Show the menu and get the player type selections.
  	2. Instantiate the chosen agents.
  	3. Run the game loop:
    	- Poll pygame events.
		- If its a human's turn, wait for a click on a valid cell.
		- If it's an AI's turn, call choose_move() and apply it.
		- Re-render for each frame.
	4. Once game-over, show the overlay and wait for restart or quit.
"""

import sys
import threading
import pygame

from game.game import Game, GameStatus
from game.constants import (WINDOW_WIDTH, WINDOW_HEIGHT, FPS, BLACK, WHITE, PLAYER_NAMES, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE, BOARD_SIZE)

from agents.base_agent import Agent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.minimax_agent import MinimaxAgent
from agents.neural_agent import NeuralAgent

from ui.renderer import Renderer
from ui.menu import Menu

#-----------------------------------------------------------------------
# Agent Factory:
#-----------------------------------------------------------------------

def make_agent(choice: str, player: int) -> Agent:
    name = PLAYER_NAMES[player]
    match choice:
        case "human": return HumanAgent(player, name)
        case "random": return RandomAgent(player, f"Random ({name})")
        case "greedy": return GreedyAgent(player, f"Greedy({name})")
        case "minimax": return MinimaxAgent(player, f"Minimax ({name})")
        case "neural": return NeuralAgent(player, f"Neural ({name})")
        case _:
        	raise ValueError(f"Unknown agent type: {choice}")
        
#-----------------------------------------------------------------------
# Board → Cell Coversion:
#-----------------------------------------------------------------------

def pixel_to_cell(mx: int, my: int) -> tuple[int, int] | None:
    """Coverts a mouse click position (in pixels) to a board cell (row, col)"""
    col = (mx - BOARD_OFFSET_X) // CELL_SIZE
    row = (my - BOARD_OFFSET_Y) // CELL_SIZE
    # Bounds check:
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None

#-----------------------------------------------------------------------
# Game Loop:
#-----------------------------------------------------------------------

def run_game(screen: pygame.Surface, renderer: Renderer, black_agent: Agent, white_agent: Agent) -> bool:
    """
    Runs one game. 
    Returns True if the player wants to play again, and False if they want to quit.
    """
    
    game = Game()
    clock = pygame.time.Clock()
    agents = {BLACK: black_agent, WHITE: white_agent}

    # Player labels for the HUD:
    player_labels = {BLACK: black_agent.name, WHITE: white_agent.name}

    # AI move is computed off the main thread so that the UI stays responsive:
    ai_move_result: list[tuple[int, int] | None] = [None]
    ai_thinking = False
    
    def ai_think(agent: Agent, board_copy, moves_copy):
        move = agent.choose_move(board_copy, moves_copy)
        ai_move_result[0] = move
    
    while True:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                     return False
        
            if not game.is_over:
                current_agent = agents[game.current_player]
                if isinstance(current_agent, HumanAgent):
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        cell = pixel_to_cell(*pygame.mouse.get_pos())
                        if cell and cell in game.valid_moves:
                            current_agent.set_move(*cell)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True # Restart
        
        if game.is_over:
            renderer.draw_game_over(
                winner_text=game.winner_name() or "Game Over",
                scores=game.scores()
			)
            continue
        
        current_agent = agents[game.current_player]
        
		# Human turn (apply move if one is pending):
        if isinstance(current_agent, HumanAgent):
            if current_agent.has_pending_move():
                try:
                    game.apply_move(*current_agent.choose_move(game.board, game.valid_moves))
                except ValueError:
                    pass # Illegal move (shouldn't happen since we check valid_moves).
        
		# AI turn (kick off background thread if not already thinking):
        else:
            if not ai_thinking:
                ai_move_result[0] = None
                ai_thinking = True
                board_copy = game.board.copy()
                moves_copy = list(game.valid_moves)
                t = threading.Thread(
                    target=ai_think,
                    args=(current_agent, board_copy, moves_copy),
                    daemon=True
				)
                t.start()
                
            if ai_move_result[0] is not None:
                move = ai_move_result[0]
                ai_move_result[0] = None
                ai_thinking = False
                try:
                    game.apply_move(*move)
                except ValueError as e:
                    print(f"Agent returned an illegal move: {e}")
        
		# Determine status text:
        if game.is_over:
            status_text = game.winner_name() or "Game Over"
        elif isinstance(agents[game.current_player], HumanAgent):
            status_text = "Your Turn"
        else:
            status_text = "Thinking..." if ai_thinking else "AI's Turn"
        
        renderer.draw(
            board=game.board,
            current_player=game.current_player,
            valid_moves=game.valid_moves if not game.is_over else [],
            scores=game.scores(),
            status_text=status_text,
            player_labels=player_labels,
        )
            
#-----------------------------------------------------------------------
# Entry Point:
#-----------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Reversi")
    renderer = Renderer(screen)
    menu = Menu(screen)
    
    while True:
        result = menu.run()
        if result is None:
            break
        
        black_choice, white_choice = result
        black_agent = make_agent(black_choice, BLACK)
        white_agent = make_agent(white_choice, WHITE)
        
        play_again = run_game(screen, renderer, black_agent, white_agent)
        if not play_again:
            break

    pygame.quit()
    sys.exit()
    
if __name__ == "__main__":
    main()
