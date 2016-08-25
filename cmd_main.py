from config import *
from class_config import *
from game import *

if __name__ == '__main__':
    # players = [AI_REFLEX, AI_REFLEX_CACHED]
    players = [AI_REFLEX_CACHED, AI_REFLEX]
    players = [AI_REFLEX_CACHED, AI_REFLEX_CACHED]
    players = [AI_REFLEX, AI_REFLEX]
    players = [HUMAN_CMD_LINE, AI_MINIMAX]

    display = DISPLAY_COMMAND_LINE

    game = Game(players, display)

    # main loop
    while True:
        game.show_board()
        game.transition(game.get_input())
        if game.has_ended():
            break
