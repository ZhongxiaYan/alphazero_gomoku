from config import *
from class_config import *
from game import *

if __name__ == '__main__':
    players = [HUMAN_CMD_LINE, AI_REFLEX] # two players, both human
    display = DISPLAY_COMMAND_LINE

    game = Game(players, display)

    # main loop
    while True:
        game.show_board()
        game.transition(game.get_input())
        if game.has_ended():
            break
