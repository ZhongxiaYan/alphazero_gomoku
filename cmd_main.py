from config import *
from class_config import *
from game import *

import sys

# input file format must be one move per row (e.g. (3, 4))
def import_game_from_input(game):
    if len(sys.argv) <= 1:
        return
    input_file_name = sys.argv[1]
    with open(input_file_name) as input_file:
        for row in input_file:
            coord = eval(row.rstrip())
            game.transition(coord)

if __name__ == '__main__':
    # players = [AI_REFLEX, AI_REFLEX_CACHED]
    players = [AI_REFLEX_CACHED, AI_REFLEX]
    players = [AI_REFLEX_CACHED, AI_REFLEX_CACHED]
    players = [AI_REFLEX, AI_REFLEX]
    players = [HUMAN_CMD_LINE, AI_MINIMAX]

    display = DISPLAY_COMMAND_LINE

    game = Game(players, display)
    import_game_from_input(game)

    # main loop
    while True:
        game.show_board()
        if game.has_ended():
            break
        game.transition(game.get_input())
