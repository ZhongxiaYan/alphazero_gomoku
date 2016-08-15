from config import *

import re

input_format = re.compile(r'(\d+)[^\d]+(\d+)')

class Agent(object):
    def __init__(self, player_num, display):
        self.player_num = player_num
        self.display = display

    def get_move(self, board, prev_moves, curr_player):
        raise RuntimeError('Unimplemented')

class CommandLineInputAgent(Agent):
    def get_move(self, board, prev_moves, curr_player):
        while True:
            coord_str = input('Enter coordinate in format (y, x): ')
            input_coord = input_format.findall(coord_str)
            if len(input_coord) != 1:
                self.display.print_message('Incorrect coordinate format!')
                continue
            y_str, x_str = input_coord[0]
            y, x = coord = int(y_str), int(x_str)
            if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
                self.display.print_message('Coordinate out of bound!')
            elif coord in board:
                self.display.print_message('Coordinate already has a piece!')
            else:
                break
        return coord

class GuiInputAgent(Agent):
    def get_move(self, board, prev_moves, curr_player):
        return self.display.wait_coord()
