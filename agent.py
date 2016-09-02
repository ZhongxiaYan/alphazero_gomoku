from config import *

import re

coord_format = re.compile(r'(\d+)[^\d]+(\d+)')
command_format = re.compile(r'^([a-z])\s+(\d+)')

VALID_COMMANDS = set(['u'])

class Agent(object):
    def __init__(self, player_num, display):
        self.player_num = player_num
        self.display = display

    def get_move(self, board, prev_moves, curr_player):
        raise RuntimeError('Unimplemented')

class CommandLineInputAgent(Agent):
    def get_move(self, board, prev_moves, curr_player):
        while True:
            coord_str = input('Enter coordinate in format "(y, x)" or command in format "cmd num": ')
            input_coord = coord_format.findall(coord_str)
            input_command = command_format.findall(coord_str)
            if len(input_coord) == 1:
                y_str, x_str = input_coord[0]
                y, x = coord = int(y_str), int(x_str)
                if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
                    self.display.print_message('Coordinate out of bound!')
                elif coord in board:
                    self.display.print_message('Coordinate already has a piece!')
                else:
                    return coord
            elif len(input_command) == 1:
                command, num_str = input_command[0]
                num = int(num_str)
                if command not in VALID_COMMANDS:
                    self.display.print_message('Invalid command ' + command)
                else:
                    num = min(num, len(prev_moves))
                    return (command, num)
            else:
                self.display.print_message('Incorrect coordinate format!')

class GuiInputAgent(Agent):
    def get_move(self, board, prev_moves, curr_player):
        return self.display.wait_input()
