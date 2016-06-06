from config import *

class Agent(object):
    def __init__(self, player_num, display):
        self.player_num = player_num
        self.display = display

    def get_move(self, board, prev_moves, curr_player):
        raise RuntimeError('Unimplemented')

class CommandLineInputAgent(Agent):
    def get_move(self, board, prev_moves, curr_player):
        while True:
            coord_str = input('Enter coordinate in format (x, y): ')
            coord = eval(coord_str)
            x, y = coord
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
