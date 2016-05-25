from config import *

class Agent(object):
    def __init__(self, player_num, display):
        self.player_num = player_num
        self.display = display

    def get_move(self):
        raise RuntimeError('Unimplemented')

class CommandLineInputAgent(Agent):
    def get_move(self, board, prev_moves):
        coord_str = input('Enter coordinate in format (x, y): ')
        coord = eval(coord_str)
        return coord