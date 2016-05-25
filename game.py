from config import *
from agent import Agent, CommandLineInputAgent
from display import CommandLineDisplay

from pprint import pprint

# maps player type to agent class
TYPE_TO_AGENT = {
    HUMAN : CommandLineInputAgent,
    AI : Agent
}

# maps display type to display class
TYPE_TO_DISPLAY = {
    DISPLAY_COMMAND_LINE : CommandLineDisplay
}

class Game:
    def __init__(self, player_types, display_type):
        self.board = {}
        self.player_types = player_types
        self.num_players = len(player_types)
        self.curr_player = 0
        self.moves = []
        self.has_ended = lambda: False

        self.display = TYPE_TO_DISPLAY[display_type](self.board, self.moves)
        self.agents = [TYPE_TO_AGENT[player_type](player_num, self.display) for player_num, player_type in enumerate(player_types)]

    def show_board(self):
        self.display.update_board()

    def transition(self):
        agent = self.agents[self.curr_player] # get the next player

        self.display.print_message('Player %s to move!' % (self.curr_player))
        while True:
            coord = agent.get_move(self.board, self.moves) # ask player for move
            x, y = coord
            if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
                self.display.print_message('Coordinate out of bound!')
            elif coord in self.board:
                self.display.print_message('Coordinate already has a piece!')
            else:
                break
        self.display.print_message('Player %s moved %s' % (self.curr_player, coord))

        # add the move
        self.moves.append((x, y, self.curr_player))
        self.board[coord] = self.curr_player

        if self.check_board():
            return

        # transition the player
        self.curr_player = (self.curr_player + 1) % self.num_players

    def check_board(self):
        winning_seq = self.check_piece(*self.moves[-1])
        if winning_seq != None:
            self.has_ended = lambda: True
            self.display.update_board(winning_seq)
            self.display.print_message('Player %s has won!' % (self.curr_player))
            return True
        return False

    # checks to see if the piece is part of winning sequence. Return the winning sequence or None if none exists
    def check_piece(self, x, y, player):
        for x_off, y_off in OFFSETS:
            seq = [(x, y)]
            # can either go the direction of the offset or reverse direction of the offset
            for x_off_dir, y_off_dir in [(x_off, y_off), (-x_off, -y_off)]:
                coord_curr = (x_cur, y_cur) = (x + x_off_dir, y + y_off_dir)
                while coord_curr in self.board and self.board[coord_curr] == player:
                    seq.append(coord_curr)
                    x_cur += x_off_dir
                    y_cur += y_off_dir
                    coord_curr = (x_cur, y_cur)

            if len(seq) >= LENGTH_NEEDED:
                return seq
        return None