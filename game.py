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
        if winning_seq != []:
            self.has_ended = lambda: True
            self.display.update_board(winning_seq)
            self.display.print_message('Player %s has won!' % (self.curr_player))
            return True
        return False

    # checks to see if the piece is part of winning sequence. Return the winning sequence or [] if none exists
    def check_piece(self, x, y, player):
        """
        >>> g = Game([HUMAN, HUMAN], DISPLAY_COMMAND_LINE)
        >>> player = 0
        >>> g.board[(0, 0)] = g.board[(0, 1)] = g.board[(0, 2)] = g.board[(0, 3)] = player
        >>> g.check_piece(0, 2, player)
        []
        >>> g.check_piece(0, 1, player)
        []
        >>> g.board[(0, 4)] = g.board[(1, 3)] = player
        >>> sorted(g.check_piece(0, 2, player))
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        >>> sorted(g.check_piece(0, 4, player))
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        >>> sorted(g.check_piece(0, 0, player))
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        >>> g.check_piece(1, 3, player)
        []
        >>> g.board[(2, 2)] = g.board[(3, 1)] = g.board[(4, 0)] = g.board[(2, 4)] = player
        >>> sorted(g.check_piece(1, 3, player))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        >>> sorted(g.check_piece(4, 0, player))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        >>> g.check_piece(2, 4, player)
        []
        """
        assert(self.board[(x, y)] == player)
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
        return []