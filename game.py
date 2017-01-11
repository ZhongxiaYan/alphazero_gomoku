from config import *
from agent import Agent, CommandLineInputAgent, GuiInputAgent
from aiAgent import ReflexAgent
from display import CommandLineDisplay, PyQtDisplay

import numpy as np

class Game:
    def __init__(self, player_types, display_type):
        self.board = {}
        self.player_types = player_types
        self.curr_player = 0
        self.moves = []
        self.has_ended = lambda: False

        self.display = display_type(self)
        self.agents = [player_type(player_num, self.display) for player_num, player_type in enumerate(player_types)]
        if self.check_draw():
            self.terminate(None)

    def show_board(self):
        self.display.update_board()

    def terminate(self, player, winning_seq=None):
        if player is None:
            self.display.print_message('Draw! Win not possible for any player')
        else:
            self.display.update_board(winning_seq)
            self.display.print_message('Player %s has won!' % (player))
        self.has_ended = lambda: True

    def get_input(self):
        self.display.print_message('Player %s to move!' % (self.curr_player))
        agent = self.agents[self.curr_player] # get the next player
        input = agent.get_move(self.board, self.moves) # ask player for move
        return input

    def transition(self, input):
        self.display.print_message('Player %s moved %s' % (self.curr_player, input))
        if input[0] == 'u': # undo
            undo_num = input[1]
            for move in self.moves[-undo_num:]:
                del self.board[move[0]]
                self.moves.pop()
            # transition the player back
            self.curr_player = len(self.moves) % NUM_PLAYERS
            return

        # coordinate input
        coord = input
        assert(coord not in self.board)
        # add the move
        self.moves.append((coord, self.curr_player))
        self.board[coord] = self.curr_player

        # check if player won
        winning_seq = self.check_piece(coord, self.curr_player)
        if winning_seq is not None:
            self.terminate(self.curr_player, winning_seq)
            return

        # transition the player
        self.curr_player = (self.curr_player + 1) % NUM_PLAYERS
        if self.check_draw():
            self.terminate(None)

    def check_draw(self):
        # check if any player can win
        check_player = self.curr_player
        num_checked = 0
        while not self.check_win_possible(check_player):
            num_checked += 1
            check_player = (check_player + 1) % NUM_PLAYERS
            if num_checked == NUM_PLAYERS:
                return True
        return False

    # checks to see if the piece is part of winning sequence. Return the winning sequence or [] if none exists
    def check_piece(self, coord, player):
        """
        >>> g = Game([HUMAN_CMD_LINE, HUMAN_CMD_LINE], DISPLAY_COMMAND_LINE)
        >>> player = 0
        >>> g.board[(0, 0)] = g.board[(0, 1)] = g.board[(0, 2)] = g.board[(0, 3)] = player
        >>> g.check_piece((0, 2), player)
        >>> g.check_piece((0, 1), player)
        >>> g.board[(0, 4)] = g.board[(1, 3)] = player
        >>> sorted(g.check_piece((0, 2), player))
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        >>> sorted(g.check_piece((0, 4), player))
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        >>> sorted(g.check_piece((0, 0), player))
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        >>> g.check_piece((1, 3), player)
        >>> g.board[(2, 2)] = g.board[(3, 1)] = g.board[(4, 0)] = g.board[(2, 4)] = player
        >>> sorted(g.check_piece((1, 3), player))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        >>> sorted(g.check_piece((4, 0), player))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        >>> g.check_piece((2, 4), player)
        """
        assert(self.board[coord] == player)
        for y_off, x_off in OFFSETS:
            seq = [coord]
            # can either go the direction of the offset or reverse direction of the offset
            for off_dir in [(y_off, x_off), (-y_off, -x_off)]:
                coord_curr = tuple(np.add(coord, off_dir))
                while coord_curr in self.board and self.board[coord_curr] == player:
                    seq.append(coord_curr)
                    coord_curr = tuple(np.add(coord_curr, off_dir))
            if len(seq) >= LENGTH_NEEDED:
                return seq
        return None

    # returns true if the player can still win
    def check_win_possible(self, player):
        pieces_added = []

        def remove_pieces():
            for piece in pieces_added:
                del self.board[piece]

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                coord = (y, x)
                if coord not in self.board:
                    pieces_added.append(coord)
                    self.board[coord] = player
                    # if the piece just added creates a winning sequence
                    if self.check_piece(coord, player) is not None:
                        remove_pieces()
                        return True
        # no winning sequence found after filling board
        remove_pieces()
        return False
