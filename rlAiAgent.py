from agent import Agent
from config import *

from collections import defaultdict
import itertools
import functools
import copy
import random
import math
import numpy as np
import time

'''
Definitions:
    Streak: a row/column/diagonal of one player's pieces that are not separated by > 3 pieces at a time or an opponent's piece and not within another streak
    Components of a streak
        Type is the maximum number of pieces in any consecutive frame of 5 pieces
        Repeat is how many frames of 5 pieces contains <type> number of the player's own pieces and none of the opponent's pieces
        Internal gaps are any empty places between the edgemost pieces
        External gaps are any empty places outside of edgemost pieces that is within the bounds of at least one sequence of 5 spots with a piece in the streak

Any move either (along a single direction) falls into no streak, interal gap of one streak, or external gaps of one / two streaks.
Falling into no streak:
    Create a single streak of one piece
Falling into internal gap of own streak:
    Update that streak and no others
Falling into internal gap of opponent streak:
    Split streak into three streaks (opponent, own, opponent)
Falling into external gap of one streak:
    Merge into streak if own streak, update streak and create new streak if opponent streak
Falling into external gap of two streaks:
    If both own: Merge both into one
    else: Update both streaks

Streak transitioning:
    Map (streak, move) to (next streak, offset from current streak, next moves)

For each direction, have a list of (start, streak, length) ordered by start index

Database / playback:
    For every move given: make move and update streaks.
Self play:
    For every move made: update streak, update potential move locations, get evaluation score for each move locations, put into sorted list
'''

class Streak:
    '''
    Represents a streak and serves as a linked list node (can get next streak in the row)
    '''
    # maps streak signature, move offset, player to new streak signatures, new streak offsets, and new streak players
    # in streak signature: 1 means current player, 0 means empty space
    # in player denotation: True means the new move is made by current player, False means the new move is made by opponent
    transitions = {}

    def __init__(self, start, end, player, signature, prev_streak=None, next_streak=None):
        self.start = start
        self.end = end
        self.player = player
        self.signature = signature
        self.prev_streak = prev_streak
        self.next_streak = next_streak

    @classmethod
    def prepend_single(streak, position, player):
        '''
        length: the length of the current row
        '''
        return

    @classmethod
    def get_sentinels(first_index, second_index):
        first = Streak(first_index, first_index, None, None)
        second = Streak(first_index, first_index, None, None, first)
        first.next_streak = second
        return first

    @classmethod
    def fill_transitions():
        if len(Streak.transitions) > 0: # prevent filling again
            return



class RLAgent(Agent):
    def __init__(self, player_num, display):
        super().__init__(player_num, display)
        # an array of the heads of the 15 rows and their lengths
        self.horizontal_board = [Streak.get_sentinels(-1, BOARD_WIDTH) for _ in range(BOARD_HEIGHT)]
        # an array of the heads of the 15 columns
        self.vertical_board = [Streak.get_sentinel(-1, BOARD_HEIGHT) for _ in range(BOARD_WIDTH)]
        # an array of the head of the 29 diagonals
        self.downdiag_board = [Streak.get_sentinel(-1, BOARD_HEIGHT - abs(BOARD_HEIGHT - i)) for i in range(1, 2 * BOARD_HEIGHT)]
        self.updiag_board = [Streak.get_sentinel(-1, BOARD_HEIGHT - abs(BOARD_HEIGHT - i)) for i in range(1, 2 * BOARD_HEIGHT)]
        Streak.fill_transitions()

    def get_move(self, board, prev_moves):

    def lookup_move_directional(self, direction, player, row_num, index):
        curr, row_length = direction[row_num]
        while curr.end <= index:
            curr = curr.next_streak
        # curr.end > index
        next = curr.next_streak
        if curr.start > index: # not in any streak
            Streak.prepend_single(curr, index, player)
        elif next.start < index: # in two streaks
            if curr.player == next.player:
                if curr.player == player: # merge the two streaks
                    Streak.merge_next(curr, index)
                else: # update the two streaks and add another streak
                    Streak.update(curr, index, player)
                    Streak.update(next, index, player)
                    Streak.prepend_single(next, index, curr_player)
            else: # the two streaks are different players', update both
                Streak.update(curr, index, player)
                Streak.update(next, index, player)
        else: # in one streak
            Streak.update(curr, index, player)

    def apply_move(self, move):
        look up move in data structure
        get updated position
        update datastructure



class SelfPlayRLAgent(RLAgent):
    '''
    Plays as BOTH player and opponent
    '''
    def __init__(self, player_num, display, self_play=False):
        '''
        if self_play False, then play as one side
        '''
        super().__init__(player_num, display)
        self.self_play = self_play

    def get_move(self, board, prev_moves):
        apply prev move to update feature values, move list, move values if needed
        pick max from move list
        apply move to update feature values, move list, move values
        use TD update
        if self_play: # swap player
            self.player_num = 1 - self.player_num

class OpponentWrapperRLAgent(RLAgent):
    def __init__(self, player_num, display, opponent_agent):
        super().__init__(player_num, display)
        self.opponent_agent = opponent_agent

    def get_move(self, board, prev_moves):
        apply prev move to update feature values if needed
        move = self.opponent_agent.get_move(board, prev_moves)
        apply move to update feature values
        use TD update
        return move
