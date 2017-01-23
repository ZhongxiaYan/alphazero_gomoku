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
    # maps tuples of (length, player bitmap, opponent bitmap) to a hash of streaks contained
    sequences = {}

    num_bits = { i : sum((i >> j) & 1 for j in range(5)) for i in range(2 ** 5) }

    @staticmethod
    def evaluate_signature(signature):
        mask = 0b11111
        length, pattern = signature
        if length < 5:
            return (0, 0)
        max_num_bits = 1
        max_count = 0
        for i in range(length - 4):
            num_bits = Streak.num_bits[pattern & mask]
            if num_bits > max_num_bits:
                max_num_bits = num_bits
                max_count = 1
            elif num_bits == max_num_bits:
                max_count += 1
            pattern >>= 1
        return (max_num_bits, max_count)

    @staticmethod
    def evaluate_sequence(length, player_bits, opponent_bits):
        '''
        returns hash of the (signature values, player) counts within this sequence
        '''
        mask = 0b11111
        A, B = player_bits, opponent_bits
        signatures = {}

        curr_player = 0
        while A | B != 0:
            while (A | B) & mask == 0:
                A >>= 1
                B >>= 1
                length -= 1
            if B & 1:
                curr_player = 1 - curr_player
                A, B = B, A
            # at this point we know that A & 1 = 1 and B & 1 = 0
            signature_length = 0
            A_temp = A << 4
            B_temp = B
            while B_temp & 1 == 0 and A_temp & mask != 0 and signature_length < length:
                signature_length += 1
                A_temp >>= 1
                B_temp >>= 1
            signature = (signature_length, A & ((1 << signature_length) - 1))
            sig_player = (signature, curr_player)
            signatures[sig_player] = signatures.setdefault(sig_player, 0) + 1
            A >>= signature_length
            B >>= signature_length
            length -= signature_length
        return { (Streak.evaluate_signature(sig), player) : count for (sig, player), count in signatures.items() }

    @staticmethod
    def fill_sequences():
        """
        >>> Streak.fill_sequences()
        >>> streaks = Streak.sequences[(11, 0b0010110010, 0b0001000001)] # [_, _, 0, 1, 0, 0, _, _, 0, _, 1]
        >>> expected = { ((0, 0), 0) : 1, ((0, 0), 1) : 2, ((3, 1), 0) : 1}
        >>> len(set(streaks.items()) ^ set(expected.items())) == 0
        True
        """
        if len(Streak.sequences) > 0: # prevent filling again
            return
        seen_centers = set()
        for sequence in itertools.product((0, 1, 2), repeat=8):
            center_length = 0
            player = 0
            opponent = 0
            for x in sequence:
                if center_length == 0 and x == 0:
                    continue
                center_length += 1
                player <<= 1
                opponent <<= 1
                if x == 1:
                    player |= 1
                else:
                    opponent |= 1
            if (player | opponent) == 0:
                continue
            while (player | opponent) & 1 == 0:
                player >>= 1
                opponent >>= 1
                center_length -= 1
            center = (player, opponent)
            if center in seen_centers:
                continue
            seen_centers.add(center)
            # we only care about gaps on the left and right from 0 to 4
            for left_space in range(LENGTH_NEEDED):
                for right_space in range(LENGTH_NEEDED):
                    length = left_space + right_space + center_length
                    p = player << right_space
                    o = opponent << right_space
                    Streak.sequences[(length, p, o)] = Streak.evaluate_sequence(length, p, o)

    @staticmethod
    def print_signature(signature):
        length, pattern = signature
        print(format(pattern, '0%sb' % length))

class RLAgent(Agent):
    def __init__(self, player_num, display):
        super().__init__(player_num, display)
        # an array of the heads of the 15 rows and their lengths
        self.horizontal_board = [(BOARD_WIDTH, 0, 0) for _ in range(BOARD_HEIGHT)]
        # an array of the heads of the 15 columns
        self.vertical_board = [(BOARD_HEIGHT, 0, 0) for _ in range(BOARD_WIDTH)]
        # an array of the head of the 29 diagonals
        self.downdiag_board = [(BOARD_HEIGHT - abs(BOARD_HEIGHT - i), 0, 0) for i in range(1, 2 * BOARD_HEIGHT)]
        self.updiag_board = [(BOARD_HEIGHT - abs(BOARD_HEIGHT - i), 0, 0) for i in range(1, 2 * BOARD_HEIGHT)]
        Streak.fill_sequences()

    def get_move(self, board, prev_moves):
        pass

    def lookup_move_directional(self, direction, player, row_num, index):
        pass

    def apply_move(self, move):
        # look up move in data structure
        # get updated position
        # update datastructure
        pass

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
        pass
        # apply prev move to update feature values, move list, move values if needed
        # pick max from move list
        # apply move to update feature values, move list, move values
        # use TD update
        if self.self_play: # swap player
            self.player_num = 1 - self.player_num

class OpponentWrapperRLAgent(RLAgent):
    def __init__(self, player_num, display, opponent_agent):
        super().__init__(player_num, display)
        self.opponent_agent = opponent_agent

    def get_move(self, board, prev_moves):
        pass
        # apply prev move to update feature values if needed
        # move = self.opponent_agent.get_move(board, prev_moves)
        # apply move to update feature values
        # use TD update

