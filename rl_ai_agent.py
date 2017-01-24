from agent import Agent
from config import *

from collections import defaultdict
import itertools
import functools
import pickle
import copy
import random
import os
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
    first_bit = { i : min(j if ((i >> j) & 1 == 1) else 5 for j in range(5)) for i in range(2 ** 5) }

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
        if max_num_bits >= LENGTH_NEEDED:
            return (LENGTH_NEEDED, 1)
        return (max_num_bits, max_count)

    @staticmethod
    def evaluate_sequence(length, player_bits, opponent_bits):
        """
        returns hash of the (signature values, player) counts within this sequence
        >>> def test(seq, expected):
        ...     return len(set(Streak.evaluate_sequence(*seq).items()) ^ set(expected.items())) == 0
        >>> seq = (10, 0b0010110010, 0b0001000001) # [_, _, 0, 1, 0, 0, _, _, 0, 1]
        >>> expected = { ((0, 0), 0) : 1, ((0, 0), 1) : 2, ((3, 1), 0) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (13, 0b0000101000000, 0b0000010111000) # [_, _, _, _, 0, 1, 0, 1, 1, 1, _, _, _]
        >>> expected = { ((0, 0), 0) : 1, ((0, 0), 1) : 1, ((1, 1), 0) : 1, ((3, 1), 1) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (12, 0b000000011000, 0b011110000000) # [_, 1, 1, 1, 1, _, _, 0, 0, _, _, _]
        >>> expected = { ((4, 2), 1) : 1, ((2, 3), 0) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (11, 0b00000010010, 0b00001000100) # [_, _, _, _, 1, _, 0, _, 1, 0, _]
        >>> expected = { ((1, 2), 1) : 1, ((0, 0), 0) : 2, ((0, 0), 1) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (11, 0b00000111100, 0b01001000000) # [_, 1, _, _, 1, 0, 0, 0, 0, _, _]
        >>> expected = { ((2, 1), 1) : 1, ((4, 1), 0) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (11, 0b00000001100, 0b01000000000) # [_, 1, _, _, _, _, _, 0, 0, _, _]
        >>> expected = { ((1, 2), 1) : 1, ((2, 3), 0) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (10, 0b0000110000, 0b0000000000) # [_, _, _, _, 0, 0, _, _, _, _]
        >>> expected = { ((2, 4), 0) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (11, 0b0, 0b01101101100) # [_, 1, 1, _, 1, 1, _, 1, 1, _, _]
        >>> expected = { ((4, 2), 1) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (10, 0b0110101100, 0b0) # [_, 0, 0, _, 0, _, 0, 0, _, _]
        >>> expected = { ((3, 5), 0) : 1 }
        >>> test(seq, expected)
        True
        >>> seq = (14, 0b00110101100000, 0b00000000010000) # [_, _, 0, 0, _, 0, _, 0, 0, 1, _, _, _, _]
        >>> expected = { ((3, 4), 0) : 1, ((1, 1), 1) : 1 }
        >>> test(seq, expected)
        True
        """
        mask = 0b11111
        A, B = player_bits, opponent_bits
        signatures = {}

        curr_player = 0
        while A | B != 0:
            while (A | B) & mask == 0:
                A >>= 1
                B >>= 1
                length -= 1
            if Streak.first_bit[B & mask] < Streak.first_bit[A & mask]:
                curr_player = 1 - curr_player
                A, B = B, A
            # at this point we know that A has the first set LSB
            A_temp = A
            signature_length = 4
            while A_temp & 0b1111 != 0:
                A_temp <<= 1
                signature_length -= 1
            B_temp = B >> signature_length
            while B_temp & 1 == 0 and A_temp & mask != 0 and signature_length < length:
                signature_length += 1
                A_temp >>= 1
                B_temp >>= 1
            A_length = signature_length
            while A & (1 << A_length - 1) == 0:
                A_length -= 1
            signature = (signature_length, A & ((1 << signature_length) - 1))
            sig_player = (signature, curr_player)
            signatures[sig_player] = signatures.setdefault(sig_player, 0) + 1
            A >>= A_length
            B >>= A_length
            length -= A_length
        sig_values = {}
        for (sig, player), count in signatures.items():
            sig_value_player = (Streak.evaluate_signature(sig), player)
            sig_values[sig_value_player] = sig_values.setdefault(sig_value_player, 0) + count
        return sig_values

    @staticmethod
    def fill_sequences():
        """
        >>> Streak.fill_sequences()
        >>> valid_seqs = [(10, 0b0010110010, 0b0001000001), (13, 0b0000101000000, 0b0000010111000), (12, 0b000000011000, 0b011110000000), (11, 0b00000010010, 0b00001000100), (11, 0b00000111100, 0b01001000000), (11, 0b00000001100, 0b01000000000), (10, 0b0000110000, 0b0000000000), (11, 0b0, 0b01101101100), (10, 0b0110101100, 0b0), (14, 0b00110101100000, 0b00000000010000)]
        >>> np.all([seq in Streak.sequences for seq in valid_seqs])
        True
        """
        if len(Streak.sequences) > 0: # prevent filling again
            return
        cache = 'rl_cache.p'
        if os.path.exists(cache):
            Streak.sequences = pickle.load(open(cache, 'rb'))
            return
        seen_centers = set()
        count = 0
        for sequence in itertools.product((0, 1, 2), repeat=15):
            count += 1
            if count % 10000 == 0:
                print(count)
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
                elif x == 2:
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
                    if length > BOARD_WIDTH:
                        continue
                    p = player << right_space
                    o = opponent << right_space
                    Streak.sequences[(length, p, o)] = Streak.evaluate_sequence(length, p, o)
        pickle.dump(Streak.sequences, open(cache, 'wb'))

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
        length, player_bits, opponent_bits = direction[row_num]
        old_streak_values = direction # TODO
        # create new row
        if player == 0:
            player_bits |= 1 << length - index - 1
        else:
            opponent_bits |= 1 << length - index - 1
        direction[row_num] = (length, player_bits, opponent_bits)
        # evaluate new row, first strip both ends of zeros if 5 or more zeros
        bits = player_bits | opponent_bits
        right_space = 0
        while bits & (0b11111 << right_space) == 0:
            right_space += 1
        left_space = 0
        if length > LENGTH_NEEDED:
            mask = 0b11111 << length - LENGTH_NEEDED
            while bits & (0b11111 << (length - LENGTH_NEEDED) - left_space) == 0:
                left_space += 1
        length -= right_space + left_space
        player_bits >>= right_space
        opponent_bits >>= right_space
        streaks = Streak.sequences.getdefault((length, player_bits, opponent_bits), None)
        if streaks is None:




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

