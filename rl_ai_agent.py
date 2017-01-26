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
    signatures = {}
    # maps tuples of (length, player bitmap, opponent bitmap) to a hash of streaks contained
    sequences = {}

    num_bits = { i : sum((i >> j) & 1 for j in range(5)) for i in range(2 ** 5) }
    first_bit = { i : min(j if ((i >> j) & 1 == 1) else 5 for j in range(5)) for i in range(2 ** 5) }

    # valid streak values: (0, 0), (1, 1 to 5), (2, 1 to 5), (3, 1 to 5), (4, 1 to 2), (5, 1)
    index_to_streak = [(0, 0)] + [(i, j) for i in [1, 2, 3] for j in range(1, LENGTH_NEEDED + 1)] + [(4, 1), (4, 2), (5, 1)]
    streak_to_index = { streak : i for i, streak in enumerate(index_to_streak) }

    @staticmethod
    def evaluate_signature(signature):
        mask = 0b11111
        length, pattern = signature
        if length < LENGTH_NEEDED:
            return ((0, 0), 0)
        max_num_bits = 1
        max_count = 0
        interesting_locations = 0
        shift = 0
        for i in range(length - 4):
            masked_pattern = pattern & mask
            num_bits = Streak.num_bits[masked_pattern]
            if num_bits > max_num_bits:
                interesting_locations = (mask - masked_pattern) << shift
                max_num_bits = num_bits
                max_count = 1
            elif num_bits == max_num_bits:
                interesting_locations |= (mask - masked_pattern) << shift
                max_count += 1
            pattern >>= 1
            shift += 1
        # enforce the valid streak values
        if max_num_bits >= LENGTH_NEEDED:
            return ((LENGTH_NEEDED, 1), interesting_locations)
        elif max_num_bits in [1, 2, 3]:
            return ((max_num_bits, min(max_count, LENGTH_NEEDED)), interesting_locations)
        # max_num_bits = 4
        return ((max_num_bits, min(max_count, 2)), interesting_locations)

    @staticmethod
    def evaluate_sequence(length, player_bits, opponent_bits):
        """
        returns hash of the (signature values, player) counts within this sequence
        >>> def test(seq, expected, expected_locations):
        ...     seq_values, interesting_locations = Streak.evaluate_sequence(*seq)
        ...     expected_values = np.zeros((2, len(Streak.index_to_streak)), dtype=np.int8)
        ...     for (sig_value, player), count in expected.items():
        ...         expected_values[player][Streak.streak_to_index[sig_value]] += count
        ...     return np.all(seq_values == expected_values) and expected_locations == interesting_locations
        >>> test((10, 0b0010110010, 0b0001000001), { ((0, 0), 0) : 1, ((0, 0), 1) : 2, ((3, 1), 0) : 1 }, 0b0000001100) # [_, _, 0, 1, 0, 0, _, _, 0, 1]
        True
        >>> test((13, 0b0000101000000, 0b0000010111000), { ((0, 0), 0) : 1, ((0, 0), 1) : 1, ((1, 1), 0) : 1, ((3, 1), 1) : 1 }, 0b1111000000110) # [_, _, _, _, 0, 1, 0, 1, 1, 1, _, _, _]
        True
        >>> test((12, 0b000000011000, 0b011110000000), { ((4, 2), 1) : 1, ((2, 3), 0) : 1 }, 0b100001100111) # [_, 1, 1, 1, 1, _, _, 0, 0, _, _, _]
        True
        >>> test((11, 0b00000010010, 0b00001000100), { ((1, 2), 1) : 1, ((0, 0), 0) : 2, ((0, 0), 1) : 1 }, 0b11110100000) # [_, _, _, _, 1, _, 0, _, 1, 0, _]
        True
        >>> test((11, 0b00000111100, 0b01001000000), { ((2, 1), 1) : 1, ((4, 1), 0) : 1 }, 0b10110000010) # [_, 1, _, _, 1, 0, 0, 0, 0, _, _]
        True
        >>> test((11, 0b00000001100, 0b01000000000), { ((1, 2), 1) : 1, ((2, 3), 0) : 1 }, 0b10111110011) # [_, 1, _, _, _, _, _, 0, 0, _, _]
        True
        >>> test((10, 0b0000110000, 0b0000000000), { ((2, 4), 0) : 1 }, 0b0111001110) # [_, _, _, _, 0, 0, _, _, _, _]
        True
        >>> test((11, 0b0, 0b01101101100), { ((4, 2), 1) : 1 }, 0b00010010000) # [_, 1, 1, _, 1, 1, _, 1, 1, _, _]
        True
        >>> test((10, 0b0110101100, 0b0), { ((3, 5), 0) : 1 }, 0b1001010010) # [_, 0, 0, _, 0, _, 0, 0, _, _]
        True
        >>> test((14, 0b00110101100000, 0b00000000010000), { ((3, 4), 0) : 1, ((1, 1), 1) : 1 }, 0b01001010001111) # [_, _, 0, 0, _, 0, _, 0, 0, 1, _, _, _, _]
        True
        >>> test((15, 0b010100000000000, 0b000000111001101), { ((2, 2), 0) : 1, ((3, 5), 1) : 1 }, 0b101011000110010) # [_, 0, _, 0, _, _, 1, 1, 1, _, _, 1, 1, _, 1]
        True
        >>> test((15, 0b011000010000111, 0b0), { ((2, 2), 0) : 1, ((1, 5), 0) : 1, ((3, 1), 0) : 1 }, 0b100111101111000) # [_, 0, 0, _, _, _, _, 0, _, _, _, _, 0, 0, 0]
        True
        """
        mask = 0b11111
        A, B = player_bits, opponent_bits

        sequence_values = np.zeros((2, len(Streak.index_to_streak)), dtype=np.int8)
        signatures = Streak.signatures
        streak_to_index = Streak.streak_to_index
        evaluate_signature = Streak.evaluate_signature
        interesting_locations = 0
        curr_player = 0
        total_shift = 0
        while A | B != 0:
            # remove trailing 0's
            while (A | B) & mask == 0:
                A >>= 1
                B >>= 1
                length -= 1
                total_shift += 1
            if Streak.first_bit[B & mask] < Streak.first_bit[A & mask]:
                curr_player = 1 - curr_player
                A, B = B, A
            # at this point we know that A has the first set LSB
            A_temp = A
            signature_length = 4
            while A_temp & 0b1111 != 0: # pad A_temp with 0's until it has 4 0's on the right
                A_temp <<= 1
                signature_length -= 1
            B_temp = B >> signature_length
            # scan until B is set or A has a gap of 4 or more
            while B_temp & 1 == 0 and signature_length < length:
                signature_length += 1
                A_temp >>= 1
                B_temp >>= 1
                if A_temp & 0b1111 == 0:
                    break
            shift_length = signature_length
            while A & (1 << shift_length - 1) == 0: # adjust shift length to remove leading spaces in signature
                shift_length -= 1
            signature = (signature_length, A & ((1 << signature_length) - 1))
            sig_value = signatures.setdefault(signature, None)
            if sig_value is None:
                sig_value = evaluate_signature(signature)
                signatures[signature] = sig_value
            streak_value, location_bits = sig_value
            sequence_values[curr_player][streak_to_index[streak_value]] += 1

            interesting_locations |= location_bits << total_shift
            A >>= shift_length
            B >>= shift_length
            length -= shift_length
            total_shift += shift_length
        return sequence_values, interesting_locations

    @staticmethod
    def fill_sequences():
        """
        >>> Streak.fill_sequences()
        >>> valid_seqs = [(10, 0b0010110010, 0b0001000001), (13, 0b0000101000000, 0b0000010111000), (12, 0b000000011000, 0b011110000000), (11, 0b00000010010, 0b00001000100), (11, 0b00000111100, 0b01001000000), (11, 0b00000001100, 0b01000000000), (10, 0b0000110000, 0b0000000000), (11, 0b0, 0b01101101100), (10, 0b0110101100, 0b0), (14, 0b00110101100000, 0b00000000010000)]
        >>> np.all([seq in Streak.sequences for seq in valid_seqs])
        True
        """
        sequences = Streak.sequences
        if len(sequences) > 0: # prevent filling again
            return
        cache_key = 'rl_cache_key.p'
        cache_value = 'rl_cache_value.p'
        if not (os.path.exists(cache_key) and os.path.exists(cache_value)):
            num_rows = sum([(3 ** length) - 1 for length in range(1, BOARD_WIDTH + 1)])
            evaluate_sequence = Streak.evaluate_sequence
            key_matrix = np.zeros((num_rows, 4), dtype=np.int16)
            value_matrix = np.zeros((num_rows, 2, len(Streak.index_to_streak)), dtype=np.int8)
            index_map = {}
            index = 0
            for length in range(1, BOARD_WIDTH + 1):
                for sequence in itertools.product((0, 1, 2), repeat=length):
                    if index % 10000 == 0:
                        print(index)
                    player = 0
                    opponent = 0
                    for x in sequence:
                        player <<= 1
                        opponent <<= 1
                        if x == 1:
                            player |= 1
                        elif x == 2:
                            opponent |= 1
                    bits = player | opponent
                    if bits == 0:
                        continue
                    sequence = (length, player, opponent)
                    key_matrix[index][0:3] = sequence
                    opposite_sequence = (length, opponent, player)
                    if opposite_sequence in index_map:
                        opposite_index = index_map[opposite_sequence]
                        value_matrix[index][:] = np.roll(value_matrix[opposite_index], 1, axis=0)
                        key_matrix[index][3] = key_matrix[opposite_index][3]
                        index += 1
                        continue
                    right_space = 0
                    while bits & (0b11111 << right_space) == 0:
                        right_space += 1
                    left_space = 0
                    if length > LENGTH_NEEDED:
                        mask = 0b11111 << length - LENGTH_NEEDED
                        while bits & (mask >> left_space) == 0:
                            left_space += 1
                    new_length = length - right_space - left_space
                    new_player = player >> right_space
                    new_opponent = opponent >> right_space
                    equiv_sequence = (new_length, new_player, new_opponent)
                    if equiv_sequence in index_map:
                        equiv_index = index_map[equiv_sequence]
                        value_matrix[index][:] = value_matrix[equiv_index]
                        key_matrix[index][3] = key_matrix[equiv_index][3]
                    else:
                        values, interesting_locations = evaluate_sequence(length, player, opponent)
                        value_matrix[index][:] = values
                        key_matrix[index][3] = interesting_locations
                        index_map[sequence] = index
                    index += 1
            pickle.dump(key_matrix, open(cache_key, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(value_matrix, open(cache_value, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            key_matrix = pickle.load(open(cache_key, 'rb'))
            value_matrix = pickle.load(open(cache_value, 'rb'))
        for key, value in zip(key_matrix, value_matrix):
            sequences[tuple(key[:3])] = (key[3], value)

    @staticmethod
    def print_signature(signature):
        length, pattern = signature
        print(format(pattern, '0%sb' % length))

Streak.fill_sequences()

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

