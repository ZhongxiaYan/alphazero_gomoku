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
    # maps streak signature to neural net feature type
    # streak signature: (length, value). Length is the total length of the streak, value has 1 as current player and 0 as empty space
    signatures = {}
    # maps streak signature, move offset, player to new streak signatures, new streak offsets, and new streak players
    # in player denotation: True means the new move is made by current player, False means the new move is made by opponent
    transitions = {}
    num_bits = { i : sum((i >> j) & 1 for j in range(5)) for i in range(2 ** 5) }

    def __init__(self, start, end, player, center, left_space=0, right_space=0, prev_streak=None, next_streak=None):
        self.start = start
        self.end = end
        self.player = player
        self.center = center
        self.left_space = left_space
        self.right_space = right_space
        self.prev_streak = prev_streak
        self.next_streak = next_streak

    @staticmethod
    def get_sentinels(first_index, second_index):
        first = Streak(first_index, first_index, None, None)
        second = Streak(second_index, second_index, None, None, 0, 0, first)
        first.next_streak = second
        return first

    @staticmethod
    def evaluate(signature):
        length, pattern = signature
        if length < 5:
            return (0, 0)
        mask = 0b11111
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
    def fill_signatures_and_transitions():
        if len(Streak.transitions) > 0: # prevent filling again
            return
        # build the set of empty centers
        nonzero_centers = set()
        cached_center_length = LENGTH_NEEDED + 1
        for center in range(1, 2 ** cached_center_length):
            while center & 1 == 0:
                center >>= 1
            if center in nonzero_centers:
                continue
            center_length = 1 + max((i * ((center >> i) & 1) for i in range(cached_center_length)))
            nonzero_centers.add((center, center_length))
        # create streaks
        for center, center_length in nonzero_centers:
            for right_space in range(LENGTH_NEEDED):
                for left_space in range(LENGTH_NEEDED): # we only care about gaps on the left from 0 to 4
                    signature = (left_space + right_space + center_length, center << right_space)
                    Streak.signatures[signature] = Streak.evaluate(signature)
        # calculate transitions of centers
        # for center, center_length in nonzero_centers:
        #     for index in range():
        #         if pattern & (1 << index) != 0:
        #             continue
        #         offset = length - index - 1
        #         # if same player makes a move
        #         new_pattern = pattern | (1 << index)
        #         new_signature = (left_space, right_space, center_length, new_pattern)
        #         self.transitions[(signature, offset, True)] = [(new_signature, 0, True)]
        #
        #         # if opponent makes a move
        #         first_pattern = pattern >> (right_space + index + 1)
        #         first_right_space = 0
        #         temp = first_pattern
        #         while temp & 1 == 0:
        #             first_right_space += 1
        #             temp >>= 1
        #         first_center_length = length - index - 1 - left_space - first_right_space
        #
        #         second_pattern = pattern & ((1 << (right_space + index)) - 1)
        #         second_left_space = 0
        #         i = right_space + index - 1
        #         while second_pattern & (1 << i) == 0:
        #             second_left_space += 1
        #             i -= 1
        #         second_center_length = index - second_left_space - right_space
        #
        #         first_signature = (left_space, first_right_space, first_center_length, first_pattern)
        #         second_signature = (second_left_space, right_space, second_center_length, second_pattern)
        #         opponent_signature = (first_right_space, 1, second_left_space, 1 << second_left_space)
        #         self.transitions[(signature, offset, False)] = [
        #             (first_signature, 0, True),
        #             (opponent_signature, left_space + first_center_length, False),
        #             (second_signature, offset + 1, True)
        #         ]


    @staticmethod
    def print_signature(signature):
        length, pattern = signature
        print(format(pattern, '0%sb' % length))

    @staticmethod
    def prepend_single(curr, index, player):
        right_space = min(curr.start + curr.left_space - index - 1, LENGTH_NEEDED - 1)
        prev = curr.prev_streak
        left_space = min(index - (prev.end - prev.right_space), LENGTH_NEEDED - 1)
        signature = (left_space + right_space + 1, 1 << right_space)
        curr.prev_streak = prev.next_streak = Streak(index - left_space, index + right_space + 1, player, 1, left_space, right_space, prev, curr)

    @staticmethod
    def update_one(curr, index, player):
        left_offset = index - curr.start
        length = curr.end - curr.start
        right_offset = length - left_offset - 1
        if curr.player == player: # same player's move
            if right_offset < curr.right_space:
                center = (curr.center << (curr.right_space - right_offset)) | 1
                next = curr.next_streak
                right_space = min((next.start + next.left_space) - index - 1, LENGTH_NEEDED - 1)
                curr.right_space = right_space
                curr.end = index + right_space + 1
                curr.center = center
            elif right_offset < length - curr.left_space:
                curr.center |= 1 << (right_offset - curr.right_space)
            else:
                curr.center |= 1 << (right_offset - curr.right_space)
                prev = curr.prev_streak
                left_space = min(index - (prev.end - prev.left_space), LENGTH_NEEDED - 1)
                curr.start = index - left_space
                curr.left_space = left_space
        else:
            if right_offset < curr.right_space:
                right_space = curr.right_space - right_offset - 1
                curr.end = index - 1
                curr.right_space = right_space
                next = curr.next
                new_left_space = right_space
                new_right_space = min((next.start + next.left_space) - index - 1, LENGTH_NEEDED - 1)
                curr.next_streak = next.prev_streak = Streak(index - new_left_space, index + new_right_space + 1, player, 1, new_left_space, new_right_space, curr, next)
            elif right_offset < length - curr.left_space:
                center_offset = right_offset - curr.right_space
                center = curr.center
                new_left_space = 0
                temp = center_offset + 1
                while center & temp == 0:
                    new_left_space += 1
                    temp += 1
                new_left_center = center >> temp
                temp = center_offset - 1
                new_right_space = 0
                while center & temp == 0:
                    new_right_space += 1
                    temp -= 1
                new_right_center = center & (1 << (temp + 1) - 1)
                middle_streak = Streak(index - new_left_space, index + new_right_space + 1, player, 1, new_left_space, new_right_space, curr, None)
                end_streak = Streak(index + 1, curr.end, curr.player, new_right_center, new_right_space, curr.right_space, middle_streak, curr.next_streak)
                middle_streak.next_streak = end_streak
                curr.next_streak = middle_streak
                curr.right_space = new_left_space
                curr.center = new_left_center
            else:
                pass

class RLAgent(Agent):
    def __init__(self, player_num, display):
        super().__init__(player_num, display)
        # an array of the heads of the 15 rows and their lengths
        self.horizontal_board = [Streak.get_sentinels(-1, BOARD_WIDTH) for _ in range(BOARD_HEIGHT)]
        # an array of the heads of the 15 columns
        self.vertical_board = [Streak.get_sentinels(-1, BOARD_HEIGHT) for _ in range(BOARD_WIDTH)]
        # an array of the head of the 29 diagonals
        self.downdiag_board = [Streak.get_sentinels(-1, BOARD_HEIGHT - abs(BOARD_HEIGHT - i)) for i in range(1, 2 * BOARD_HEIGHT)]
        self.updiag_board = [Streak.get_sentinels(-1, BOARD_HEIGHT - abs(BOARD_HEIGHT - i)) for i in range(1, 2 * BOARD_HEIGHT)]
        Streak.fill_signatures_and_transitions()

    def get_move(self, board, prev_moves):
        pass

    def lookup_move_directional(self, direction, player, row_num, index):
        curr = direction[row_num]
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
                    Streak.prepend_single(next, index, player)
            else: # the two streaks are different players', update both
                Streak.update(curr, index, player)
                Streak.update(next, index, player)
        else: # in one streak
            Streak.update_one(curr, index, player)

    def apply_move(self, move):
        # look up move in data structure
        # get updated position
        # update datastructure
        pass

agent = RLAgent(0, None)
agent.lookup_move_directional(agent.horizontal_board, 0, 7, 7)

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
        if self_play: # swap player
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
        return move
