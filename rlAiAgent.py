from agent import Agent
from config import *

from collections import defaultdict
import itertools
import functools
import random
import math
import numpy as np
import time

class RLAgent(Agent):
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
    def __init__(self, player_num, display):
        super().__init__(player_num, display)
        # initialize arrays of binary rows representing boards from each of the four orientation
        self.boards =
        self.opponent_boards =
        # map binary patterns to (number, repeat) paradigms
        self.patterns =
        # a stack of feature counts of all previous board positions
        self.feature_stack =
        # cached dictionary of position evaluations
        self.position_evaluations =

    def get_move(self, board, prev_moves):
