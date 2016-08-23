from agent import Agent
from config import *

from collections import defaultdict
import itertools
import math
import numpy as np
import time

def iterate_all_directions(coord, radius, func):
    for (y_off, x_off) in OFFSETS:
        for y_off_dir, x_off_dir in ((y_off, x_off), (-y_off, -x_off)):
            new_y, new_x = coord
            for r in range(radius):
                new_y += y_off_dir
                new_x += x_off_dir
                if not func((new_y, new_x)):
                    break

def out_of_bound(coord):
    y, x = coord
    return not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT)

class AIInputAgent(Agent):

    # return all the positions that may extend at least one current sequence
    def get_nearby_empty_locations(self, board):
        """
        >>> board = { (2, 7) : 0, (3, 5) : 0, (3, 6) : 1}
        >>> aiAgent = AIInputAgent(None, None)
        >>> sorted([(coord, tuple(players)) for coord, players in aiAgent.get_nearby_empty_locations(board).items()])
        [((0, 5), (True, False)), ((0, 7), (True, False)), ((0, 9), (True, False)), ((1, 3), (True, False)), ((1, 4), (False, True)), ((1, 5), (True, False)), ((1, 6), (True, True)), ((1, 7), (True, False)), ((1, 8), (True, False)), ((2, 4), (True, False)), ((2, 5), (True, True)), ((2, 6), (True, True)), ((2, 8), (True, False)), ((2, 9), (True, False)), ((3, 3), (True, False)), ((3, 4), (True, False)), ((3, 7), (True, True)), ((3, 8), (True, True)), ((4, 4), (True, False)), ((4, 5), (True, True)), ((4, 6), (True, True)), ((4, 7), (True, True)), ((4, 9), (True, False)), ((5, 3), (True, False)), ((5, 4), (False, True)), ((5, 5), (True, False)), ((5, 6), (False, True)), ((5, 7), (True, False)), ((5, 8), (False, True))]
        """
        search_radius = LENGTH_NEEDED // 2
        nearby_empty_locations = {}
        new_coord_player = None

        def set_nearby_empty_location(new_coord):
            if not out_of_bound(new_coord) and new_coord not in board:
                location_nearby_players = nearby_empty_locations.get(new_coord, EMPTY_PIECE)
                if location_nearby_players == EMPTY_PIECE:
                    location_nearby_players = get_empty_location()
                    nearby_empty_locations[new_coord] = location_nearby_players
                location_nearby_players[new_coord_player] = True
                return True
            return False

        for coord, player in board.items():
            new_coord_player = player
            iterate_all_directions(coord, search_radius, set_nearby_empty_location)
        return nearby_empty_locations

    @staticmethod
    def evaluate_direction(board, coord, player, offset):
        offset_seq = AIInputAgent.get_sequence(board, coord, offset)
        num_pieces_per_frame = AIInputAgent.get_num_pieces_per_frame(offset_seq, player)
        return AIInputAgent.score_num_pieces_per_frame(num_pieces_per_frame)

    @staticmethod
    def score_num_pieces_per_frame(num_pieces_per_frame):
        """
        >>> AIInputAgent.score_num_pieces_per_frame([2, 0, 0, 0, 0, 0, 3])
        3.0
        >>> AIInputAgent.score_num_pieces_per_frame([3])
        3.0
        >>> AIInputAgent.score_num_pieces_per_frame([1, 2, 3, 3, 4, 4, 3, 1])
        5.0
        >>> AIInputAgent.score_num_pieces_per_frame([1, 2, 3, 3, 4, 4, 4, 3, 1])
        5.0
        >>> AIInputAgent.score_num_pieces_per_frame([1, 2, 3, 3, 3, 1])
        4.0
        >>> AIInputAgent.score_num_pieces_per_frame([1, 2, 3, 3, 1])
        3.5
        """
        m = max(num_pieces_per_frame)
        if m == 5:
            return float('inf')
        total_m = 0
        unblocked_score = 0
        for num_pieces in num_pieces_per_frame:
            if num_pieces == m:
                total_m += 1
        if m == 3:
            unblocked_score = (total_m - 1) / 2
        elif total_m > 1:
            unblocked_score = 1
        return float(min(m + unblocked_score, LENGTH_NEEDED))

    @staticmethod
    def get_num_pieces_per_frame(offset_seq, player):
        """
        >>> AIInputAgent.get_num_pieces_per_frame([1, None, None, 1, None, 0, 1, None, 1, None, 1], 1)
        [2, 0, 0, 0, 0, 0, 3]
        >>> AIInputAgent.get_num_pieces_per_frame([1, None, None, 1, 1], 1)
        [3]
        """
        len_offset_seq = len(offset_seq)
        if len_offset_seq < LENGTH_NEEDED:
            return [0]
        num_pieces_per_frame = [0 for _ in range(len_offset_seq - LENGTH_NEEDED + 1)]
        len_frames = len(num_pieces_per_frame)
        for piece_index, piece in enumerate(offset_seq):
            frame_start_inclusive = max(piece_index - LENGTH_NEEDED + 1, 0)
            frame_end_exclusive = min(piece_index + 1, len_frames)
            if piece == player:
                value = 1
            elif piece is EMPTY_PIECE:
                value = 0
            else:
                value = -LENGTH_NEEDED
            for frame_index in range(frame_start_inclusive, frame_end_exclusive):
                num_pieces_per_frame[frame_index] += value
        for frame_index, value in enumerate(num_pieces_per_frame):
            if value < 0:
                num_pieces_per_frame[frame_index] = 0
        return num_pieces_per_frame

    @staticmethod
    # returns an ordered list of pieces within LENGTH_NEEDED of coord along the offset direction
    def get_sequence(board, coord, offset):
        """
        >>> board = { (7, 0) : 1, (6, 1) : 0, (5, 2) : 0, (4, 3) : 0, (3, 4) : -1, (2, 5) : 1, (1, 6) : 0, (0, 7) : 1}
        >>> AIInputAgent.get_sequence(board, (3, 4), (1, -1))
        [1, 0, 0, 0, -1, 1, 0, 1]
        """
        forward_offset_seq = []
        reverse_offset_seq = []
        for func, offset_seq in [(np.add, forward_offset_seq), (np.subtract, reverse_offset_seq)]:
            new_coord = coord
            for i in range(LENGTH_NEEDED - 1):
                new_coord = func(new_coord, offset)
                if out_of_bound(new_coord):
                    break
                new_coord = tuple(new_coord)
                piece = board.get(new_coord, EMPTY_PIECE)
                offset_seq.append(piece)
        piece = board.get(coord, EMPTY_PIECE)
        return list(reversed(forward_offset_seq)) + [piece] + reverse_offset_seq

    @staticmethod
    def get_score_from_direction_scores(direction_scores):
        score = 0
        for ds in direction_scores:
            score += 10 ** ds
        return score

    # evaluate the sequence lengths with or without the piece at coord
    def evaluate_piece(self, board, coord, player):
        """
        >>> board = {}
        >>> board[(3, 4)] = 1
        >>> aiAgent = AIInputAgent(None, None)
        >>> aiAgent.evaluate_piece(board, (3, 4), 1)
        400.0
        >>> aiAgent.evaluate_piece(board, (3, 5), 1)
        130.0
        >>> board[(3, 5)] = 1
        >>> board[(4, 5)] = 1
        >>> aiAgent.evaluate_piece(board, (4, 5), 1)
        2200.0
        """
        direction_scores = [AIInputAgent.evaluate_direction(board, coord, player, offset) for offset in OFFSETS]
        return AIInputAgent.get_score_from_direction_scores(direction_scores)

    def get_move(self, board, prev_moves, curr_player):
        raise RuntimeError('Unimplemented')

class ReflexAgent(AIInputAgent):
    def get_relevant_move_scores(self, board, moves, curr_player):
        scores = {}
        for move, relevant_players in moves.items():
            for relevant_player, should_evaluate_player in enumerate(relevant_players):
                if should_evaluate_player:
                    board[move] = relevant_player
                    move_score = self.evaluate_piece(board, move, relevant_player)

                    is_opponent = (curr_player != relevant_player)
                    if is_opponent:
                        move_score = self.adjust_opponent_score(move_score)

                    prev_score = scores.get(move, 0)
                    scores[move] = prev_score + move_score
                    del board[move]
        return scores

    def get_move(self, board, prev_moves, curr_player):
        if len(prev_moves) == 0:
            return (BOARD_HEIGHT // 2, BOARD_WIDTH // 2)
        final_scored_moves = self.get_scored_moves(board, prev_moves, curr_player)
        max_score, best_move = max(final_scored_moves)
        return best_move

    def get_scored_moves(self, board, prev_moves, curr_player):
        opponent = (curr_player + 1) % NUM_PLAYERS
        relevant_moves = self.get_nearby_empty_locations(board)

        player_move_scores = self.get_relevant_move_scores(board, relevant_moves, curr_player)
        final_scored_moves = [(score, move) for move, score in player_move_scores.items()]
        return final_scored_moves

    def adjust_opponent_score(self, score):
        return score / 2

def get_empty_location():
    return [False, False]

def get_empty_piece_evaluation():
    return [[None, [None, None, None, None]], [None, [None, None, None, None]]]

class CachedAIAgent(AIInputAgent):
    def __init__(self, player_num, display):
        super().__init__(player_num, display)
        self.moves_seen = []
        self.empty_locations = {}
        self.piece_evaluations = {}

    def process_new_moves(self, board, prev_moves):
        search_radius = LENGTH_NEEDED // 2
        curr_player = None
        previous_empty_locations = {}

        def add_to_empty_locations(new_coord):
            if not out_of_bound(new_coord) and new_coord not in board:
                location_nearby_players = self.empty_locations.setdefault(new_coord, get_empty_location())
                if not location_nearby_players[curr_player]:
                    previous_empty_locations.setdefault(new_coord, list(location_nearby_players))
                    location_nearby_players[curr_player] = True
                return True
            return False

        new_moves = prev_moves[len(self.moves_seen):]
        for coord, curr_player in new_moves:
            # add to locations to try to evaluate
            move_location = self.empty_locations.pop(coord, get_empty_location())
            previous_empty_locations.setdefault(coord, move_location)
            iterate_all_directions(coord, search_radius, add_to_empty_locations)

            # invalidate nearby cached evaluations
            self.piece_evaluations.pop(coord, None)
            for index, offset in enumerate(OFFSETS):
                for func in (add, sub):
                    affected_coord = coord
                    for r in range(LENGTH_NEEDED - 1):
                        affected_coord = func(affected_coord, offset)

                        player = board.get(affected_coord, EMPTY_PIECE)
                        if player == EMPTY_PIECE:
                            piece_evals = self.piece_evaluations.setdefault(affected_coord, get_empty_piece_evaluation())
                            for player, score_direction_scores in enumerate(piece_evals):
                                score_direction_scores[0] = None # set the score to None
                                direction_scores = score_direction_scores[1]
                                direction_scores[index] = None
            self.moves_seen.append(coord)
        return previous_empty_locations

    def get_nearby_empty_locations(self, board):
        return self.empty_locations

    def evaluate_piece(self, board, coord, player):
        player_evals = self.piece_evaluations.setdefault(coord, get_empty_piece_evaluation())
        score_direction_scores = player_evals[player]
        score = score_direction_scores[0]
        if score != None:
            return score
        direction_scores = score_direction_scores[1]
        for offset_index, direction_score in enumerate(direction_scores):
            if direction_score == None:
                direction_score = AIInputAgent.evaluate_direction(board, coord, player, OFFSETS[offset_index])
                direction_scores[offset_index] = direction_score
        score = AIInputAgent.get_score_from_direction_scores(direction_scores)
        score_direction_scores[0] = score
        return score

class ReflexCachedAgent(CachedAIAgent, ReflexAgent):
    def get_scored_moves(self, board, prev_moves, curr_player):
        # time.sleep(1)
        opponent = (curr_player + 1) % NUM_PLAYERS
        prev_empty_locations = self.process_new_moves(board, prev_moves)
        relevant_moves = self.get_nearby_empty_locations(board)

        player_move_scores = self.get_relevant_move_scores(board, relevant_moves, curr_player)
        final_scored_moves = [(score, move) for move, score in player_move_scores.items()]
        return final_scored_moves

# class MinimaxAgent(ReflexAgent):
#     def get_move(self, board, prev_moves, curr_player, minimax_depth=4):
#         time.sleep(2)

#         def get_move_minimax(player, depth):
#             final_scored_moves = self.get_scored_moves(board, prev_moves, curr_player)
#             if len(final_scored_moves) == 0:
#                 return (0, (BOARD_HEIGHT // 2, BOARD_WIDTH // 2))

#             if depth <= 1:
#                 max_score, move = max(final_scored_moves)
#                 return (max_score, move)

#             relevant_moves = self.get_sequence_locations(board, curr_player)
#             player_move_scores = self.get_relevant_move_scores(board, relevant_moves, curr_player)
#             sorted_scored_moves = sorted([(score, move) for move, score in player_move_scores.items()], reverse=True)
#             # prune everything besides the moves with the best scores
#             if len(sorted_scored_moves) > 3:
#                 sorted_scored_moves = sorted_scored_moves[:3]

#             opponent = (player + 1) % NUM_PLAYERS
#             opponent_relevant_moves = self.get_sequence_locations(board, opponent)
#             opponent_move_scores = self.get_relevant_move_scores(board, opponent_relevant_moves, opponent)
#             opponent_scored_moves = sorted([(score, move) for move, score in opponent_move_scores.items()], reverse=True)
#             if len(opponent_scored_moves) > 3:
#                 opponent_scored_moves = opponent_scored_moves[:3]

#             for score, move in opponent_scored_moves:
#                 for player_score, player_move in sorted_scored_moves:
#                     if player_move == move:
#                         break
#                 else:
#                     sorted_scored_moves.append((0, move))

#             print('best', sorted_scored_moves)
#             minimax_scores = []
#             for score, move in sorted_scored_moves:
#                 if math.isinf(score):
#                     return (score, move)
#                 board[move] = player
#                 best_opponent_score, best_opponent_move = get_move_minimax(opponent, depth - 1)
#                 minimax_scores.append((score - best_opponent_score, move))
#                 del board[move]
#             best_overall_move = max(minimax_scores)
#             return best_overall_move
#         print()
#         best_score, best_move = get_move_minimax(curr_player, minimax_depth)
#         return best_move
