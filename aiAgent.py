from agent import Agent
from config import *

from collections import defaultdict
import itertools
import functools
import random
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
    def get_move_options(self, board):
        """
        >>> board = { (2, 7) : 0, (3, 5) : 0, (3, 6) : 1}
        >>> aiAgent = AIInputAgent(None, None)
        >>> sorted([(coord, tuple(players)) for coord, players in aiAgent.get_move_options(board).items()])
        [((0, 5), (True, False)), ((0, 7), (True, False)), ((0, 9), (True, False)), ((1, 3), (True, False)), ((1, 4), (False, True)), ((1, 5), (True, False)), ((1, 6), (True, True)), ((1, 7), (True, False)), ((1, 8), (True, False)), ((2, 4), (True, False)), ((2, 5), (True, True)), ((2, 6), (True, True)), ((2, 8), (True, False)), ((2, 9), (True, False)), ((3, 3), (True, False)), ((3, 4), (True, False)), ((3, 7), (True, True)), ((3, 8), (True, True)), ((4, 4), (True, False)), ((4, 5), (True, True)), ((4, 6), (True, True)), ((4, 7), (True, True)), ((4, 9), (True, False)), ((5, 3), (True, False)), ((5, 4), (False, True)), ((5, 5), (True, False)), ((5, 6), (False, True)), ((5, 7), (True, False)), ((5, 8), (False, True))]
        """
        search_radius = LENGTH_NEEDED // 2
        nearby_empty_locations = {}
        new_coord_player = None

        def set_nearby_empty_location(new_coord):
            if not out_of_bound(new_coord) and new_coord not in board:
                location_nearby_players = nearby_empty_locations.get(new_coord, EMPTY_PIECE)
                if location_nearby_players == EMPTY_PIECE:
                    location_nearby_players = get_empty_coord_option()
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
        # print(sorted(final_scored_moves, reverse=True))
        max_score, best_move = max(final_scored_moves)
        return best_move

    def get_scored_moves(self, board, prev_moves, curr_player):
        opponent = (curr_player + 1) % NUM_PLAYERS
        relevant_moves = self.get_move_options(board)

        player_move_scores = self.get_relevant_move_scores(board, relevant_moves, curr_player)
        final_scored_moves = [(score, move) for move, score in player_move_scores.items()]
        return final_scored_moves

    def adjust_opponent_score(self, score):
        return score / 2

def get_empty_coord_option():
    return [False, False]

def get_empty_piece_evaluation():
    return [[None, [None, None, None, None]], [None, [None, None, None, None]]]

class CachedAIAgent(AIInputAgent):
    def __init__(self, player_num, display):
        super().__init__(player_num, display)
        self.moves_seen = []
        self.move_options = {}
        self.piece_evaluations = {}

    def process_new_moves(self, board, new_moves):
        search_radius = LENGTH_NEEDED // 2
        curr_player = None
        prev_move_options = {} # maps coord to data in the format of empty_coord_option (e.g. [False, False])
        prev_evaluations = {} # maps coord to data in the format of empty_piece_evaluation (e.g. [45, [3, 4, 5, 6]])

        def add_coord_to_move_option(new_coord):
            if not out_of_bound(new_coord) and new_coord not in board:
                new_coord_move_options = self.move_options.setdefault(new_coord, get_empty_coord_option())
                if not new_coord_move_options[curr_player]:
                    prev_move_options.setdefault(new_coord, list(new_coord_move_options))
                    new_coord_move_options[curr_player] = True
                return True
            return False

        for coord, curr_player in new_moves:
            # remove coord from being considered as a move option
            coord_option = self.move_options.pop(coord, get_empty_coord_option())
            prev_move_options.setdefault(coord, coord_option)
            # update surrounding coords for being considered as move options
            iterate_all_directions(coord, search_radius, add_coord_to_move_option)

            # invalidate previous evaluations for directions if needed
            for index, offset in enumerate(OFFSETS):
                for func in (add, sub):
                    affected_coord = coord
                    for r in range(LENGTH_NEEDED - 1):
                        affected_coord = func(affected_coord, offset)

                        coord_piece_evals = self.piece_evaluations.setdefault(affected_coord, get_empty_piece_evaluation())
                        coord_piece_evals_backup = []
                        for player, score_direction_scores in enumerate(coord_piece_evals):
                            direction_scores = score_direction_scores[1]
                            coord_piece_evals_backup.append([score_direction_scores[0], list(direction_scores)])

                            score_direction_scores[0] = None # set the score to None
                            direction_scores[index] = None
                        prev_evaluations.setdefault(affected_coord, coord_piece_evals_backup)

        def revert_moves():
            self.move_options.update(prev_move_options)
            self.piece_evaluations.update(prev_evaluations)

        return revert_moves

    def get_move_options(self, board):
        return self.move_options

    def evaluate_piece(self, board, coord, player):
        player_evals = self.piece_evaluations.setdefault(coord, get_empty_piece_evaluation())
        score_direction_scores = player_evals[player]
        score = score_direction_scores[0]
        if score is not None:
            return score
        direction_scores = score_direction_scores[1]
        for offset_index, direction_score in enumerate(direction_scores):
            if direction_score is None:
                direction_score = AIInputAgent.evaluate_direction(board, coord, player, OFFSETS[offset_index])
                direction_scores[offset_index] = direction_score
        score = AIInputAgent.get_score_from_direction_scores(direction_scores)
        score_direction_scores[0] = score
        return score

class ReflexCachedAgent(CachedAIAgent, ReflexAgent):
    def get_scored_moves(self, board, prev_moves, curr_player):
        # time.sleep(1)
        new_moves = prev_moves[len(self.moves_seen):]
        self.process_new_moves(board, new_moves)
        relevant_moves = self.get_move_options(board)

        player_move_scores = self.get_relevant_move_scores(board, relevant_moves, curr_player)
        final_scored_moves = [(score, move) for move, score in player_move_scores.items()]
        return final_scored_moves

MAX_BRANCH_FACTOR = 6

class MinimaxAgent(ReflexCachedAgent):
    def get_relevant_move_scores(self, board, moves, curr_player):
        scores = {}
        for move, relevant_players in moves.items():
            for relevant_player, should_evaluate_player in enumerate(relevant_players):
                if should_evaluate_player:
                    is_opponent = (curr_player != relevant_player)

                    if is_opponent:
                        move_score = self.evaluate_piece(board, move, relevant_player)
                        # move_score = self.adjust_opponent_score(move_score)
                    else:
                        board[move] = relevant_player
                        move_score = self.evaluate_piece(board, move, relevant_player)
                        del board[move]

                    prev_score = scores.get(move, 0)
                    scores[move] = prev_score + move_score
        return scores

    def get_move(self, board, prev_moves, curr_player, minimax_depth=6):
        # time.sleep(0.5)
        if len(prev_moves) == 0:
            return (BOARD_HEIGHT // 2, BOARD_WIDTH // 2)

        # print('\n\n')
        new_moves = prev_moves[len(self.moves_seen):]
        self.process_new_moves(board, new_moves)
        self.moves_seen.extend(new_moves)
        relevant_moves = self.get_move_options(board)
        for coord, move_options in relevant_moves.items():
            for player, should_evaluate in enumerate(move_options):
                if should_evaluate:
                    board[coord] = player
                    self.evaluate_piece(board, coord, player)
                    del board[coord]

        # returns the best move and its corresponding score (maximum)
        def get_move_minimax(player, depth, previous_score, alphabeta):
            relevant_moves = self.get_move_options(board)
            player_move_scores = self.get_relevant_move_scores(board, relevant_moves, player)
            scored_moves = [(score, move) for move, score in player_move_scores.items()]
            if depth == 0:
                max_score, max_coord = max(scored_moves)
                # print('\t' * (minimax_depth - depth) + str(max_coord) + ' ' + str(max_score - previous_score) + ' ' + str(alphabeta))
                return (max_score - previous_score, max_coord)
            opponent = (player + 1) % NUM_PLAYERS

            scored_moves.sort(reverse=True)
            minimax_move_scores = []
            for score, coord in scored_moves[:MAX_BRANCH_FACTOR]:
                board[coord] = player
                self.moves_seen.append((coord, player))
                reverse_moves = self.process_new_moves(board, [(coord, player)])

                self_score = self.evaluate_piece(board, coord, player)
                # print('\t' * (minimax_depth - depth) + str(coord))
                overall_score, next_move = get_move_minimax(opponent, depth - 1, self_score - previous_score, list(alphabeta))
                # print('\t' * (minimax_depth - depth) + str(coord) + ' ' + str(overall_score) + ' ' + str(alphabeta))

                reverse_moves()
                self.moves_seen.pop()
                del board[coord]

                if overall_score < alphabeta[opponent]:
                    # print('\t' * (minimax_depth - depth) + 'broke')
                    return (overall_score, coord)
                minimax_move_scores.append((overall_score, coord))
                alphabeta[player] = max(alphabeta[player], overall_score)

            return max(minimax_move_scores)

        best_score, best_move = get_move_minimax(curr_player, minimax_depth, 0, [-float('inf')] * NUM_PLAYERS)
        return best_move

FULL_BOARD = (1 << BOARD_WIDTH * BOARD_HEIGHT) - 1

class MCTSAgent(AIInputAgent):
    def __init__(self, player_num, display):
        super().__init__(player_num, display)
        self.board_state_playouts = defaultdict(lambda: (0, 0)) # maps a board state to the (total times seen, total times won)
        self.neighbor_masks = MCTSAgent.generate_neighbor_masks(2)

    @staticmethod
    def get_mask_from_coord(coord):
        y, x = coord
        index = y * BOARD_WIDTH + x
        return 1 << index

    @staticmethod
    def get_mask_from_coords(coord_list):
        mask = 0
        for coord in coord_list:
            mask |= (1 << MCTSAgent.get_index_from_coord(coord))
        return mask

    @staticmethod
    def get_index_from_coord(coord):
        y, x = coord
        return y * BOARD_WIDTH + x

    @staticmethod
    def generate_neighbor_masks(mask_range):
        """
        >>> neighbor_masks = MCTSAgent.generate_neighbor_masks(2)
        >>> sample_index = MCTSAgent.get_index_from_coord((2, 2))
        >>> sample_mask = neighbor_masks[sample_index]
        >>> middle_equal = True
        >>> for y in range(2, BOARD_HEIGHT - 2):
        ...     for x in range(2, BOARD_WIDTH - 2):
        ...         index = MCTSAgent.get_index_from_coord((y, x))
        ...         middle_equal = middle_equal and (sample_mask << (index - sample_index) == neighbor_masks[index])
        >>> middle_equal
        True
        >>> neighbors_1_1 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (3, 1), (3, 3)]
        >>> mask_1_1 = MCTSAgent.get_mask_from_coords(neighbors_1_1)
        >>> mask_1_1 == neighbor_masks[MCTSAgent.get_index_from_coord((1, 1))]
        True
        >>> neighbors_14_14 = [(14, 14), (14, 13), (14, 12), (13, 14), (13, 13), (12, 14), (12, 12)]
        >>> mask_14_14 = MCTSAgent.get_mask_from_coords(neighbors_14_14)
        >>> mask_14_14 == neighbor_masks[MCTSAgent.get_index_from_coord((14, 14))]
        True
        """
        board_masks = [0 for _ in range(BOARD_HEIGHT * BOARD_WIDTH)]
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                mask = 0 # bit mask locating the neighbors of the coordinate (x, y)
                for r in range(1, mask_range + 1):
                    for v_offset in (-1, 0, 1):
                        for h_offset in (-1, 0, 1):
                            coord = (y + r * v_offset, x + r * h_offset)
                            if not out_of_bound(coord):
                                mask |= MCTSAgent.get_mask_from_coord(coord)
                board_masks[MCTSAgent.get_index_from_coord((y, x))] = mask
        return board_masks

    @staticmethod
    def get_compressed_board(board):
        compressed_board = [0, 0]
        for coord, player in board.items():
            compressed_board[player] |= MCTSAgent.get_mask_from_coord(coord)
        return tuple(compressed_board)

    @staticmethod
    def is_full_board(board_state):
        """
        >>> board_state = (FULL_BOARD ^ (1 << 3 | 1 << 100), (1 << 3 | 1 << 100))
        >>> MCTSAgent.is_full_board(board_state)
        True
        >>> MCTSAgent.is_full_board((1232, 1231212))
        False
        """
        all_player_board = 0
        for player_state in board_state:
            all_player_board |= player_state
        return all_player_board == FULL_BOARD

    @staticmethod
    def has_won(board_state, move, player):
        """
        >>> coord_list = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        >>> board_state = (MCTSAgent.get_mask_from_coords(coord_list), 0)
        >>> functools.reduce(lambda x, y: x and y, map(lambda move: MCTSAgent.has_won(board_state, move, 0), coord_list))
        True
        >>> functools.reduce(lambda x, y: x and y, map(lambda move: MCTSAgent.has_won(board_state, move, 1), coord_list))
        False
        >>> coord_list = [(2, 3), (3, 4), (4, 5), (5, 6)]
        >>> board_state = (MCTSAgent.get_mask_from_coords(coord_list), 0)
        >>> functools.reduce(lambda x, y: x and y, map(lambda move: MCTSAgent.has_won(board_state, move, 0), coord_list))
        False
        """
        player_board_state = board_state[player]
        for (y_off, x_off) in OFFSETS:
            direction_length = 1
            for y_off_dir, x_off_dir in ((y_off, x_off), (-y_off, -x_off)):
                new_y, new_x = move
                for _ in range(LENGTH_NEEDED):
                    new_y += y_off_dir
                    new_x += x_off_dir
                    new_coord = (new_y, new_x)
                    if out_of_bound(new_coord):
                        break
                    elif player_board_state & MCTSAgent.get_mask_from_coord(new_coord) != 0:
                        direction_length += 1
                    else:
                        break
            if direction_length >= LENGTH_NEEDED:
                return True
        return False

    def get_next_states(self, board_state, player):
        player_state = board_state[player]
        opponent = (player + 1) % 2
        opponent_player_state = board_state[opponent]
        combined_board_state = player_state | opponent_player_state

        viable_moves = []
        neighbor_masks = self.neighbor_masks
        for index, index_neighbor_mask in enumerate(neighbor_masks):
            index_mask = (1 << index)
            # check no piece at index and that index HAS neighbors
            if (combined_board_state & index_mask == 0) and (index_neighbor_mask & combined_board_state != 0):
                move = (index // BOARD_WIDTH, index % BOARD_WIDTH)
                new_board_state = [0, 0]
                new_board_state[player] = player_state | index_mask
                new_board_state[opponent] = opponent_player_state
                viable_moves.append((tuple(new_board_state), move))
        return viable_moves

    def get_max_score_state_and_move(self, next_states_and_moves, player):
        total_playouts = 0
        board_state_data = []
        random.shuffle(next_states_and_moves)
        for board_state, move in next_states_and_moves:
            board_state_total_playouts, board_state_won_playouts = self.board_state_playouts[board_state]
            if board_state_total_playouts == 0: # must visit previously unvisited node
                return (board_state, move)
            total_playouts += board_state_total_playouts
            board_state_data.append((board_state, move, board_state_total_playouts, board_state_won_playouts))
        max_board_state = None
        max_move = None
        max_score = -float('inf')
        log = math.log
        sqrt = math.sqrt
        for board_state, move, board_state_total_playouts, board_state_won_playouts in board_state_data:
            score = board_state_won_playouts / board_state_total_playouts + sqrt(2 * log(total_playouts) / board_state_total_playouts) # UCB1 strategy for MCTS
            if score > max_score:
                max_score = score
                max_board_state = board_state
                max_move = move
        return (max_board_state, max_move)

    def search(self, board, curr_player, max_search_time):
        compressed_board = MCTSAgent.get_compressed_board(board)
        start_time = time.time()
        base_next_states_and_moves = self.get_next_states(compressed_board, curr_player)
        num_simulations = 0
        while time.time() - start_time < max_search_time or num_simulations < 50000:
            num_simulations += 1
            simulated_player = curr_player
            next_states_and_moves = base_next_states_and_moves
            board_stack = []
            while True:
                board_state, move = self.get_max_score_state_and_move(next_states_and_moves, simulated_player)
                board_stack.append(board_state)

                if MCTSAgent.has_won(board_state, move, simulated_player):
                    winner = simulated_player
                    break
                elif MCTSAgent.is_full_board(board_state):
                    winner = None
                    break
                simulated_player = (simulated_player + 1) % NUM_PLAYERS
                next_states_and_moves = self.get_next_states(board_state, simulated_player)
            curr_player_win_score = 1 if winner == curr_player else 0.5 if winner is None else 0
            for board_state in board_stack:
                total_playouts, winning_playouts = self.board_state_playouts[board_state]
                self.board_state_playouts[board_state] = (total_playouts + 1, winning_playouts + curr_player_win_score)
                curr_player_win_score = 1 - curr_player_win_score
        board_state, move = self.get_max_score_state_and_move(base_next_states_and_moves, curr_player)
        print('Number of simulations: %s' % (num_simulations,))
        return move

    def get_move(self, board, prev_moves, curr_player, max_move_time=5.0):
        if len(prev_moves) == 0:
            return (BOARD_HEIGHT // 2, BOARD_WIDTH // 2)
        return self.search(board, curr_player, max_move_time)
