from agent import Agent
from config import *

from collections import defaultdict

class AIInputAgent(Agent):

    # return all the positions that may extend at least one current sequence
    def get_sequence_locations(self, board, player):
        """
        >>> board = { (2, 7) : 0, (3, 5): 0, (3, 6) : 0, (4, 4) : 0, (4, 6) : 0, (4, 7) : 0, (5, 4) : 1, (5, 5) : 0, (5, 6) : 1, (6, 5) : 0}
        >>> aiAgent = AIInputAgent(None, None)
        >>> sorted(aiAgent.get_sequence_locations(board, 1))
        [(3, 2), (3, 4), (4, 3), (4, 5), (5, 2), (5, 3), (5, 7), (5, 8), (6, 3), (6, 4), (6, 6), (6, 7), (7, 2), (7, 4), (7, 6), (7, 8)]
        """
        new_locations = defaultdict(int)
        not_piece_value = -(LENGTH_NEEDED * 2) ** 2 # large enough negative number
        for (y, x), piece_player in board.items():
            coord = (y, x)
            new_locations[coord] += not_piece_value
            if player != piece_player:
                continue
            for (y_off, x_off) in OFFSETS:
                for y_off_dir, x_off_dir in [(y_off, x_off), (-y_off, -x_off)]:
                    for r in range(1, 3):
                        new_coord = (y + y_off_dir * r, x + x_off_dir * r)
                        if new_coord in board:
                            break
                        new_locations[new_coord] += 1
        return [coord for coord, value in new_locations.items() if value > 0 and not self.out_of_bound(coord)]

    def out_of_bound(self, coord):
        y, x = coord
        return not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT)

    # evaluate the sequence lengths with or without the piece at coord
    def evaluate_piece(self, board, coord, player):
        """
        >>> board = { (1, 8) : 1, (2, 7) : 0, (3, 4) : 1, (3, 5): 0, (3, 6) : 0, (4, 4) : 0, (4, 6) : 0, (4, 7) : 0, (5, 4) : 1, (5, 5) : 1, (5, 6) : 1}
        >>> aiAgent = AIInputAgent(None, None)
        >>> aiAgent.evaluate_piece(board, (4, 5), 0)
        ([(0, 1), (0, 0), (2, 1), (2, 0)], [(1, 0), (1, 1), (0, 0), (1, 1)], [2, 1, 4, 3], [1, 2, 0, 2])
        """
        (y, x) = coord
        seq_amounts_with_piece = []
        seq_amounts_without_piece = []
        blocked_ends_with_piece = []
        blocked_ends_without_piece = []

        for y_off, x_off in OFFSETS:
            opposite_sequences_count = []
            blocked_ends = []
            # can either go the direction of the offset or reverse direction of the offset
            for y_off_dir, x_off_dir in [(y_off, x_off), (-y_off, -x_off)]:
                count = 0
                coord_curr = (y_cur, x_cur) = (y + y_off_dir, x + x_off_dir)
                while coord_curr in board and board[coord_curr] == player:
                    y_cur += y_off_dir
                    x_cur += x_off_dir
                    coord_curr = (y_cur, x_cur)
                    count += 1
                opposite_sequences_count.append(count)
                # end is blocked if terminated by another player's piece or out of board's boundary
                blocked_ends.append(1 if (self.out_of_bound(coord_curr) or coord_curr in board) else 0)
            seq_amounts_without_piece.append(tuple(opposite_sequences_count))
            seq_amounts_with_piece.append(sum(opposite_sequences_count) + 1)
            blocked_ends_without_piece.append(tuple(blocked_ends))
            blocked_ends_with_piece.append(sum(blocked_ends))
        return seq_amounts_without_piece, blocked_ends_without_piece, seq_amounts_with_piece, blocked_ends_with_piece

class ReflexAgent(AIInputAgent):
    def get_move(self, board, prev_moves, curr_player):
        pass