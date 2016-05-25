from config import *

class Display(object):
    def __init__(self, board, moves):
        self.board = board
        self.moves = moves

    def print_message(self, message):
        raise RuntimeError('Unimplemented')

    def update_board(self, highlighted):
        raise RuntimeError('Unimplemented')

class CommandLineDisplay(Display):
    def print_message(self, message):
        print(message)

    def update_board(self, highlighted=[]):
        row_format ="{:>3}" * (BOARD_WIDTH + 2)
        column_label = row_format.format("", *range(BOARD_WIDTH), "")
        print(column_label)

        # fills out a matrix corresponding to the current board
        board_matrix = [['+' for w in range(BOARD_WIDTH)] for l in range(BOARD_HEIGHT)]
        for (x, y), player in self.board.items():
            board_matrix[y][x] = str(player)

        # print the actual rows of the board with a label to the left and right
        for row_num, row in enumerate(board_matrix):
            print(row_format.format(row_num, *row, row_num))

        print(column_label)