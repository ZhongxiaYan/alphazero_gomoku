from config import *
import sys
from PyQt4 import QtCore, QtGui

class Display(object):
    def __init__(self, game):
        self.game = game

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
        for (x, y), player in self.game.board.items():
            board_matrix[y][x] = str(player)

        # print the actual rows of the board with a label to the left and right
        for row_num, row in enumerate(board_matrix):
            print(row_format.format(row_num, *row, row_num))

        print(column_label)

# not only contains the logic to send/receive signals to/from GUI,
# also contains the main program loop, since this is on the nonGUI thread anyways
class PyQtDisplay(Display, QtCore.QObject):
    start_main_loop = QtCore.pyqtSignal()

    def __init__(self, game):
        Display.__init__(self, game)
        QtCore.QObject.__init__(self)
        self.start_main_loop.connect(self.run)

    def print_message(self, message):
        import time
        while True:
            time.sleep(1)
            print("hi")

    def update_board(self, highlighted=[]):
        import time
        while True:
            time.sleep(1)
            print("hi")

    @QtCore.pyqtSlot()
    def run(self):
        # main loop
        while True:
            self.game.show_board()
            self.game.transition(self.game.get_input())
            if self.game.has_ended():
                break
