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
        row_format = '{:>3}' * (BOARD_WIDTH + 2)
        column_label = row_format.format('', *range(BOARD_WIDTH), '')
        print(column_label)

        # fills out a matrix corresponding to the current board
        board_matrix = [['+' for w in range(BOARD_WIDTH)] for l in range(BOARD_HEIGHT)]
        for (y, x), player in self.game.board.items():
            board_matrix[y][x] = str(player)

        # print the actual rows of the board with a label to the left and right
        for row_num, row in enumerate(board_matrix):
            print(row_format.format(row_num, *row, row_num))

        print(column_label)

# not only contains the logic to send/receive signals to/from GUI,
# also contains the main program loop, since this is on the nonGUI thread anyways
class PyQtDisplay(Display, QtCore.QObject):
    signal_start_main_loop = QtCore.pyqtSignal()
    signal_highlight_pieces = QtCore.pyqtSignal(list)
    signal_display_board = QtCore.pyqtSignal()

    def __init__(self, game):
        Display.__init__(self, game)
        QtCore.QObject.__init__(self)
        self.signal_start_main_loop.connect(self.run)
        self.result = None

    ## starts the main logic thread
    @QtCore.pyqtSlot()
    def run(self):
        # main loop
        self.event_loop = QtCore.QEventLoop()
        while True:
            self.game.show_board()
            if self.game.has_ended():
                break
            self.game.transition(self.game.get_input())

    ## signals to the gui thread ##

    def print_message(self, message):
        return

    def update_board(self, highlighted=[]):
        self.signal_display_board.emit()
        if highlighted != []:
            self.signal_highlight_pieces.emit(highlighted)

    def wait_input(self):
        self.event_loop.exec_() # block until get the signal from input source
        return self.result

    ## slots to handle signals from the gui thread ##

    @QtCore.pyqtSlot(int, int)
    def slot_coord(self, y, x):
        self.result = (y, x)
        self.event_loop.quit()

    @QtCore.pyqtSlot(str, int)
    def slot_command(self, command, num):
        self.result = (command, num)
        self.event_loop.quit()
