from config import *
from game import *

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import numpy as np

class Board(QWidget):

    signal_coord = pyqtSignal(int, int)
    signal_command = pyqtSignal(str, int)

    def __init__(self, game):
        super(Board, self).__init__()
        self.game = game
        self.init_ui()
        self.signal_coord.connect(self.game.display.slot_coord) # notify the loop thread when a move occurs
        self.signal_command.connect(self.game.display.slot_command) # notify the loop thread when a command (e.g. undo) occurs
        self.game.display.signal_highlight_pieces.connect(self.slot_highlight_pieces)
        self.game.display.signal_display_board.connect(self.display_board)
        self.moves = []

    # generates a color for the curr_player that has rdb values between 0 and 255
    def generate_color(self, curr_player):
        rgb = int(curr_player * 0xFF / (NUM_PLAYERS - 1))
        rgb_hex_str = '%X' % rgb
        return '#' + rgb_hex_str * 3

    def style_button(self, button, color, hover_color):
        button_style_string = """
            .QPushButton {
                background-color: %s;
                border-radius: %spx;
            }
            .QPushButton:hover {
                background-color: %s;
            }
        """ % (color, self.piece_radius - 0.5, hover_color)
        button.setStyleSheet(button_style_string)

    @pyqtSlot()
    def display_board(self):
        displayed_moves_len = len(self.moves)
        board_moves_len = len(self.game.moves)
        if displayed_moves_len < board_moves_len: # new moves need to be updated
            for move in self.game.moves[displayed_moves_len:]:
                (y, x), player = move
                button = self.buttons[y][x]
                button.clicked.disconnect(self.on_button_click)
                self.style_button(button, self.generate_color(player), '')
                self.moves.append(move)
        elif board_moves_len < displayed_moves_len: # undo has been called
            for move in self.moves[board_moves_len:]:
                (y, x), player = move
                button = self.buttons[y][x]
                button.clicked.connect(self.on_button_click)
                self.style_button(button, '', 'green')
                self.moves.pop()

    # called when a button is clicked
    @pyqtSlot()
    def on_button_click(self):
        button = self.sender()
        self.signal_coord.emit(button.y_index, button.x_index)

    # signaled by the game controls (nonGui thread in PyQtDisplay)
    @pyqtSlot(list)
    def slot_highlight_pieces(self, pieces):
        board = self.game.board
        for (y, x), button in np.ndenumerate(self.buttons):
            if (y, x) in board:
                continue
            self.style_button(button, '', '')
            button.clicked.disconnect()
        for (y, x) in pieces:
            self.style_button(self.buttons[y][x], 'yellow', '')

    # creates the buttons and set the background
    def init_ui(self):
        self.buttons = [[QPushButton("", self) for x in range(BOARD_WIDTH)] for y in range(BOARD_HEIGHT)]

        for (y, x), button in np.ndenumerate(self.buttons):
            button.y_index = y
            button.x_index = x
            button.clicked.connect(self.on_button_click)

        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor('#CC9900'))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    # center the board to take up as much space as possible
    def calc_dimensions(self):
        geom = self.geometry()
        y_start = geom.y()
        x_start = geom.x()
        width = geom.width()
        height = geom.height()

        v_max_spacing = height // BOARD_HEIGHT
        h_max_spacing = width // BOARD_WIDTH
        self.spacing = min(h_max_spacing, v_max_spacing)

        h_board = self.spacing * BOARD_HEIGHT
        w_board = self.spacing * BOARD_WIDTH

        v_board_start = y_start + (height - h_board) // 2
        h_board_start = x_start + (width - w_board) // 2

        self.v_line_start = v_board_start + self.spacing // 2
        self.h_line_start = h_board_start + self.spacing // 2
        self.v_line_end = self.v_line_start + self.spacing * (BOARD_HEIGHT - 1)
        self.h_line_end = self.h_line_start + self.spacing * (BOARD_WIDTH - 1)

    # always called when generating / resizing a window. Before paintEvent
    def resizeEvent(self, event):
        self.calc_dimensions()
        piece_diam = self.spacing * 0.9
        self.piece_radius = piece_radius = piece_diam / 2
        for (y, x), button in np.ndenumerate(self.buttons):
            if (y, x) not in self.game.board:
                self.style_button(button, '', 'green')
            else:
                self.style_button(button, self.generate_color(self.game.board[(y, x)]), '')
            button.resize(piece_diam, piece_diam)
            button.move(self.h_line_start + x * self.spacing - piece_radius, self.v_line_start + y * self.spacing - piece_radius)

    # always called after resizeEvent
    def paintEvent(self, event):
        painter = QPainter(self)

        pen = QPen()
        pen.setWidth(4)
        painter.setPen(pen)

        h_lines = [QLineF(self.h_line_start, y, self.h_line_end, y) for y in range(self.v_line_start, self.v_line_end + self.spacing, self.spacing)]
        v_lines = [QLineF(x, self.v_line_start, x, self.v_line_end) for x in range(self.h_line_start, self.h_line_end + self.spacing, self.spacing)]
        painter.drawLines(h_lines + v_lines)

        painter.end()

# Main window, contains the Board widget and the buttons widget
class GuiWindow(QWidget):
    def __init__(self):
        super(GuiWindow, self).__init__()
        self.init_ui(game)

    def init_ui(self, game):
        self.setWindowTitle(NAME)
        self.setWindowState(QtCore.Qt.WindowMaximized)

        self.button_layout = QVBoxLayout()
        self.undo_button = QPushButton("&Undo", self)
        self.button_layout.addWidget(self.undo_button)
        self.undo_button.setMinimumWidth(40)
        self.undo_button.setMaximumWidth(100)

        self.main_layout = QHBoxLayout(self)
        self.game_board = Board(game)
        self.main_layout.addWidget(self.game_board)
        # self.main_layout.addLayout(self.board_layout)
        self.main_layout.addLayout(self.button_layout)
        # self.layout.addWidget(self.undo_button)
        self.undo_button.clicked.connect(self.on_undo_button_click)
        self.show()

    # called when the undo button is clicked
    @pyqtSlot()
    def on_undo_button_click(self):
        num_moves_to_undo, is_int = QInputDialog.getInt(self, "integer input dualog", "enter a number")
        if is_int:
            self.game_board.signal_command.emit('u', min(num_moves_to_undo, len(self.game_board.game.moves)))

    def closeEvent(self, event):
        main_loop_thread.terminate()
        app.quit()
        event.accept()

from class_config import *

if __name__ == '__main__':
    players = [AI_REFLEX, AI_REFLEX_CACHED]
    players = [AI_REFLEX_CACHED, AI_REFLEX]
    players = [AI_REFLEX_CACHED, HUMAN_GUI]
    players = [AI_MINIMAX, HUMAN_GUI]
    players = [AI_REFLEX_CACHED, AI_MINIMAX]
    players = [AI_MINIMAX, AI_REFLEX_CACHED]
    players = [AI_MINIMAX, HUMAN_GUI]
    players = [HUMAN_GUI, HUMAN_GUI]
    display_type = DISPLAY_GUI

    game = Game(players, display_type)

    app = QtGui.QApplication(sys.argv)
    window = GuiWindow()

    # used to run the steps of the moves
    main_loop_thread = QThread()
    main_loop_thread.start()
    # game.display sends and receives signals from the GUI
    # also contains the logic of the main loop, run in main_loop_thread
    game.display.moveToThread(main_loop_thread)
    game.display.signal_start_main_loop.emit() # must be called after app is created else there's a race condition

    app.exec_()

