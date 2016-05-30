from config import *
from game import *

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import numpy as np

class Board(QWidget):

    def __init__(self, game):
        super(Board, self).__init__()
        self.game = game
        self.init_ui()

    def generate_color(self, num_player, curr_player):
        rgb = int(curr_player * 0xFF / (num_player - 1))
        rgb_hex_str = '%X' % rgb
        return '#' + rgb_hex_str * 3

    def put_piece(self):
        button = self.sender()
        piece_diam = self.spacing * 0.9
        piece_radius = piece_diam / 2
        button_style_string = """
            .QPushButton {
                background-color: %s;
                border-radius: %spx;
            }
            .QPushButton:hover {
                background-color: blue;
            }
        """ % (self.generate_color(self.game.num_players, self.game.curr_player), piece_radius - 0.5)
        button.setStyleSheet(button_style_string)
        coord = (button.x_index, button.y_index)
        self.game.transition(coord)

    def init_ui(self):
        self.buttons = [[QPushButton("", self) for x in range(BOARD_WIDTH)] for y in range(BOARD_HEIGHT)]

        for (x, y), button in np.ndenumerate(self.buttons):
            button.x_index = x
            button.y_index = y
            button.clicked.connect(self.put_piece)

        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor('#CC9900'))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    # center the board to take up as much space as possible
    def calc_dimensions(self):
        geom = self.geometry()
        x_start = geom.x()
        y_start = geom.y()
        width = geom.width()
        height = geom.height()

        h_max_spacing = width // BOARD_WIDTH
        v_max_spacing = height // BOARD_HEIGHT
        self.spacing = min(h_max_spacing, v_max_spacing)

        w_board = self.spacing * BOARD_WIDTH
        h_board = self.spacing * BOARD_HEIGHT

        h_board_start = x_start + (width - w_board) // 2
        v_board_start = y_start + (height - h_board) // 2

        self.h_line_start = h_board_start + self.spacing // 2
        self.v_line_start = v_board_start + self.spacing // 2
        self.h_line_end = self.h_line_start + self.spacing * (BOARD_WIDTH - 1)
        self.v_line_end = self.v_line_start + self.spacing * (BOARD_HEIGHT - 1)

    # always called when generating / resizing a window. Before paintEvent
    def resizeEvent(self, event):
        self.calc_dimensions()
        piece_diam = self.spacing * 0.9
        piece_radius = piece_diam / 2
        button_style_string = """
            .QPushButton {
                border-radius: %spx;
            }
            .QPushButton:hover {
                background-color: blue;
            }
        """ % (piece_radius - 0.5)
        for (y, x), button in np.ndenumerate(self.buttons):
            button.setStyleSheet(button_style_string)
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
        self.main_layout.addWidget(Board(game))
        # self.main_layout.addLayout(self.board_layout)
        self.main_layout.addLayout(self.button_layout)
        # self.layout.addWidget(self.undo_button)
        self.show()

players = [HUMAN, HUMAN] # two players, both human
display_type = DISPLAY_GUI

game = Game(players, display_type)

# used to run the steps of the moves
main_loop_thread = QThread()
main_loop_thread.start()

# game.display sends and receives signals from the GUI
# also contains the logic of the main loop, run in main_loop_thread
game.display.moveToThread(main_loop_thread)
game.display.start_main_loop.emit()

app = QtGui.QApplication(sys.argv)
window = GuiWindow()
app.exec_()