from config import *
from game import *

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import numpy as np

players = [HUMAN, HUMAN] # two players, both human
display = DISPLAY_GUI

game = Game(players, display)

class Board(QWidget):
    def __init__(self):
        super(Board, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.buttons = [[QPushButton("", self) for x in range(BOARD_WIDTH)] for y in range(BOARD_HEIGHT)]

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
        button_style_string = """
            .QPushButton {
                background-color: black;
                border-radius: %spx;
                width: 50px;
                height: 50px;
                }
            .QPushButton:hover {
                background-color: blue;
                border-style: inset;
                }
            """ % (self.spacing / 2 - 0.1)
        for (y, x), button in np.ndenumerate(self.buttons):
            button.setStyleSheet(button_style_string)
            button.resize(self.spacing, self.spacing)
            button.move(self.h_line_start + (x - 0.5) * self.spacing, self.v_line_start + (y - 0.5) * self.spacing)

    # always called after resizeEvent
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.geometry(), QColor('#CC9900'))

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
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(NAME)
        self.setWindowState(QtCore.Qt.WindowMaximized)

        self.button_layout = QVBoxLayout()
        self.undo_button = QPushButton("&Undo", self)
        self.button_layout.addWidget(self.undo_button)
        self.undo_button.setMinimumWidth(40)
        self.undo_button.setMaximumWidth(100)

        self.main_layout = QHBoxLayout(self)
        self.main_layout.addWidget(Board())
        # self.main_layout.addLayout(self.board_layout)
        self.main_layout.addLayout(self.button_layout)
        # self.layout.addWidget(self.undo_button)
        self.show()

app = QtGui.QApplication(sys.argv)
window = GuiWindow()
# window.show()
sys.exit(app.exec_())