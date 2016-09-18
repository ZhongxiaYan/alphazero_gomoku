NAME = 'Connect 5'

BOARD_WIDTH = 15
BOARD_HEIGHT = 15
LENGTH_NEEDED = 5
NUM_PLAYERS = 2

OFFSETS = [(1, 0), (1, 1), (0, 1), (-1, 1)] # horizontal, diagonal, vertical, diagonal
EMPTY_PIECE = None

def add(x, y):
    return tuple([x_elem + y_elem for x_elem, y_elem in zip(x, y)])

def sub(x, y):
    return tuple([x_elem - y_elem for x_elem, y_elem in zip(x, y)])
