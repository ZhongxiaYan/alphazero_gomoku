NAME = 'Connect 5'

BOARD_WIDTH = 5
BOARD_HEIGHT = 5
LENGTH_NEEDED = 4
NUM_PLAYERS = 2

OFFSETS = [(1, 0), (1, 1), (0, 1), (-1, 1)] # horizontal, diagonal, vertical, diagonal
EMPTY_PIECE = None

def add(x, y):
    return tuple([x_elem + y_elem for x_elem, y_elem in zip(x, y)])

def sub(x, y):
    return tuple([x_elem - y_elem for x_elem, y_elem in zip(x, y)])
