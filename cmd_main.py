from config import *
from game import *

players = [HUMAN, HUMAN] # two players, both human
display = DISPLAY_COMMAND_LINE
game = Game(players, display)

# main loop
while True:
    game.show_board()
    game.transition()
    if game.has_ended():
        break

