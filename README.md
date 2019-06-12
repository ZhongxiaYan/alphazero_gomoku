# AlphaZero Gomoku
Multiprocess implementation of [AlphaZero](https://deepmind.com/blog/alphazero-shedding-new-light-grand-games-chess-shogi-and-go/) applied to the board game Gomoku (五子棋; 5 in a row; tic-tac-toe but with 5 pieces on a 20x20 board).

## Alphazero
[alphazero](https://github.com/ZhongxiaYan/alphazero_gomoku/tree/master/alphazero) directory contains code for training, code for simulating games between trained agents, and code to interface with Piskvork.

## Piskvork
[piskvork_remote](https://github.com/ZhongxiaYan/piskvork_remote/tree/ca9947b98209b57e28210d2ed4bb3f3cff7d568a) directory contains more code to interface with Piskvork.

Note that this is an optional interface, it may be easier to write your own CLI visualizer (especially if you don't have a Windows computer).

[Piskvork](https://gomocup.org/download-gomocup-manager/) is a Windows-based GUI client for interacting with Gomoku agents and is used by Gomocup AI bots; basically it has a game board that you can play on and also watch AI agents play each other.

Since Piskvork is Windows-based, I wrote [piskvork_remote/pisqpipe.py](https://github.com/ZhongxiaYan/piskvork_remote/blob/ca9947b98209b57e28210d2ed4bb3f3cff7d568a/pisqpipe.py), which compiles to piskvork_remote/dist/pbrain-port.exe, to relay commands to / from a Linux machine. On the Linux side, I wrote [piskvork_remote/remote_brain.py](https://github.com/ZhongxiaYan/piskvork_remote/blob/ca9947b98209b57e28210d2ed4bb3f3cff7d568a/remote_brain.py) to listen for the messages from piskpipe.py; piskvork/remote_brain.py is overriden by [alphazero/alphazero_brain.py](https://github.com/ZhongxiaYan/alphazero_gomoku/blob/master/alphazero/alphazero_brain.py) so that trained models can be used to respond to moves.

If you want to work with Piskvork, the [documentations](http://petr.lastovicka.sweb.cz/protocl2en.htm) may be helpful.
