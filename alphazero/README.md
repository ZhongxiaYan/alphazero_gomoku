# AlphaZero

This directory contains a working, multiprocess PyTorch re-implementation of AlphaZero by DeepMind. I will only give a summary here, and you can read the code for details if you want to train your own models.

## Overview
This [cheatsheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0) provides a good overview of AlphaZero. This implementation utilizes python multiprocessing, consisting of 1 (preferably gpu) process for training, 1 (preferably gpu) process for evaluating board states for Monte Carlo Tree Search (MCTS), and N cpu processes for running MCTS. I ran with up to N = 60 processes for MCTS. 

## Results
Directly solving full Gomoku (connect 5 on 20x20 board) takes too long with just 2 gpus and N MCTS processes, so I experimented with curriculum learning --- first training a fully convolutional connect 5 model on 8x8 board, then further training the model on 12x12 board, then finally training the model on 20x20 board --- with some success.

I trained the initial convolutional model on 8x8 board for around 1 day with two Titan X Pascal GPUs and 60 threads, finetuned on 12x12 board for around 1 days, and finetuned on 20x20 board for around 1 days. The model seems pretty robust on 8x8 and decently robust on 12x12, but not as robust on 20x20 (might need more training).

## How to Run
The different training configurations are specified in results_*/config.json, for example results_8x8/config.json . I won't document the options here, but you can go through the code to see where the configurations are used. They are somewhat intuitively named (although not perfect).

### Training
Run train.py on the directory with the config.json . If you would like use Visdom for plotting, you can specify the Visdom env variables.

```VISDOM_SERVER=localhost VISDOM_PORT=8000 python3 train.py results_tic_tac_toe -dt cuda:0 -dv cuda:1```

### Simulation
Simulation plays a trained model against another trained model (perhaps a different architecture or a different epoch). simulate.py looks for simulation configs in the [simulation_configs/](https://github.com/ZhongxiaYan/alphazero_gomoku/tree/master/alphazero/simulation_configs) directory.

```python3 simulate.py 8x8_0 8x8_60000 -v cuda:0```

### Playing Against Yourself
You can play against yourself by changing the start of the alphazero_brain.py file for the config you want to play. alphazero_brain.py lets you play against Piskvork, see the [main README](https://github.com/ZhongxiaYan/alphazero_gomoku) for more details.
```python3 alphazero_brain.py --send port_to_windows --recv port_from_windows```

#### Playing from CLI
You may also enter command line commands if you use the --stdin options, but unfortunately I didn't try to display the pieces (although this can be done easily).

You would first run
```python3 alphazero_brain.py --stdin```

There's no prompt (sorry), but you may directly enter `START [Board_size]` and other [Piskvork commands](http://petr.lastovicka.sweb.cz/protocl2en.htm) to play.
