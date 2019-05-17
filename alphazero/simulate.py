import argparse

import numpy as np
import util

from e.src.config import Config
from model import Model
import mcts
from mcts import MCTSNode
from util import *
from u import *
from u.progress_bar import RangeProgress

proj = Path('/data/scratch/zxyan/go/gomoku_ai/alphazero')
configs = proj / 'simulation_configs'

parser = argparse.ArgumentParser(description='Simulating play between two models')
parser.add_argument('config1', type=str, help='Config 1 name')
parser.add_argument('config2', type=str, help='Config 2 name')
parser.add_argument('--N', type=int, default=20, help='Number of games to play')
parser.add_argument('-v', dest='device', type=str, help='Device for running model evaluation')

def play_game(model_first, model_second):
    config_first, config_second = model_first.config, model_second.config

    def get_eval_fn(model):
        def eval_fn(state):
            with torch.no_grad():
                v, p = model.fit_batch((np.array([state]),), train=False)
                return v, p[0]
        return eval_fn
    eval_first, eval_second = map(get_eval_fn, [model_first, model_second])
    
    start_state = get_start_state(config_first)
    curr = MCTSNode(start_state, evaluator=eval_first)
    next = MCTSNode(start_state, evaluator=eval_second)
    mcts.config, next_config = config_first, config_second

    info = []
    for _ in RangeProgress(0, config_first.board_dim ** 2, desc='Moves'):
        util.config = mcts.config

        start = time()
        if mcts.config.eval_mcts_iterations == 0:
            score = curr.p
        else:
            for _ in RangeProgress(0, mcts.config.eval_mcts_iterations, desc='MCTS'):
                curr.select()
            score = curr.N
        move = np.unravel_index(score.argmax(), score.shape)

        info.append(dict(
            state=curr.state,
            curr_p=curr.p,
            curr_v=curr.value,
            curr_W=curr.W,
            curr_N=curr.N,
            next_p=next.p,
            next_v=next.value,
            move=move,
            time=time() - start
        ))

        next, curr = curr.step(move), next.step(move)
        mcts.config, next_config = next_config, mcts.config
        if curr.terminal:
            break

    merged_info = {k: [info_i[k] for info_i in info] for k in info[0].keys()}
    merged_info['state'].append(curr.state)
    merged_info['curr_v'].append(-1)
    merged_info['next_v'].append(-1)
    return {k: np.array(v) for k, v in merged_info.items()}

def get_save_dir(config1, config2):
    name = '-'.join(sorted([config1.name, config2.name]))
    return (proj / 'simulations' / name).mk()

if __name__ == '__main__':
    args = parser.parse_args()
    
    config1 = Config(**(configs / (args.config1 + '.json')).load()).var(device=args.device)
    config2 = Config(**(configs / (args.config2 + '.json')).load()).var(device=args.device)

    model1 = Model(config1).set_state(config1.load_model_state(config1.epoch))
    model2 = Model(config2).set_state(config2.load_model_state(config2.epoch))

    save_dir = get_save_dir(config1, config2)
    
    for i in RangeProgress(0, args.N, desc='Games'):
        print('Simulating game %s' % i)
        game = play_game(model1, model2)
        np.save(save_dir / '%04d.npy' % i, dict(config1=model1.config.name, config2=model2.config.name, game=game))
        num_moves = len(game['move'])
        print('%s won in %s moves' % ((model1 if num_moves % 2 != 0 else model2).config.name, num_moves))
        print()
        model1, model2 = model2, model1
