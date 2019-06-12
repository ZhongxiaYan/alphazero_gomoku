import os, re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

def set_config(conf):
    import mcts
    global config
    mcts.config = config = conf

# gomoku related functions
def get_start_state():
    start_state = np.zeros((config.state_size, config.board_dim, config.board_dim), dtype=np.float32)
    start_state[-1] = 1
    return start_state

def index_to_move(index):
    return index // config.board_dim, index % config.board_dim

def move_to_index(move):
    y, x = move
    return y * config.board_dim + x

def step_state(state, move):
    # move and then flip the board
    if config.state_size == 5:
        this_state, opp_state, last_move_2, _, this_first = state.copy()
    else:
        this_state, opp_state, _, this_first = state.copy()
    last_move = np.zeros_like(this_state)
    last_move[move] = 1
    this_state[move] = 1
    if config.state_size == 5:
        return np.stack([opp_state, this_state, last_move, last_move_2, 1 - this_first])
    else:
        return np.stack([opp_state, this_state, last_move, 1 - this_first])

def check_win(player_board, move):
    n_w = config.n_win
    win = np.ones(n_w)
    y, x = move
    start_y = max(y - n_w + 1, 0)
    start_x = max(x - n_w + 1, 0)
    roi = player_board[start_y: y + n_w, start_x: x + n_w]
    
    y -= start_y
    x -= start_x
    diag_k = x - y
    x_flip = roi.shape[1] - x - 1
    diag_k_flip = x_flip - y
    return any((
        (np.convolve(roi[y], win) >= n_w).any(),
        (np.convolve(roi[:, x], win) >= n_w).any(),
        (np.convolve(np.diag(roi, k=diag_k), win) >= n_w).any(),
        (np.convolve(np.diag(np.fliplr(roi), k=diag_k_flip), win) >= n_w).any()
    ))

def save_psq(file, moves, values):
    move_lines = ['%s,%s,0' % (y + 1, x + 1) for y, x in moves]
    lines = ['Piskvorky 20x20, 11:11, 0'] + move_lines
    lines.extend(['AlphaZero 1', 'AlphaZero 2', '-1', '%s,Freestyle' % (1 if values[0] == 1 else 2)])
    save_text(file, '\n'.join(lines))


# compute related functions
from collections import OrderedDict
def recurse(x, fn):
    T = type(x)
    if T in [dict, OrderedDict]:
        return T((k, recurse(v, fn)) for k, v in x.items())
    elif T in [list, tuple]:
        return T(recurse(v, fn) for v in x)
    return fn(x)

def from_numpy(x):
    def helper(x):
        if type(x).__module__ == np.__name__:
            if type(x) == np.ndarray:
                return recurse(list(x), helper)
            return np.asscalar(x)
        return x
    return recurse(x, helper)

import torch
def to_torch(x, device='cuda'):
    def helper(x):
        if x is None:
            return None
        elif type(x) == torch.Tensor:
            return x.to(device)
        elif type(x) in [str, bool, int, float]:
            return x
        return torch.from_numpy(x).to(device)
    return recurse(x, helper)

def from_torch(t):
    def helper(t):
        x = t.detach().cpu().numpy()
        if x.size == 1 or np.isscalar(x):
            return np.asscalar(x)
        return x
    return recurse(t, helper)


# file / display related functions
import visdom

class Visdom(visdom.Visdom):
    def line(self, Y, X=None, win=None, env=None, opts={}, update='append', name=None):
        all_opts = dict(title=win, showlegend=True)
        all_opts.update(opts)
        if update == 'remove':
            all_opts = None
        super(Visdom, self).line(Y=Y, X=X, win=win, env=env, opts=all_opts, update=update, name=name)

_visdom_cache = {}
def get_visdom(env='main', server=None, port=None, raise_exceptions=True, **kwargs):
    server = server or os.environ['VISDOM_SERVER']
    port = port or os.environ['VISDOM_PORT']
    key = (server, port, env or 'main')
    if key not in _visdom_cache:
        _visdom_cache[key] = Visdom(server=server, port=port, env=env, raise_exceptions=raise_exceptions, **kwargs)
    return _visdom_cache[key]

import json
def load_json(path):
    with open(str(path), 'r+') as f:
        return json.load(f)

def save_json(path, dict_):
    with open(str(path), 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)

def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)

def save_text(path, string):
    with open(str(path), 'w+') as f:
        f.write(string)

class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def var(self, *args, **kwargs):
        for a in args:
            kwargs[a] = True
        self.__dict__.update(kwargs)
        return self
    
    def unvar(self, *args):
        for a in args:
            self.__dict__.pop(a)
        return self
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)

class Config(Namespace):
    excludes = ['res', 'name', 'train_epoch', 'device', 'debug']

    def __init__(self, res, **kwargs):
        self.res = Path(res)
        self.name = kwargs.get('name', self.res.resolve().name)
        super(Config, self).__init__(**kwargs)
    
    def __repr__(self):
        return format_json(from_numpy(vars(self)))

    def __hash__(self):
        return hash(repr(self))

    @property
    def path(self):
        return self.res / (type(self).__name__.lower() + '.json')
    
    def load(self):
        if self.path.exists():
            for k, v in load_json(self.path).items():
                if k in ['model', 'data']:
                    v = Path(v)
                setattr(self, k, v)
        return self

    def save(self, force=False):
        if force or not self.path.exists():
            self.res.mkdir(exist_ok=True)
            save_json(self.path, from_numpy({k: v for k, v in vars(self).items() if self.should_save(k)}))
        return self
    
    def should_save(self, k):
        return k not in self.excludes

    @property
    def train_results(self):
        return self.res / 'train_results.csv'
    
    def load_train_results(self):
        if self.train_results.exists():
            return pd.read_csv(self.train_results, index_col=0)
        return None

    def save_train_results(self, results):
        results.to_csv(self.train_results, float_format='%.6g')


    @property
    def stopped_early(self):
        return self.res / 'stopped_early'
    
    def set_stopped_early(self):
        self.stopped_early.save_txt('')

    
    @property
    def models(self):
        path = self.res / 'models'
        path.mkdir(exist_ok=True)
        return path
    
    def model_save(self, epoch):
        return self.models / ('model-%s.pth' % epoch)

    def get_saved_model_epochs(self):
        save_paths = [x for x in self.models.iterdir() if x.is_file()]
        if len(save_paths) == 0:
            return []
        match_epoch = lambda path: re.match('.+/model-(\d+)\.pth', str(path))
        return sorted([int(m.groups()[0]) for m in (match_epoch(p) for p in save_paths) if m is not None])

    def clean_models(self, keep=5):
        model_epochs = self.get_saved_model_epochs()
        delete = len(model_epochs) - keep
        keep_paths = [self.model_best._real, self.model_save(model_epochs[-1])._real]
        for e in model_epochs:
            if delete <= 0:
                break
            path = self.model_save(e)._real
            if path in keep_paths:
                continue
            path.rm()
            delete -= 1
            print('Removed model %s' % path)

    def load_model_state(self, epoch=None, path=None):
        if epoch is not None:
            path = self.model_save(epoch)
        save_path = Path(path)
        if save_path.exists():
            return to_torch(torch.load(str(save_path)), device=self.device)
        return None

    def save_model_state(self, epoch, state, clean=True):
        save_path = self.model_save(epoch)
        torch.save(state, str(save_path))
        print('Saved model %s at epoch %s' % (save_path, epoch))
        if clean and self.get('max_save'):
            self.clean_models(keep=self.max_save)
        return save_path

    def load_max_model_state(self, min_epoch=0):
        epochs = self.get_saved_model_epochs()
        if len(epochs) == 0:
            print('No saved model found in %s' % self.models)
            return None
        epoch = max(epochs)
        if epoch <= min_epoch:
            print('Model is already at epoch %s, no need to load' % min_epoch)
            return None
        return self.load_model_state(epoch=epoch)

import enlighten

progress_manager = enlighten.get_manager()
active_counters = []

class Progress(object):

    def __init__(self, total, desc='', leave=False):
        self.counter = progress_manager.counter(total=total, desc=desc, leave=leave)
        active_counters.append(self.counter)

    def __iter__(self):
        return self
    
    def __next__(self):
        raise NotImplementedError()
    
    def close(self):
        self.counter.close()
        active_counters.remove(self.counter)
        if len(active_counters) == 0:
            progress_manager.stop()

    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

class RangeProgress(Progress):
    def __init__(self, start, end, step=1, desc='', leave=False):
        self.i = start
        self.start = start
        self.end = end
        self.step = step
        super(RangeProgress, self).__init__((end - start) // step, desc=desc, leave=leave)
    
    def __next__(self):
        if self.i != self.start:
            self.counter.update()
        if self.i == self.end:
            self.close()
            raise StopIteration()
        i = self.i
        self.i += self.step
        return i