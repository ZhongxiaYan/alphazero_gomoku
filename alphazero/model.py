import queue
import numpy as np
import pandas as pd
import visdom

from torch.utils.data import Dataset, DataLoader

from expres.src.models.nn_model import NNModel
from expres.src.callbacks import Callback
from expres.src.callbacks.result_monitor import ResultMonitor
from u import from_torch, to_torch

from network import ResNetGomoku
from util import *

class ReceiverDataset(Dataset):
    def __init__(self, config, q_from_mcts):
        self.c = config
        self.q_from_mcts = q_from_mcts
        self.states = np.zeros((0, 2, config.board_dim, config.board_dim), dtype=np.float32)
        self.policies = np.zeros((0, config.board_dim ** 2), dtype=np.float32)
        self.values = np.zeros((0,), dtype=np.float32)
        self.last_game = None

    def __len__(self):
        news = []
        def append_rots(new):
            new_states, new_policies, new_values, _ = new
            new_policies = new_policies.reshape(-1, self.c.board_dim, self.c.board_dim)
            for k in range(4):
                rot_states = np.rot90(new_states, k=k, axes=(-2, -1))
                rot_policies = np.rot90(new_policies, k=k, axes=(-2, -1))
                flip_states = np.flip(rot_states, axis=-1)
                flip_policies = np.flip(rot_policies, axis=-1)
                news.append((rot_states, rot_policies.reshape(-1, self.c.board_dim ** 2), new_values))
                news.append((flip_states, flip_policies.reshape(-1, self.c.board_dim ** 2), new_values))
            return len(new_states) * 8
        try:
            num_new = 0
            while True:
                self.last_game = new = self.q_from_mcts.get_nowait()
                num_new += append_rots(new)
        except queue.Empty:
            while num_new + len(self.states) < self.c.min_num_states:
                self.last_game = new = self.q_from_mcts.get()
                num_new += append_rots(new)
            print('Retrieved %s new states from queue' % num_new)
        if num_new == 0:
            return len(self.states)
        new_states, new_policies, new_values = zip(*news)

        max_mcts_queue = self.c.max_mcts_queue
        self.states = np.concatenate((self.states[-(max_mcts_queue - num_new):], *new_states))
        self.policies = np.concatenate((self.policies[-(max_mcts_queue - num_new):], *new_policies))
        self.values = np.concatenate((self.values[-(max_mcts_queue - num_new):], *new_values))
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.values[idx], self.policies[idx]

class ModelCallback(Callback):
    def __init__(self, config, p_to_eval):
        super().__init__(config)
        self.p_to_eval = p_to_eval
        self.train_results = None
        self.vis = visdom.Visdom(port=config.get('port', 8099))
    
    def on_train_start(self, model, train_state):
        self.train_results = self.config.load_train_results()
        if self.train_results is not None:
            for key, column in self.train_results.iteritems():
                self.plot_line(key, column.index, column, 'replace')
    
    def on_epoch_end(self, model, train_state):
        self.put_train_result(model.epoch, train_state.epoch_result)
        self.config.save_train_results(self.train_results)
        if model.epoch % self.config.epoch_model_save == 0:
            save_path = self.save_model(model.epoch, model.get_state())
            print('Saved model at epoch %s to %s' % (model.epoch, save_path))
            psq_file = (self.config.res / 'sample_states').mk() / ('epoch-%07d.psq' % model.epoch)
            _, _, values, indices = model.data.last_game
            save_psq(psq_file, indices, values)

        if not model.eval_proc.is_alive():
            print('Resetting eval process')
            model.eval_proc = model.start_eval()
        elif model.epoch % self.config.epoch_model_update == 0:
            model.p_to_eval.send(model.get_net_state())

    def save_model(self, epoch, state):
        path = self.config.save_model_state(epoch, state)
        if self.config.max_save > 0:
            self.config.clean_models(keep=self.config.max_save)
        return path
    
    def put_train_result(self, epoch, result):
        if self.train_results is None:
            self.train_results = pd.DataFrame([result], index=pd.Series([epoch], name='epoch'))
            for key, column in self.train_results.iteritems():
                self.plot_line(key, column.index, column, 'replace')
        else:
            self.train_results.loc[epoch] = result
            for key, value in result.iteritems():
                self.plot_line(key, [epoch], [value], 'append')
    
    def plot_line(self, key, X, Y, update):
        self.vis.line(X=X, Y=Y, win=key, update=update, name=self.config.name, opts=dict(title=key, showlegend=True))
    
class Model(NNModel):
    def init_model(self):
        self.set_network(ResNetGomoku(self.c))
    
    def set_communication(self, q_from_mcts, p_to_eval):
        self.q_from_mcts = q_from_mcts
        self.p_to_eval = p_to_eval
    
    def set_watch_eval(self, start_eval):
        self.start_eval = start_eval
        self.eval_proc = start_eval()
    
    def get_callbacks(self):
        get_model_callback = lambda config: ModelCallback(config, self.p_to_eval)
        return [get_model_callback]
        
    def fit_batch(self, xy, train=True):
        loss, pred = self.network(*to_torch(xy, device=self.config.device))
        pred = from_torch(pred)
        if not train:
            return pred['value'], pred['policy']
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        pred['num_data'] = len(self.data.states)
        return pred
    
    def get_train_data(self):
        self.data = ReceiverDataset(self.c, self.q_from_mcts)
        return DataLoader(self.data, batch_size=self.c.train_batch, shuffle=True)

    def get_val_data(self):
        return None
    
    def get_net_state(self):
        return to_torch(dict(epoch=self.epoch,
                             network=self.network.state_dict()),
                        device='cpu')

    def set_net_state(self, state):
        self.network.load_state_dict(state['network'])
        self.epoch = state['epoch']
        return self