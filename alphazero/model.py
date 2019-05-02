import queue
import numpy as np
import pandas as pd
import visdom

from torch.utils.data import Dataset, DataLoader

from expres.src.models.nn_model import NNModel
from expres.src.callbacks.result_monitor import ResultMonitor
from u import *

from network import ResNetGomoku
from util import *

class ReceiverDataset(Dataset):
    def __init__(self, config, model, q_from_mcts):
        self.c = config
        self.model = model
        self.q_from_mcts = q_from_mcts

        if (config.res / 'saved_games.npz').exists():
            games = np.load(config.res / 'saved_games.npz')
            self.states = games['states']
            self.policies = games['policies']
            self.values = games['values']
            print('Loaded %s cached game states' % len(self.states))
        else:
            self.states = np.zeros((0, 4, config.board_dim, config.board_dim), dtype=np.float32)
            self.policies = np.zeros((0, config.board_dim, config.board_dim), dtype=np.float32)
            self.values = np.zeros((0,), dtype=np.float32)
        self.last_game = None
        self.batches_left = 0

    def __len__(self):
        news = []
        num_new = 0
        if len(self.states) < self.c.min_num_states:
            while len(self.states) + num_new < self.c.min_num_states:
                try:
                    self.last_game = new = self.q_from_mcts.get(timeout=10.0)
                    news.append(new)
                    num_new += len(new[0])
                    print('Retrieved %s / %s states needed to start training' % (len(self.states) + num_new, self.c.min_num_states))
                except queue.Empty:
                    pass
        else:
            num_new = 0
            try:
                while True:
                    self.last_game = new = self.q_from_mcts.get_nowait()
                    news.append(new)
                    num_new += len(new[0])
            except queue.Empty:
                print('Retrieved %s new states from queue' % num_new)
            if num_new == 0:
                return len(self.states)
        new_states, new_policies, new_values, _ = zip(*news)
        max_mcts_queue = self.c.max_mcts_queue
        if max_mcts_queue is None:
            self.states = np.concatenate(new_states)
            self.policies = np.concatenate(new_policies)
            self.values = np.concatenate(new_values)
        else:
            self.states = np.concatenate((self.states[-(max_mcts_queue - num_new):], *new_states))
            self.policies = np.concatenate((self.policies[-(max_mcts_queue - num_new):], *new_policies))
            self.values = np.concatenate((self.values[-(max_mcts_queue - num_new):], *new_values))
            if self.model.epoch % self.c.epoch_save_games == 0:
                np.savez(self.c.res / 'saved_games', states=self.states, policies=self.policies, values=self.values)
                print('Saving %s cached game states' % len(self.states))
        return len(self.states)
    
    def __getitem__(self, idx):
        state, value, policy = self.states[idx], self.values[idx], self.policies[idx]
        k = np.random.randint(4)
        if k != 0:
            state = np.rot90(state, k=k, axes=(-2, -1))
            policy = np.rot90(policy, k=k, axes=(-2, -1))
        if np.random.randint(2):
            state = np.flip(state, axis=-1)
            policy = np.flip(policy, axis=-1)
        return np.ascontiguousarray(state), value, np.ascontiguousarray(policy)

class ModelCallback(ResultMonitor):
    def __init__(self, config, p_to_eval):
        self.config = self.c = config
        self.p_to_eval = p_to_eval
        self.train_results = None
        self.vis = visdom.Visdom(port=config.get('port', 8099))
        self.last_update_time = time()
    
    def on_epoch_end(self, model, train_state):
        self.put_train_result(model.epoch, train_state.epoch_result)
        self.c.save_train_results(self.train_results)
        if model.epoch % self.c.epoch_save_model == 0:
            save_path = self.save_model(model.epoch, model.get_state())
            print('Saved model at epoch %s to %s' % (model.epoch, save_path))
            psq_file = (self.c.res / 'sample_states').mk() / ('epoch-%07d.psq' % model.epoch)
            _, _, values, moves = model.data.last_game
            save_psq(psq_file, moves, values)

        since_last_update = time() - self.last_update_time
        if since_last_update > self.c.min_time_update_model and \
            (model.epoch % self.c.epoch_update_model == 0 or since_last_update > self.c.time_update_model):
            print('Sent model to eval process')
            model.p_to_eval.send(model.get_net_state())
            self.last_update_time = time()
        
    def save_model(self, epoch, state):
        path = self.c.save_model_state(epoch, state)
        if self.c.max_save > 0:
            self.c.clean_models(keep=self.c.max_save)
        return path
    
class Model(NNModel):
    def init_model(self):
        self.set_network(ResNetGomoku(self.c))
    
    def set_communication(self, q_from_mcts, p_to_eval):
        self.q_from_mcts = q_from_mcts
        self.p_to_eval = p_to_eval
    
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
        self.data = ReceiverDataset(self.c, self, self.q_from_mcts)
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