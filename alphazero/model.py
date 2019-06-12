import queue
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from network import ResNetGomoku, ConvNetGomoku, ConvNetLargeGomoku, FullyConvNetGomoku
from util import Config

class Data(Dataset):
    def __init__(self, config):
        self.c = config
        if (config.res / 'saved_games.npz').exists():
            games = np.load(config.res / 'saved_games.npz')
            self.states = games['states']
            self.policies = games['policies']
            self.values = games['values']
            print('Loaded %s cached game states' % len(self.states))
        else:
            self.states = np.zeros((0, config.state_size, config.board_dim, config.board_dim), dtype=np.float32)
            self.policies = np.zeros((0, config.board_dim, config.board_dim), dtype=np.float32)
            self.values = np.zeros((0,), dtype=np.float32)

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.values[idx], self.policies[idx]

    def update(self, new_states, new_policies, new_values, save=False):
        config = self.c
        if config.max_mcts_queue is None: # discard previous data
            self.states = new_states
            self.policies = new_policies
            self.values = new_values
        else:
            num_keep = config.max_mcts_queue - len(new_states)
            self.states = np.concatenate((self.states[-num_keep:], new_states))
            self.policies = np.concatenate((self.policies[-num_keep:], new_policies))
            self.values = np.concatenate((self.values[-num_keep:], new_values))
        if save:
            np.savez(config.res / 'saved_games', states=self.states, policies=self.policies, values=self.values)
            print('Saving %s cached game states' % len(self.states))

class Model(NNModel):
    def init_model(self):
        net_cls = eval(self.c.get('net', 'ResNetGomoku'))
        self.set_network(net_cls(self.c))
    
    def set_communication(self, q_mcts_to_train, p_te_send):
        self.q_mcts_to_train = q_mcts_to_train
        self.p_te_send = p_te_send

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
        self.data = ReceiverDataset(self.c, self, self.q_mcts_to_train)
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
    
    def on_train_start(self):
        config.use_reward_save = False
        config.use_periodic_save = True
        super(Model, self).on_train_start()
        self.last_update_time = time()
    
        self.last_game = None
        self.batches_left = self.c.get('max_batch_wo_new_data')

        self.fetch_data()
    
    def on_epoch_end(self, epoch_result):
        super(Model, self).on_epoch_end()
        config = self.c
        epoch = self.epoch
        
        if epoch % config.epoch_save_sample_game == 0:
            psq_path = (config.res / 'sample_states').mk() / ('epoch-%07d.psq' % epoch)
            values, moves = self.last_game
            save_psq(psq_path, moves, values)

        since_last_update = time() - self.last_update_time
        if since_last_update > config.min_time_update_model and \
            (epoch % config.epoch_update_model == 0 or since_last_update > config.time_update_model):
            print('Sent model to eval process')
            self.p_te_send.send(self.get_net_state())
            self.last_update_time = time()
        
        self.fetch_data()
    
    def fetch_data(self):
        config = self.c
        num_prev = len(self.data)

        news = []
        num_new = 0

        def append_new(new):
            nonlocal num_new
            states, policies, values, moves = new
            self.last_game = (values, moves)

            for k in range(4):
                s, p = states, policies
                if k != 0:
                    s = np.rot90(s, k=k, axes=(-2, -1))
                    p = np.rot90(p, k=k, axes=(-2, -1))
                news.append((s, p, values))
                s = np.flip(s, axis=-1)
                p = np.flip(p, axis=-1)
                news.append((s, p, values))
            num_new += len(states) * 8

        if num_prev < config.min_num_states:
            while num_prev + num_new < config.min_num_states:
                try:
                    append_new(self.q_mcts_to_train.get(timeout=10.0))
                    print('Retrieved %s / %s states needed to start training' % (num_prev + num_new, config.min_num_states))
                except queue.Empty:
                    pass
        else:
            try:
                while True:
                    block = (self.batches_left == 0)
                    append_new(self.q_mcts_to_train.get(block))
                    self.batches_left = config.get('max_batch_wo_new_data')
            except queue.Empty:
                print('Retrieved %s new states from queue' % num_new)
            if num_new == 0:
                self.batches_left -= 1
                return

        new_states, new_policies, new_values = map(np.float32, map(np.concatenate, zip(*news)))
        self.data.update(new_states, new_policies, new_values,
            self.epoch % config.epoch_save_games == 0)
        