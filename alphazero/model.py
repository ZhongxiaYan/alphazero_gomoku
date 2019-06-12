import queue
from time import time
from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from network import ResNetGomoku, ConvNetGomoku, ConvNetLargeGomoku, FullyConvNetGomoku
from util import Config, RangeProgress, to_torch, from_torch, get_visdom, save_psq

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

class Model:
    def __init__(self, config):
        self.config = self.c = config
    
        self.epoch = 0
    
        net_cls = eval(self.c.get('net', 'ResNetGomoku'))
        network = net_cls(self.c)
        
        self.optimizer = self.opt = network.optimizer
        network.to(config.device)
        self.network = self.net = network
    
    def set_communication(self, q_mcts_to_train, p_te_send):
        self.q_mcts_to_train = q_mcts_to_train
        self.p_te_send = p_te_send

    def fit(self, stop_epoch):
        self.net.train()

        self.stop_epoch = stop_epoch
        self.on_train_start()

        train_gen = DataLoader(self.data, batch_size=self.c.train_batch, shuffle=True)

        while self.epoch < self.stop_epoch:
            start = time()
            t_results = pd.DataFrame([self.fit_batch(xy) for xy in train_gen])

            self.epoch += 1
            epoch_result = t_results.mean(axis=0)
            epoch_result['execution_time'] = round(time() - start, 5)
            
            self.on_epoch_end(epoch_result)
        self.on_train_end()
        return self

    def fit_batch(self, xy, train=True):
        loss, pred = self.network(*to_torch(xy, device=self.c.device))
        pred = from_torch(pred)
        if not train:
            return pred['value'], pred['policy']
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        pred['num_data'] = len(self.data.states)
        return pred

    def get_state(self):
        return to_torch(dict(epoch=self.epoch,
                             network=self.network.state_dict(),
                             optimizer=self.optimizer.state_dict()),
                        device='cpu')

    def set_state(self, state):
        if state is None: return self
        self.net.load_state_dict(state['network'])
        if 'optimizer' in state:
            self.opt.load_state_dict(state['optimizer'])
        else:
            print('Loaded state contains no optimizer')
        self.epoch = state['epoch']
        return self
    
    def get_net_state(self):
        return to_torch(dict(epoch=self.epoch,
                             network=self.network.state_dict()),
                        device='cpu')

    def set_net_state(self, state):
        self.network.load_state_dict(state['network'])
        self.epoch = state['epoch']
        return self
    
    def on_train_start(self):
        config = self.c
        epoch = self.epoch
        stop_epoch = self.stop_epoch
        
        self.train_results = train_results = config.load_train_results()

        self.vis = None
        if config.get('use_visdom', True):
            try:
                env = config.get('env', 'main')
                self.vis = get_visdom(env=env)
                print('Plotting to visdom env %s' % env)
            except ConnectionError:
                print('Visdom failed to connect, not recording to Visdom')

        # Load previous results
        if train_results is not None and len(train_results):
            # Plot previous results
            if self.vis:
                for key, column in train_results.iteritems():
                    self.vis.line(win=key, X=column.index, Y=column, update='replace', name=config.name)
            
            # Stop if previous results exceed stop epoch
            if train_results.index[-1] >= stop_epoch:
                print('Preempting training because already trained past %s epochs' % stop_epoch)
                self.stop_epoch = epoch
                return
        else:
            print('No previous history to plot')
        
        # Early stopped before
        if config.stopped_early.exists():
            print('Preempting training because already stopped')
            self.stop_epoch = epoch
            return
        
        self.set_state(config.load_max_model_state(min_epoch=epoch or -1))
        self.last_save_epoch = epoch = self.epoch
        self.last_save_time = time()
        self.last_update_time = time()

        self.save_after_epochs = config.get('save_after_epochs', np.inf)
        self.save_after_time = config.get('save_after_time', np.inf)
    
        self.last_game = None
        self.batches_left = self.c.get('max_batch_wo_new_data', np.inf)
        
        # Print progress
        print('Training %s from epoch %s to epoch %s' % (config.name, epoch, stop_epoch))
        self.prog = iter(RangeProgress(epoch, stop_epoch, desc=config.name))

        self.data = Data(self.c)
        self.fetch_data()
    
    def on_epoch_end(self, epoch_result):
        config = self.c
        epoch = self.epoch
        train_results = self.train_results

        vis_update = 'append'
        if train_results is None:
            self.train_results = train_results = pd.DataFrame(columns=epoch_result.index, index=pd.Series(name='epoch'))
            vis_update = 'replace'
        train_results.loc[epoch] = epoch_result
        
        if self.vis:
            for key, value in epoch_result.iteritems():
                self.vis.line(win=key, X=[epoch], Y=[value], update=vis_update, name=config.name)

        if any((
            epoch - self.last_save_epoch >= self.save_after_epochs,
            time() - self.last_save_time >= self.save_after_time
        )):
            config.save_train_results(train_results)
            config.save_model_state(epoch, self.get_state())
            self.last_save_epoch = epoch
            self.last_save_time = time()

        if self.last_game and epoch % config.epoch_save_sample_game == 0:
            dir_path = config.res / 'sample_states'
            dir_path.mkdir(exist_ok=True)
            psq_path = dir_path / ('epoch-%07d.psq' % epoch)
            values, moves = self.last_game
            save_psq(psq_path, moves, values)
            self.last_game = None

        since_last_update = time() - self.last_update_time
        if since_last_update > config.min_time_update_model and \
            (epoch % config.epoch_update_model == 0 or since_last_update > config.time_update_model):
            print('Sent model to eval process')
            self.p_te_send.send(self.get_net_state())
            self.last_update_time = time()
        
        print('Epoch %s:\n%s\n' % (epoch, epoch_result.to_string(header=False)))
        next(self.prog)
        
        self.fetch_data()
    
    def on_train_end(self):
        config = self.c
        epoch = self.epoch
        
        # Save train results
        if self.train_results is not None:
            config.save_train_results(self.train_results)

        # Save latest model
        if epoch > 0 and not config.model_save(epoch).exists():
            save_path = config.save_model_state(epoch, self.get_state())
            print('Saved model at epoch %s to %s' % (epoch, save_path))
    
        self.prog.close()
    
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
                    self.batches_left = config.get('max_batch_wo_new_data', np.inf)
            except queue.Empty:
                print('Retrieved %s new states from queue' % num_new)
            if num_new == 0:
                self.batches_left -= 1
                return

        new_states, new_policies, new_values = map(np.float32, map(np.concatenate, zip(*news)))
        self.data.update(new_states, new_policies, new_values,
            self.epoch % config.epoch_save_games == 0)
        