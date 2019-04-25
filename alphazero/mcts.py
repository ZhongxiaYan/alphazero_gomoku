import numpy as np
import util
from util import *

class MCTSNode:
    def __init__(self, state, value=0, evaluator=None):
        self.state = state
        if evaluator is None:
            self.value = value # either the terminal value
            self.terminal = True
        else:
            self.evaluator = evaluator
            self.value, self.P = evaluator(state)
            self.terminal = False
            self.N = np.zeros_like(self.P)
            self.W = np.zeros_like(self.P)
            self.mask = state.sum(axis=0).reshape(-1).astype(np.bool)
            self.score = self.P
            self.next = {}
            self.next_total = 0
    
    def select(self):
        if self.terminal:
            return self.value
        score = self.score
        score[self.mask] = -np.inf
        index = score.argmax()
        if index in self.next:
            new_node = self.next[index]
            value = -new_node.select()
        else:
            move = index_to_move(index)

            # flip the board
            this_state, opp_state = self.state.copy()
            this_state[move] = 1
            new_state = np.stack([opp_state, this_state])
            
            if check_win(this_state, move):
                new_node = MCTSNode(new_state, value=-1)
            elif new_state.sum() == config.board_dim ** 2:
                new_node = MCTSNode(new_state, value=0)
            else:
                new_node = MCTSNode(new_state, evaluator=self.evaluator)
            self.next[index] = new_node
            value = -new_node.value
        self.N[index] += 1
        self.W[index] += value
        self.next_total += 1
        self.score = np.nan_to_num(self.W / self.N) + config.c_puct * self.P * np.sqrt(self.next_total) / (1 + self.N) # UCB
        return value

# from time import time

class MCTS:
    def __init__(self, state, evaluator, config_):
        global config
        util.config = config = config_
        self.state = state
        self.evaluator = evaluator

    def run(self):
        # start = time()
        head = MCTSNode(self.state, evaluator=self.evaluator)
        states = []
        policies = []
        indices = []
        while not head.terminal:
            for _ in range(config.mcts_iterations):
                head.select()
            inv_temp = 1 / config.temp
            if len(states) > config.move_temp_decay:
                inv_temp = np.sqrt(len(states) - config.move_temp_decay)
            policy = head.N ** inv_temp
            policy /= policy.sum()
            index = np.random.choice(len(policy), p=policy)

            states.append(head.state)
            policies.append(policy)
            indices.append(index)

            head = head.next[index]
        value = head.value if len(states) % 2 == 1 else -head.value
        values = []
        for _ in states:
            values.append(value)
            value = -value
        # print('MCTS took %s' % (time() - start))
        return np.array(states), np.array(policies), np.array(values, dtype=np.float32), np.array(indices)
