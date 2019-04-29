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
            self.value, self.p = evaluator(state)
            self.terminal = False
            self.N = np.zeros_like(self.p)
            self.W = np.zeros_like(self.p)
            self.mask = state.sum(axis=0).reshape(-1).astype(np.bool)
            self.next = {}
            self.N_total = 1 # 1 initially makes things simpler to code
    
    def select(self):
        if self.terminal:
            return self.value
        P = (1 - config.mcts_eps) * self.p + \
            config.mcts_eps * np.random.dirichlet(config.mcts_alpha * np.ones_like(self.p))
        score = np.nan_to_num(self.W / self.N) + config.c_puct * P * np.sqrt(self.N_total) / (1 + self.N) # UCB
        score[self.mask] = -np.inf
        index = score.argmax()
        if index in self.next:
            new_node = self.next[index]
            value = -new_node.select()
        else:
            move = index_to_move(index)
            
            opp_state, this_state = new_state = step_state(self.state, move)
            
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
        self.N_total += 1
        return value

# from time import time

class MCTS:
    def __init__(self, state, evaluator):
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
                inv_temp *= np.sqrt(len(states) - config.move_temp_decay)
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
