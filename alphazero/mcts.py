import numpy as np
from util import step_state, check_win

# global config is set by util.set_config

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
            self.mask = state[:2].sum(axis=0).astype(np.bool)
            self.next = {}
            self.N_total = 1 # 1 initially makes things simpler to code
    
    def select(self):
        if self.terminal:
            return self.value
        P = (1 - config.mcts_eps) * self.p + \
            config.mcts_eps * np.random.dirichlet([config.mcts_alpha] * self.p.size).reshape(self.p.shape)
        self.U = config.c_puct * P * np.sqrt(self.N_total) / (self.N + 1) # UCB
        self.Q = np.nan_to_num(self.W / self.N)
        self.score = score = self.Q + self.U
        score[self.mask] = -np.inf

        move = np.unravel_index(score.argmax(), score.shape)

        if move in self.next:
            value = -self.next[move].select()
        else:
            self.next[move] = self.step(move)
            value = -self.next[move].value
        self.N[move] += 1
        self.W[move] += value
        self.N_total += 1
        return value
    
    def step(self, move):
        if move in self.next:
            return self.next[move]
        else:
            opp_state, this_state, *_ = new_state = step_state(self.state, move)
            
            if check_win(this_state, move):
                return MCTSNode(new_state, value=-1)
            elif self.mask.sum() == config.board_dim ** 2 - 1:
                return MCTSNode(new_state, value=0)
            else:
                return MCTSNode(new_state, evaluator=self.evaluator)

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
        moves = []
        while not head.terminal:
            for _ in range(config.mcts_iterations):
                head.select()
            inv_temp = 1 / config.mcts_temp
            policy = head.N ** inv_temp
            policy /= policy.sum()
            index = np.random.choice(policy.size, p=policy.reshape(-1))
            move = np.unravel_index(index, policy.shape)

            states.append(head.state)
            policies.append(policy)
            moves.append(move)

            head = head.next[move]
        value = head.value if len(states) % 2 == 0 else -head.value
        values = []
        for _ in states:
            values.append(value)
            value = -value
        # print('MCTS took %s' % (time() - start))
        return np.array(states), np.array(policies), np.array(values, dtype=np.float32), np.array(moves)
