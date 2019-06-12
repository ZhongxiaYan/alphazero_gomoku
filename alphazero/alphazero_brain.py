from pathlib import Path
import sys
from tqdm import tqdm

import torch
import numpy as np

from model import Model
from mcts import MCTSNode
from util import Config, set_config, get_start_state, step_state

sys.path.append('../piskvork_remote')

import remote_brain
from remote_brain import Brain, main

config = Config('results_12x12').load().var(device='cuda:2')
set_config(config)
# config.eval_mcts_iterations = 0

model = Model(config).set_state(config.load_max_model_state(min_epoch=-1))
def evaluator(state):
	with torch.no_grad():
		v, p = model.fit_batch((np.array([state]),), train=False)
		return v, p[0]

class AlphaZero(Brain):
	def info_init(self):
		super().info_init()
		self.info_text = 'name="pbrain-alphazero", author="Zhongxia Yan", version="0.0", country="USA", www="https://github.com/ZhongxiaYan/gomoku_ai"'

	def set_mcts(self, state):
		self.head = MCTSNode(state, evaluator=evaluator)

	def brain_init(self):
		self.set_mcts(get_start_state())
		self.send('OK')

	def brain_restart(self):
		self.brain_init()

	def step_mcts(self, x, y):
		print('Moved %s,%s' % (x, y))
		move = (y, x)
		head = self.head
		if move in head.next:
			self.head = head.next[move]
		else:
			new_state = step_state(head.state, move)
			self.set_mcts(new_state)

	brain_my = brain_opponents = step_mcts

	def brain_takeback(self, x, y):
		state = self.head.state
		state[:, y, x] = 0
		self.set_mcts(state)
		return 0

	def brain_turn(self):
		if self.terminate_ai:
			return
		head = self.head
		if head.terminal:
			return
		if config.eval_mcts_iterations == 0:
			score = head.p
		else:
			for _ in tqdm(range(config.eval_mcts_iterations)):
				head.select()
			score = head.N
		(y, x) = move = np.unravel_index(score.argmax(), score.shape)

		np.set_printoptions(precision=2, linewidth=200)
		r = lambda x: x.reshape(config.board_dim, config.board_dim)
		# import q; q.d()

		self.do_mymove(x, y)

	def brain_end(self):
		sys.exit(0)

remote_brain.Brain = AlphaZero

if __name__ == '__main__':
	main()
