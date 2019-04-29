import argparse
import multiprocessing
from multiprocessing import Manager, Process, Pool, Queue, Pipe

from expres.src.config import Config
from u import *

import util
from model import Model
from mcts import MCTS

parser = argparse.ArgumentParser(description='Run AlphaZero training')
parser.add_argument('res', type=Path, help='Result directory')
parser.add_argument('-dt', dest='device_t', type=str, help='Device for running model training')
parser.add_argument('-dv', dest='device_v', type=str, help='Device for running model evaluation')
parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='Debug state')

def eval_fn(config, p_train, q_from_mcts, ps_mcts):
    p_from_train, p_to_eval = p_train
    p_to_eval.close()
    
    ps_from_eval, ps_to_mcts = zip(*ps_mcts)
    [p_from_eval.close() for p_from_eval in ps_from_eval]

    state = p_from_train.recv()

    print('Started eval process %s' % os.getpid())
    sys.stdout.flush()
    config.device = config.device_v
    model = Model(config).set_net_state(state)
    model.network.eval()

    start = time()
    done = False
    with torch.no_grad():
        procs = []
        states = []
        while not done:
            proc_id, state = q_from_mcts.get()
            procs.append(proc_id)
            states.append(state)
            if len(procs) == config.pred_batch:
                pred_vs, pred_ps = model.fit_batch((np.array(states),), train=False)
                for proc_id, pred_v, pred_p in zip(procs, pred_vs, pred_ps):
                    ps_to_mcts[proc_id].send((pred_v, pred_p))
                procs = []
                states = []
                done = (time() - start) > config.time_reset_eval
            if p_from_train.poll():
                new_state = p_from_train.recv()
                model.set_net_state(new_state)
    print('Exiting eval process')
    exit()

def mcts_fn(config, q_to_train, q_to_eval, p_eval, process_id):
    import mcts
    import util
    mcts.config = util.config = config
    print('Starting MCTS process %s' % process_id)
    sys.stdout.flush()
    np.random.seed(process_id)
    
    p_from_eval, p_to_mcts = p_eval
    p_to_mcts.close()

    def eval_state(state):
        q_to_eval.put((process_id, state))
        return p_from_eval.recv()
    
    start_state = np.zeros((2, config.board_dim, config.board_dim), dtype=np.float32)
    mcts = MCTS(start_state, eval_state)
    while True:
        q_to_train.put(mcts.run())

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    print('Main process id %s' % os.getpid())

    args = parser.parse_args()
    util.config = config = Config(**vars(args)).load()
    if args.debug:
        config.num_mcts_processes = config.pred_batch = 2
        config.min_num_states = 2
        config.mcts_iterations = 100
        config.epoch_model_update = 5
        config.epoch_model_save = 1000

    config.device = config.device_t
    state = config.load_max_model_state(min_epoch=-1)
    model = Model(config)
    if state is None:
        state = model.get_state()
    else:
        model.set_state(state)

    # communication pipes and queues
    p_from_train, p_to_eval = p_eval = Pipe()
    q_mcts_to_eval = Queue()
    ps_eval_to_mcts = [Pipe() for _ in range(config.num_mcts_processes)]
    q_from_mcts = Queue()
    
    # mcts processes
    mcts_procs = []
    for i, p_mcts_eval in enumerate(ps_eval_to_mcts):
        proc = Process(target=mcts_fn, args=(config, q_from_mcts, q_mcts_to_eval, p_mcts_eval, i))
        proc.daemon = True
        proc.start()
        mcts_procs.append(proc)

    def start_eval():
        eval_proc = Process(target=eval_fn, args=(config, p_eval, q_mcts_to_eval, ps_eval_to_mcts))
        eval_proc.daemon = True
        eval_proc.start()
        state = model.get_net_state()
        print('Sending model state at epoch %s from train to eval' % state['epoch'])
        p_to_eval.send(state)
        return eval_proc

    model.set_watch_eval(start_eval)
    model.set_communication(q_from_mcts, p_to_eval)

    model.fit(config.train_epochs)
    
