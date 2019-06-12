import argparse
import multiprocessing
from multiprocessing import Manager, Process, Pool, Queue, Pipe

from u import Config, Path

from util import set_config
from model import Model
from mcts import MCTS

parser = argparse.ArgumentParser(description='Run AlphaZero training')
parser.add_argument('res', type=Path, help='Result directory')
parser.add_argument('-dt', dest='device_t', type=str, help='Device for running model training')
parser.add_argument('-dv', dest='device_v', type=str, help='Device for running model evaluation')
parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='Debug state')

# runs on the evaluation (gpu) process
def eval_fn(config, p_train_to_eval, q_mcts_to_eval, ps_eval_to_mcts):
    p_te_recv, p_te_send = p_train_to_eval
    p_te_send.close()
    
    ps_em_recv, ps_em_send = zip(*ps_eval_to_mcts)
    [p_em_recv.close() for p_em_recv in ps_em_recv]

    state = p_te_recv.recv()

    print('Started eval process %s' % os.getpid())
    sys.stdout.flush()
    config.device = config.device_v
    model = Model(config).set_net_state(state)
    model.network.eval()

    with torch.no_grad():
        procs = []
        states = []
        while True:
            proc_id, state = q_mcts_to_eval.get()
            procs.append(proc_id)
            states.append(state)
            if len(procs) == config.pred_batch:
                pred_vs, pred_ps = model.fit_batch((np.array(states, dtype=np.float32),), train=False)
                for proc_id, pred_v, pred_p in zip(procs, pred_vs, pred_ps):
                    ps_em_send[proc_id].send((pred_v[0], pred_p.tolist()))
                procs = []
                states = []
            if p_te_recv.poll():
                print('Received model from train process')
                new_state = p_te_recv.recv()
                model.set_net_state(new_state)

# runs on each mcts (cpu) process
def mcts_fn(config, q_mcts_to_train, q_mcts_to_eval, p_eval_to_mcts, process_id):
    set_config(config)
    print('Starting MCTS process %s' % process_id)
    sys.stdout.flush()
    np.random.seed(process_id)
    
    p_em_recv, p_em_send = p_eval_to_mcts
    p_em_send.close()

    def eval_state(state):
        # randomly rotate and flip before evaluating
        k = np.random.randint(4)
        flip = np.random.randint(2)
        if k != 0:
            state = np.rot90(state, k=k, axes=(-2, -1))
        if flip:
            state = np.flip(state, axis=-1)
        q_mcts_to_eval.put((process_id, state.tolist()))
        v, p = p_em_recv.recv()
        p = np.array(p, dtype=np.float32)
        if flip:
            p = np.flip(p, axis=-1)
        if k != 0:
            p = np.rot90(p, k=-k, axes=(-2, -1))
        return v, p
    
    # (curr_player, opponent, last_opponent_move, is_curr_player_first)
    start_state = get_start_state()
    mcts = MCTS(start_state, eval_state)
    while True:
        q_mcts_to_train.put(tuple(x.tolist() for x in mcts.run()))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    print('Main (training) process id %s' % os.getpid())

    args = parser.parse_args()

    # load config
    config = Config(**vars(args)).load()
    set_config(config)
    if args.debug:
        config.num_mcts_processes = config.pred_batch = 2
        config.min_num_states = 2
        config.max_mcts_queue = 100
        config.epoch_update_model = 15
        config.time_update_model = 5
        config.mcts_iterations = 100

    # load or initialize model for training
    config.device = config.device_t
    state = config.load_max_model_state(min_epoch=-1)
    model = Model(config)
    if state is None:
        print('No saved models found')
        state = model.get_state()
    else:
        print('Loaded model state from epoch %s' % state['epoch'])
        model.set_state(state)

    # communication pipes and queues
    _, p_te_send = p_train_to_eval = Pipe() # train process sends models to eval process
    q_mcts_to_eval = Queue() # MCTS processes sends states for evaluation to eval process
    ps_eval_to_mcts = [Pipe() for _ in range(config.num_mcts_processes)] # eval process sends evaluations back to MCTS processes
    q_mcts_to_train = Queue() # MCTS processes sends labeled states to train process
    
    # start (gpu) process to evaluate game states
    eval_proc = Process(target=eval_fn, args=(config, p_train_to_eval, q_mcts_to_eval, ps_eval_to_mcts))
    eval_proc.daemon = True
    eval_proc.start()
    print('Sending model state at epoch %s from train to eval' % state['epoch'])
    p_te_send.send(state)
    
    # start mcts cpu processes
    for i, p_eval_to_mcts in enumerate(ps_eval_to_mcts):
        proc = Process(target=mcts_fn, args=(config, q_mcts_to_train, q_mcts_to_eval, p_eval_to_mcts, i))
        proc.daemon = True
        proc.start()

    model.set_communication(q_mcts_to_train, p_te_send)

    model.fit(config.train_epochs)
    
