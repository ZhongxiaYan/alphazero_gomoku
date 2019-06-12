import numpy as np

def set_config(conf):
    import mcts
    global config
    mcts.config = config = conf

def get_start_state():
    start_state = np.zeros((config.state_size, config.board_dim, config.board_dim), dtype=np.float32)
    start_state[-1] = 1
    return start_state

def index_to_move(index):
    return index // config.board_dim, index % config.board_dim

def move_to_index(move):
    y, x = move
    return y * config.board_dim + x

def step_state(state, move):
    # move and then flip the board
    if config.state_size == 5:
        this_state, opp_state, last_move_2, _, this_first = state.copy()
    else:
        this_state, opp_state, _, this_first = state.copy()
    last_move = np.zeros_like(this_state)
    last_move[move] = 1
    this_state[move] = 1
    if config.state_size == 5:
        return np.stack([opp_state, this_state, last_move, last_move_2, 1 - this_first])
    else:
        return np.stack([opp_state, this_state, last_move, 1 - this_first])


def check_win(player_board, move):
    n_w = config.n_win
    win = np.ones(n_w)
    y, x = move
    start_y = max(y - n_w + 1, 0)
    start_x = max(x - n_w + 1, 0)
    roi = player_board[start_y: y + n_w, start_x: x + n_w]
    
    y -= start_y
    x -= start_x
    diag_k = x - y
    x_flip = roi.shape[1] - x - 1
    diag_k_flip = x_flip - y
    return any((
        (np.convolve(roi[y], win) >= n_w).any(),
        (np.convolve(roi[:, x], win) >= n_w).any(),
        (np.convolve(np.diag(roi, k=diag_k), win) >= n_w).any(),
        (np.convolve(np.diag(np.fliplr(roi), k=diag_k_flip), win) >= n_w).any()
    ))

def save_psq(file, moves, values):
    move_lines = ['%s,%s,0' % (y + 1, x + 1) for y, x in moves]
    lines = ['Piskvorky 20x20, 11:11, 0'] + move_lines
    lines.extend(['AlphaZero 1', 'AlphaZero 2', '-1', '%s,Freestyle' % (1 if values[0] == 1 else 2)])
    file.save_txt('\n'.join(lines))