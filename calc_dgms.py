from gph import ripser_parallel 
import numpy as np
import os
import sys
import torch

def calc_dgms(m, in_token_len, kind = 0):
    # m is lower triangular
    d = np.copy(m.numpy())
    d = d + d.T
    d = 1 - d
    np.fill_diagonal(d, 0)

    if kind == 0:
        d[:in_token_len, :in_token_len] = 0.
    else:
        d[in_token_len:, in_token_len:] = 0.
    
    r = ripser_parallel(d, metric = 'precomputed', n_threads = -1, return_generators = True)

    return r

            
def add_features(data, a_matrix, prompt_len):
    data.extend([torch.sum(torch.diag(a_matrix)[:prompt_len]).item(), torch.sum(torch.diag(a_matrix)[prompt_len:]).item()])
    data.extend([calc_dgms(a_matrix, prompt_len, kind = 0), calc_dgms(a_matrix, prompt_len, kind = 1)])

if __name__ == '__main__':
    try:
        seq, part = int(sys.argv[1]), int(sys.argv[2])
        fname = '/att_tmp/att%d_%d.th' % (seq, part)
        att = torch.load(fname, weights_only=False)

        prompt_len, index = torch.load('/att_tmp/prompt_len.th', weights_only=False)

        all_data = {}
        print('running part', part)
        layer = part 
        thinking_len = index[seq]

        for head in range(32):
            if thinking_len == 0:
                a_matrix = att[head]

                data = []
                add_features(data, a_matrix, prompt_len)
            else:
                a_matrix = att[head][:prompt_len + thinking_len, :prompt_len + thinking_len]

                data = []
                add_features(data, a_matrix, prompt_len)

                a_matrix = att[head][prompt_len:, prompt_len:]
                add_features(data, a_matrix, thinking_len)

            all_data[(seq, layer, head)] = data

        print('LAYER %d, THINKING_LEN = %d' % (layer, thinking_len))

        print(seq, part, 'finished')
        torch.save(all_data, '%s_res' % fname)

    except Exception as e:
        print('calc_dgms', e)
