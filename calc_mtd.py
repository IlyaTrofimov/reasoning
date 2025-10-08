from gph import ripser_parallel 
import numpy as np
import os
import sys
import torch

def calc_mtd(m, in_token_len, kind = 0):
    d = np.copy(m.numpy())
    d = d + d.T
    d = 1 - d
    np.fill_diagonal(d, 0)

    if kind == 0:
        d[:in_token_len, :in_token_len] = 0.
    else:
        d[in_token_len:, in_token_len:] = 0.
    
    r = ripser_parallel(d, metric = 'precomputed', n_threads = -1)
    dgm0 = r['dgms'][0]
    dgm1 = r['dgms'][1]
    mtd0 = np.sum(dgm0[dgm0 < np.inf])
    if dgm1.shape[0]:
        mtd1 = np.sum(dgm1[:, 1] - dgm1[:, 0])
    else:
        mtd1 = 0

    return mtd0, mtd1

if __name__ == '__main__':
    seq, part = int(sys.argv[1]), int(sys.argv[2])
    fname = '/att_tmp/att%d_%d.th' % (seq, part)

    prompt_len = torch.load('/att_tmp/prompt_len.th', weights_only=False)
    att = torch.load(fname, weights_only=False)

    all_data = {}

    for layer in range(8):
        for head in range(32):
            a_matrix = att[layer, head]
        
            data = [torch.sum(torch.diag(a_matrix)[:prompt_len]).item(), torch.sum(torch.diag(a_matrix)[prompt_len:]).item()]
            data.extend([calc_mtd(a_matrix, prompt_len, kind = 0), calc_mtd(a_matrix, prompt_len, kind = 1)])

            all_data[(seq, part*8 + layer, head)] = data

    print(seq, part, 'finished')
       
    torch.save(all_data, '%s_res' % fname)
