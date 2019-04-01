import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.modules.distance import PairwiseDistance

def truncate_seqs(xs):
    '''
    Args:
        xs: iterable of sequences
    
    Return:
        np array with shape (batch_size, min_seq_len)
    '''
    x_lens = [len(x) for x in xs]
    min_len = np.min(x_lens)
    xs = np.array([x[:min_len] for x in xs])
    return xs
