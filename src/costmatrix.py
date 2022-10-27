import numpy as np
import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def euclidean_cost_matrix(height, width, exponent = 2, tens = False):
    """
    computes a 2-dimensional C-order numpy array corresponding to a Euclidean distance cost matrix for distributions of size width*height.
    :param width: width of the distributions.
    :param height: height of the distributions.
    :param exponent: exponent of the Euclidean distance. Defaults to 2, i.e. returning the squared Euclidean distance.
    :param tens: if True, returns a tensor object instead.
    """
    n = width*height
    M = np.zeros([n, n], dtype=np.float32)#was np.float64. Is np.float32 more consistent?
    for a in range(n):
        for b in range(n):
            ax = a // width
            ay = a % width
            bx = b // width
            by = b % width
            eucl = math.sqrt((ax - bx)**2 + (ay - by)**2)
            M[a][b] = eucl**exponent
    if tens:
        return torch.tensor(M, dtype=torch.float64).to(device)
    return M
