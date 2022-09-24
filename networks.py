import torch
import torch.nn as nn
import math

from utils import compute_c_transform
from datacreation import euclidean_cost_matrix

class FCNN(nn.Module):
    """
    Concatenates the two distributions before the first layer.
    """
    def __init__(self, dim, symmetry = False, doubletransform = False, zerosum = False):
        """
        :param symmetry: if True, enforces symmetry
        :param doubletransform: if True, returns the double c transform of the output.
        :param zerosum: if True, shifts the output such that it has zero sum (like the potentials in the training data).
        """
        super(FCNN3, self).__init__()
        self.symmetry = symmetry
        self.doubletransform = doubletransform
        self.zerosum = zerosum
        self.dim = dim
        l = int(math.sqrt(dim))
        self.cost = euclidean_cost_matrix(l, l, 2, True)
        self.l1 = nn.Sequential(
            nn.Linear(2*dim, 6*dim),
            nn.BatchNorm1d(6*dim),
            nn.ReLU(),
            )
        self.l2 = nn.Sequential(
            nn.Linear(6*dim, 6*dim),
            nn.BatchNorm1d(6*dim),
            nn.ReLU(),
            )
        self.l3 = nn.Sequential(
            nn.Linear(6*dim, 1*dim),
            )
        self.layers = [
            self.l1,
            self.l2,
            self.l3
        ]

    def forward(self, x):
        if not self.symmetry:
            for l in self.layers:
                x = l(x)
        else:
            x1 = x.clone()
            for l in self.layers:
                x1 = l(x1)
            n = x.size(1)//2
            x2 = torch.cat((x[:, n:], x[:, :n]), 1)
            for l in self.layers:
                x2 = l(x2)
            x2 = compute_c_transform(self.cost, x2)
            x = (x1 + x2)/2
        if self.doubletransform:
            x = compute_c_transform(self.cost, compute_c_transform(self.cost, x))
        if self.zerosum:
            x = x - x.sum(1)[:,None]/self.dim
        return x
