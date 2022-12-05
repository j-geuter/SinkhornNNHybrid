import torch
import torch.nn as nn
import math
from torchvision.transforms import Resize

from src.utils import compute_c_transform
from src.costmatrix import euclidean_cost_matrix


class FCNN(nn.Module):
    """
    Concatenates the two distributions before the first layer.
    """
    def __init__(self, dim, symmetry = False, doubletransform = False, zerosum = False):
        """
        :param dim: dimension of the input images, results in 2*`dim` dimensional input.
        :param symmetry: if True, enforces symmetry
        :param doubletransform: if True, returns the double c transform of the output.
        :param zerosum: if True, shifts the output such that it has zero sum (like the potentials in the training data).
        """
        super(FCNN, self).__init__()
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

class genNet(nn.Module):
    """
    Network that generates new data samples from given samples.
    """
    def __init__(self, dim):
        """
        :param dim: dimension of each of the input distributions. Results in 2*`dim` dimensional input.
        """
        super(genNet, self).__init__()
        self.dim = dim
        self.length = int(math.sqrt(self.dim))
        self.l1 = nn.Sequential(
            nn.Linear(128, 2*dim),
            nn.ReLU(),
            )
        self.layers = [
            self.l1
        ]

    def forward(self, x):
        x_0 = x.detach().clone().reshape(2, x.size(0), 8, 8)
        transform = Resize((self.length, self.length))
        x_0 = torch.cat((transform(x_0[0]).reshape(x.size(0), self.dim), transform(x_0[1]).reshape(x.size(0), self.dim)), 1)
        #x = x.reshape(x.size(0), 2, self.length, self.length)
        for l in self.layers:
            x = l(x)
        x = x + .3*nn.functional.relu(x_0)
        #x = x.reshape(x.size(0), 2*self.dim)
        x = x.to(torch.float64)
        x += 1e-2
        x[:, :self.dim] /=  x[:, :self.dim].sum(1)[:, None]
        x[:, self.dim:] /=  x[:, self.dim:].sum(1)[:, None]
        return x
