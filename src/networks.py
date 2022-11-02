import torch
import torch.nn as nn
import math
from torch.fft import fft, ifft


from utils import compute_c_transform
from costmatrix import euclidean_cost_matrix
from complexbatchnorm import CBatchNorm

def complexrelu(input):
    #r = torch.relu(input.real)
    #i = torch.relu(input.imag)
    #return torch.view_as_complex(torch.cat((r[None, :], i[None, :]), 0).T.contiguous()).T.contiguous()
    return torch.where(input.imag<torch.pi/2, input, torch.zeros(1).to(input.dtype))


class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return complexrelu(input)


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
            nn.Linear(2*dim, 6*dim).to(torch.cfloat),
            #CBatchNorm(input_size=6*dim),
            )
        self.l2 = nn.Sequential(
            nn.Linear(6*dim, 6*dim).to(torch.cfloat),
            #CBatchNorm(input_size=6*dim),
            )
        self.l3 = nn.Sequential(
            nn.Linear(6*dim, 1*dim).to(torch.cfloat),
            )
        self.layers = [
            self.l1,
            self.l2
        ]

    def forward(self, x):
        dt = x.dtype
        x = fft(x)
        if not self.symmetry:
            for l in self.layers:
                x = l(x)
                x = complexrelu(x)
            x = self.l3(x)
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
        x = ifft(x).to(dt)
        return x
