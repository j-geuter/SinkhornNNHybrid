import torch
import torch.nn as nn
import math

from utils import compute_c_transform
from datacreation import euclidean_cost_matrix


class FCNN(nn.Module):
    """
    Passes each distribution through separate linear layers before computing a matrix product from the representations which is then passed through multiple linear layers.
    """
    def __init__(self, dim):
        super(FCNN, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.ReLU(),
        )
        self.l1_2 = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.ReLU(),
        )
        self.l2_2 = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.ReLU(),
        )
        self.conn1 = nn.Sequential(
            nn.Linear(4*dim*dim, 2*dim*dim),
            nn.ReLU(),
        )
        self.conn2 = nn.Sequential(
            nn.Linear(2*dim*dim, dim*dim),
            nn.ReLU(),
        )
        self.conn3 = nn.Sequential(
            nn.Linear(dim*dim, dim),
        )
        self.dim = dim

    def forward(self, x):
        x1 = x[:, :self.dim]
        x1 = self.l1(x1)
        x1 = self.l2(x1)
        x1 = torch.transpose(x1[:, None, :], -1, -2)
        x2 = x[:, self.dim:]
        x2 = self.l1_2(x2)
        x2 = self.l2_2(x2)
        x2 = x2[:, None, :]
        x = torch.matmul(x1, x2)
        x = x.flatten(1)
        x = self.conn1(x)
        x = self.conn2(x)
        x = self.conn3(x)
        return x


class FCNN2(nn.Module):
    """
    Passes each distribution through separate linear layers before concatenating them and again passing through linear layers.
    """
    def __init__(self, dim):
        super(FCNN2, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(4*dim, 4*dim),
            nn.ReLU(),
        )
        self.l1_2 = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
        )
        self.l2_2 = nn.Sequential(
            nn.Linear(4*dim, 4*dim),
            nn.ReLU(),
        )
        self.conn1 = nn.Sequential(
            nn.Linear(8*dim, 16*dim),
            nn.ReLU(),
        )
        self.conn2 = nn.Sequential(
            nn.Linear(16*dim, 16*dim),
            nn.ReLU(),
        )
        self.conn3 = nn.Sequential(
            nn.Linear(16*dim, dim),
        )
        self.dim = dim

    def forward(self, x):
        x1 = x[:, :self.dim]
        x1 = self.l1(x1)
        x1 = self.l2(x1)
        x2 = x[:, self.dim:]
        x2 = self.l1_2(x2)
        x2 = self.l2_2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.conn1(x)
        x = self.conn2(x)
        x = self.conn3(x)
        return x


class FCNN3(nn.Module):
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


class FCNN4(nn.Module):
    """
    Passes each distribution through separate CNNs before concatenating the representations.
    """
    def __init__(self,dim):
        super(FCNN4, self).__init__()
        self.dim = dim
        self.length = int(math.sqrt(dim))
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conn1 = nn.Sequential(
            nn.Linear(128*14*14*2, 4*dim),
            #nn.BatchNorm1d(4*dim),
            nn.ReLU(),
        )
        self.conn2 = nn.Sequential(
            nn.Linear(4*dim, 8*dim),
            #nn.BatchNorm1d(8*dim),
            nn.ReLU(),
        )
        self.conn3 = nn.Sequential(
            nn.Linear(8*dim, 16*dim),
            #nn.BatchNorm1d(16*dim),
            nn.ReLU(),
        )
        self.conn4 = nn.Sequential(
            nn.Linear(16*dim, dim),
        )

    def forward(self, x):
        x1 = x[:, :self.dim]
        x1 = x1.view(x1.size(0), self.length, self.length)
        x1 = x1[:, None]
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x[:, self.dim:]
        x2 = x2.view(x2.size(0), self.length, self.length)
        x2 = x2[:, None]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        x2 = self.conv6(x2)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1,x2), 1)
        x = self.conn1(x)
        x = self.conn2(x)
        x = self.conn3(x)
        x = self.conn4(x)
        return x


class FCNN5(nn.Module):
    """
    Stacks the two distributions as two channels in 4-dimensional input to a CNN.
    """
    def __init__(self,dim):
        super(FCNN5, self).__init__()
        self.dim = dim
        self.length = int(math.sqrt(dim))

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conn1 = nn.Sequential(
            nn.Linear(256*14*14, 4*dim),
            nn.BatchNorm1d(4*dim),
            nn.ReLU(),
        )
        self.conn2 = nn.Sequential(
            nn.Linear(4*dim, 8*dim),
            nn.BatchNorm1d(8*dim),
            nn.ReLU(),
        )
        self.conn3 = nn.Sequential(
            nn.Linear(8*dim, 16*dim),
            nn.BatchNorm1d(16*dim),
            nn.ReLU(),
        )
        self.conn4 = nn.Sequential(
            nn.Linear(16*dim, dim),
        )

    def forward(self, x):
        x1 = x[:, :self.dim]
        x1 = x1.view(x1.size(0), self.length, self.length)
        x2 = x[:, self.dim:]
        x2 = x2.view(x2.size(0), self.length, self.length)
        x = torch.stack((x1, x2), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.conn1(x)
        x = self.conn2(x)
        x = self.conn3(x)
        x = self.conn4(x)
        return x
        
