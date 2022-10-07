import ot
import math
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import DataLoader
import random
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(name):
    """
    Loads data from pickle file `name`.
    """
    with open(name, 'rb') as f:
        data = pickle.load(f)
    for batch in data:
        for key in batch.keys():
            batch[key] = batch[key].to(device)
    return data

def save_data(data, name):
    """
    Saves `data` in file `name`.
    """
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def create_more_data(
                        n_samples = 1000,
                        dataloader = 'MNIST',
                        rescale = True,
                        remove_zeros = True,
                        original = False
                    ):
    """
    Increase amount of training data available by taking the first `n_samples` of `dataset` and combining any two of them into a new distribution.
    Also works for other datasets than MNIST.
    Quadratically increases amount of data. If n_samples gets too large (>>1000), this might cause memory overload.
    In case no `dataloader` is passed, a new iterable is created within the function. This means it will always start with the first iterate, whereas if a `dataloader`
    is passed to the function, it will subsequently iterate through it whenever the function is called on the same object again.
    :param dataloader: a dataset which needs to be iterable, and one iterate needs to be of size n_samples. In case of 'MNIST', this gets taken care of.
    :param rescale: if True, rescales each sample to sum to one.
    :param remove_zeros: if True, adds 1e-3 to each point in the distribution before rescaling. This prevents arbitrary values at these points of dual solutions in the OT problem.
    :param original: if True, this parameter allows for using the original mnist samples instead of creating samples from two mnist samples each. Essentially, this will transform MNIST into a tensor object and return it.
    :return: tensor of size approx. (n_samples**2/2, samplesize).
    """
    if dataloader == 'MNIST':
        dataloader = iter(DataLoader(MNIST(root='./Data/mnist_dataset',train=True,download=True,transform=torchvision.transforms.ToTensor()), batch_size=n_samples))
    data = next(dataloader)[0].squeeze().double().to(device)
    if data.dim() == 4: # this is the case for colored images, in which case they're converted to black and white
        data = data.sum(1)
    if original:
        if remove_zeros:
            data = data + 1e-3*torch.ones(data.size())
        if rescale:
            data /= data.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        return data
    size = data.size()
    if not remove_zeros:
        t = torch.cat([data+data.roll(i+1, 0) for i in range(math.floor(n_samples/2))]).to(device)
    else:
        t = torch.cat([data+data.roll(i+1, 0)+1e-3*torch.ones(size).to(device) for i in range(math.floor(n_samples/2))]).to(device)
    if rescale:
        t /= t.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    return t

def generate_dataset_data(
                            name,
                            n_files,
                            dataloader = None,
                            remove_zeros = True,
                            train = True,
                            batch_size = 100,
                            n_samples = 100000,
                            center = True,
                            length = 28
                        ):
    """
    Generates samples from a DataLoader object passed via `dataloader`, and saves the data in a file named `name`.
    :param name: file name where data will be saved (concatenated with '_i.py' for i in range(`n_files`)).
    :param n_files: number of files created.
    :param dataloader: A torch.DataLoader object.
    :param remove_zeros: removes zeros from distributions by adding a constant 1e-3 to all points in the distribution before rescaling it to sum to 1.
    :param train: if True, collects distributions from the dataset's training data. Otherwise, collects it from the test data.
    :param batch_size: batch size of the dataset to be created.
    :param n_samples: number of samples saved in each of the `n` files.
    :param center: if True, centers potential data such that each potential has zero sum.
    :param length: lets you adjust the size of the output samples; `length` refers to the height resp. width. Defaults to 28 (original mnist sample size).
    """
    if dataloader == None:
        dataloader = DataLoader(MNIST(root='./Data/mnist_dataset',train=train,download=True,transform=torchvision.transforms.ToTensor()), batch_size=60000)
    gen = iter(dataloader)
    data = create_more_data(60000, gen, True, remove_zeros, True)
    data = data.reshape(data.size(0), data.size(1)*data.size(2))
    if length*length != data.size(1):
        data = resize_tensor(data, (length, length))
    cost_matrix = euclidean_cost_matrix(length, length, 2, True)
    for l in range(n_files):
        dataset = []
        a = torch.cat(([data[torch.randperm(data.size(0))] for i in range(n_samples//data.size(0) + 1)]))
        b = torch.cat(([data[torch.randperm(data.size(0))] for i in range(n_samples//data.size(0) + 1)]))
        print(f'Progress for file {l+1} of {n_files}:')
        for i in tqdm(range(n_samples//batch_size)):
            batch_data = []
            for j in range(batch_size):
                log = ot.emd(a[i*batch_size+j], b[i*batch_size+j], cost_matrix, log=True)
                batch_data.append({'d1': a[i*batch_size+j].float()[None,:].to(device), 'd2': b[i*batch_size+j].float()[None,:].to(device), 'u': log[1]['u'].float()[None,:].to(device), 'cost': torch.tensor([log[1]['cost']], dtype=torch.float)[None,:].to(device)})
            batch = {'d1': torch.cat([batch_data[i]['d1'] for i in range(batch_size)], 0), 'd2': torch.cat([batch_data[i]['d2'] for i in range(batch_size)], 0), 'u': torch.cat([batch_data[i]['u'] for i in range(batch_size)], 0), 'cost': torch.cat([batch_data[i]['cost'] for i in range(batch_size)], 0)}
            if center:
                batch['u'] = batch['u'] - batch['u'].sum(1)[:, None]/(length*length)
            dataset.append(batch)
        save_data(dataset, f'{name}_{l}.py')

def resize_tensor(data, size, filename = None):
    """
    Resizes `data` into samples of size `size`.
    :param data: 2-dimensional tensor of size [n, dim], where n is the number of samples and dim the dimension of each sample.
    :param size: new size of the data. Either a square number (for the dimension) or a tuple of height and length.
    :param filename: optional filename of where the new data is saved. If 'None', returns data instead.
    """
    l = len(data)
    length = int(math.sqrt(len(data[0])))
    data = data.reshape([l, length, length])
    if isinstance(size,tuple):
        new_dim = size[0]*size[1]
        new_size = size
    else:
        new_dim = size
        new_size = (int(math.sqrt(size)),int(math.sqrt(size)))
    transform = Resize(new_size)
    new_data = transform(data).double().to(device)
    new_data = new_data.reshape(l, new_dim)
    new_data /= new_data.sum(-1).unsqueeze(-1)
    if filename == None:
        return new_data
    else:
        save_data(new_data, filename)

def resize_file(from_file, to_file, size, center = True):
    """
    Resizes a file containing data. Data must be a list where each entry is a dictionary with keys 'd1', 'd2', 'u' and 'cost'.
    :param from_file: name of file of input data as string.
    :param to_file: name of file to save new data in as string.
    :param size: new size of data as a tuple of two numbers.
    :param center: if True centers the potentials such that each has zero sum.
    """
    new_data = []
    data = load_data(from_file)
    cost_matrix = euclidean_cost_matrix(size[0], size[1], 2, True)
    dim = size[0]*size[1]
    for j in tqdm(range(len(data))):
        batch = data[j]
        d1, d2 = batch['d1'], batch['d2']
        d1 = resize_tensor(d1, size)
        d2 = resize_tensor(d2, size)
        batch_data = []
        batch_size = len(d1)
        for i in range(batch_size):
            log = ot.emd(d1[i], d2[i], cost_matrix, log=True)
            batch_data.append({'d1': d1[i].float()[None,:].to(device), 'd2': d2[i].float()[None,:].to(device), 'u': log[1]['u'].float()[None,:].to(device), 'cost': torch.tensor([log[1]['cost']], dtype=torch.float)[None,:].to(device)})
        new_batch = {'d1': torch.cat([batch_data[i]['d1'] for i in range(batch_size)], 0), 'd2': torch.cat([batch_data[i]['d2'] for i in range(batch_size)], 0), 'u': torch.cat([batch_data[i]['u'] for i in range(batch_size)], 0), 'cost': torch.cat([batch_data[i]['cost'] for i in range(batch_size)], 0)}
        if center:
            new_batch['u'] = new_batch['u'] - new_batch['u'].sum(1)[:, None]/dim
        new_data.append(new_batch)
    save_data(new_data, to_file)

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

def generate_simple_data(
                            file_name,
                            length = 2,
                            n_samples = 100000,
                            cost_exp = 2,
                            batchsize = 100,
                            center = True,
                            remove_zeros = True,
                            mult = 1,
                            sinkhorn = False,
                            eps = 0.4
                        ):
    """
    Generates a dataset of 'n_samples' distributions of size 'length'*'length', their OT cost using the euclidean distance with exponent 'cost_exp' (meaning for default 2, we get
    the squared Wasserstein-2-distance), and their two dual variables. Saves dataset in a file.
    :param file_name: name of the file to save dataset in.
    :param length: width/height of data samples.
    :param n_samples: total number of samples created.
    :param cost_exp: exponential of the distance in the cost matrix.
    :param batchsize: size of each batch created. Creates `n_samples//batchsize` batches.
    :param center: if True, centers all dual potentials such that each sums to zero.
    :param remove_zeros: if True, removes zeros in data (which are unlikely to occur anyways).
    :param mult: number of times samples are multiplied by themselves, leading to more sparse samples with greater Wasserstein distances.
    :param sinkhorn: if True, generates data using the Sinkhorn algorithm.
    :param eps: regularizer for Sinkhorn algorithm.
    """
    dataset = []
    dist_dim = length*length
    cost_matrix = euclidean_cost_matrix(length, length, cost_exp)
    for i in tqdm(range(n_samples//batchsize)):
        data = []
        for j in range(batchsize):
            a, b = np.random.random(dist_dim), np.random.random(dist_dim)
            for l in range(mult):
                a *= a
                b *= b
            if remove_zeros:
                a += 1e-3*np.ones(len(a))
                b += 1e-3*np.ones(len(b))
            a /= sum(a)
            b /= sum(b)
            if not sinkhorn:
                log = ot.emd(a, b, cost_matrix, log=True)
            else:
                log = ot.sinkhorn2(a, b, cost_matrix, eps, log=True)
                log[1]['cost'] = log[0]
            data.append({'d1': torch.tensor(a, dtype=torch.float)[None,:].to(device), 'd2': torch.tensor(b, dtype=torch.float)[None,:].to(device), 'u': torch.tensor(log[1]['u'], dtype=torch.float)[None,:].to(device), 'cost': torch.tensor([log[1]['cost']], dtype=torch.float)[None,:].to(device)})
        batch = {'d1': torch.cat([data[i]['d1'] for i in range(batchsize)], 0), 'd2': torch.cat([data[i]['d2'] for i in range(batchsize)], 0), 'u': torch.cat([data[i]['u'] for i in range(batchsize)], 0), 'cost': torch.cat([data[i]['cost'] for i in range(batchsize)], 0)}
        if center:
            batch['u'] = batch['u'] - batch['u'].sum(1)[:, None]/dist_dim
        dataset.append(batch)
    save_data(dataset, file_name)


def center_data_potential(from_file, to_file):
    """
    Takes data from file `from_file` as input and centers its potential data s.t. each potential has zero sum. Saves new data in file `to_file`.
    """
    data = load_data(from_file)
    dim = data[0]['u'].size(1)
    for batch in data:
        batch['u'] = batch['u'] - batch['u'].sum(1)[:, None]/dim
    save_data(data, to_file)


def compute_c_transform(cost,sample):
    """
    Computes the c transform of 'sample' w.r.t. to cost matrix 'cost'. Both inputs are required to be of dimension 2. Supports multiple samples.
    --- Copy of the function in utils.py. ---
    """
    lamext=sample.reshape(len(sample),len(sample[0]),1).expand(len(sample),len(sample[0]),len(sample[0])).transpose(2,1)
    lamstar=(cost-lamext).amin(dim=2).float()
    del lamext
    torch.cuda.empty_cache()
    return lamstar

def data_to_list(from_file):
    """
    Takes as input a list of batches and outputs a single dictionary with all samples.
    :param from_file: name of file input data is located at.
    """
    data = load_data(from_file)
    new_data = {'d1': torch.cat(([data[i]['d1'] for i in range(len(data))])), 'd2': torch.cat(([data[i]['d2'] for i in range(len(data))])), 'u': torch.cat(([data[i]['u'] for i in range(len(data))])), 'cost': torch.cat(([data[i]['cost'] for i in range(len(data))]))}
    return new_data
