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
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from skimage.draw import random_shapes
from costmatrix import euclidean_cost_matrix
from sinkhorn import sinkhorn

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

def load_files_quickdraw(dir, categories = 8, per_category = 1000, rand = True, names = None, not_names = None):
	'''
	Loads data. NOTE: Change directory accordingly. CAREFUL: attempting to load multiple categories at once (50+) can kill the call.
    :param dir: directory where data is stored.
	:param categories: number of categories to load. If set to 'ALL', loads all categories. WARNING: this will most likely kill the call.
	:param per_category: number of samples to load per category. Set to `None` to load complete category.
	:param rand: if True, chooses random categories to load. If False, either loads categories from `names` if passed or the first `categories` from alphabetical order.
	:param names: optional argument to pass specific names of categories to load. Iterable that contains strings. Needs `rand` to be set to False. Ignores the `categories` argument if passed.
	:param not_names: Iterable that contains strings. If passed, the selection of names makes sure not to take any names in `not_names`.
	:return: dictionary with category names as keys, and np.arrays as data.
	'''
	filenames = sorted(os.listdir(dir))
	if categories == 'ALL':
		categories = len(filenames)
	if not rand:
		if names:
			for i in range(len(names)):
				if names[i].endswith('.npy'):
					names[i] = names[i][:-4]
			data = {name: None for name in names}
		else:
			filenames = [filename for filename in filenames if not filename in not_names]
			data = {name[:-4]: None for name in filenames[:categories]}
	else:
		if not_names == None:
			not_names = []
		for i in range(len(not_names)):
			if not not_names[i].endswith('.npy'):
				not_names[i] += '.npy'
		cats = random.sample([filename for filename in filenames if not filename in not_names], categories)
		data = {name[:-4]: None for name in cats}
	for name in data:
		data[name] = np.load(dir+name+'.npy')[:per_category]
	return data

def create_more_data(
                        n_samples = 1000,
                        dataloader = 'MNIST',
                        rescale = True,
                        remove_zeros = True,
                        original = False,
                        data = None
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
    :param data: optional parameter, with which one can pass a three-dimensional tensor of dim (n_samples, length, length) containing data to be used instead.
    :return: tensor of size approx. (n_samples**2/2, samplesize).
    """
    if data == None:
        if dataloader == 'MNIST':
            dataloader = iter(DataLoader(MNIST(root='./Data/mnist_dataset',train=True,download=True,transform=torchvision.transforms.ToTensor()), batch_size=n_samples))
        data = next(dataloader)[0].squeeze().double().to(device)
    else:
        data = data.double().to(device)
    if data.dim() == 4: # this is the case for colored images, in which case they're converted to black and white
        data = data.sum(1)
    if original:
        if remove_zeros:
            data = data + 1e-3*torch.ones(data.size()).to(device)
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
                            dataloader = None,
                            remove_zeros = True,
                            train = True,
                            batch_size = 100,
                            n_samples = 100000,
                            center = True,
                            length = 28,
                            data = None
                        ):
    """
    Generates samples from a DataLoader object passed via `dataloader`, and saves the data in a file named `name`.
    Can also generate samples from data passed via the optional `data` argument.
    :param name: file name where data will be saved.
    :param dataloader: A torch.DataLoader object.
    :param remove_zeros: removes zeros from distributions by adding a constant 1e-3 to all points in the distribution before rescaling it to sum to 1.
    :param train: if True, collects distributions from the dataset's training data. Otherwise, collects it from the test data.
    :param batch_size: batch size of the dataset to be created.
    :param n_samples: number of samples saved in each of the `n` files.
    :param center: if True, centers potential data such that each potential has zero sum.
    :param length: lets you adjust the size of the output samples; `length` refers to the height resp. width. Defaults to 28 (original mnist sample size).
    :param data: optional parameter, with which one can pass a tensor that contains two-dimensional or three-dimensional data to be used.
    """
    if data == None:
        if dataloader == None:
            dataloader = DataLoader(MNIST(root='./Data/mnist_dataset',train=train,download=True,transform=torchvision.transforms.ToTensor()), batch_size=60000)
        gen = iter(dataloader)
        data = create_more_data(60000, gen, True, remove_zeros, True)
    else:
        if data.dim() == 2:
            l = int(math.sqrt(data.size(1)))
            data = data.reshape(data.size(0), l, l)
            data = create_more_data(60000, None, True, remove_zeros, True, data)
    data = data.reshape(data.size(0), data.size(1)*data.size(2))
    if length*length != data.size(1):
        data = resize_tensor(data, (length, length))
    cost_matrix = euclidean_cost_matrix(length, length, 2, True)
    dataset = []
    a = torch.cat(([data[torch.randperm(data.size(0)).to(device)] for i in range(n_samples//data.size(0) + 1)]))
    b = torch.cat(([data[torch.randperm(data.size(0)).to(device)] for i in range(n_samples//data.size(0) + 1)]))
    for i in tqdm(range(n_samples//batch_size)):
        batch_data = []
        for j in range(batch_size):
            log = ot.emd(a[i*batch_size+j], b[i*batch_size+j], cost_matrix, log=True)
            batch_data.append({'d1': a[i*batch_size+j].float()[None,:].to(device), 'd2': b[i*batch_size+j].float()[None,:].to(device), 'u': log[1]['u'].float()[None,:].to(device), 'cost': torch.tensor([log[1]['cost']], dtype=torch.float)[None,:].to(device)})
        batch = {'d1': torch.cat([batch_data[i]['d1'] for i in range(batch_size)], 0), 'd2': torch.cat([batch_data[i]['d2'] for i in range(batch_size)], 0), 'u': torch.cat([batch_data[i]['u'] for i in range(batch_size)], 0), 'cost': torch.cat([batch_data[i]['cost'] for i in range(batch_size)], 0).to(device)}
        if center:
            batch['u'] = batch['u'] - batch['u'].sum(1)[:, None]/(length*length)
        dataset.append(batch)
    save_data(dataset, name)

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

def create_random_cov_matrix(
                                length = 28
                            ):
    """
    Creates a random symmetric, positive semi-definite matrix.
    :param length: height and width of the distributions. Dimension equals `length`*`length`.
    :return: two-dimensional tensor.
    """
    dim = length*length
    sigma = torch.rand(dim, dim)
    sigma = torch.mm(sigma, sigma.T)
    sigma.add_(torch.eye(dim*dim))
    return sigma


def generate_simple_data(
                            file_name,
                            length = 2,
                            n_samples = 2000,
                            cost_exp = 2,
                            batchsize = 100,
                            center = True,
                            remove_zeros = True,
                            mult = 1,
                            sink = True,
                            iters = 1000,
                            eps = 0.3,
                            gauss = False,
                            means = None,
                            covs = None,
                            random_shape=False,
                            dtypein = torch.float64,
                            dtypeout = torch.float64
                        ):
    """
    Generates a dataset of 'n_samples' distributions of size 'length'*'length', their OT cost using the euclidean distance with exponent 'cost_exp' (meaning for default 2, we get
    the squared Wasserstein-2-distance), and their two dual variables. Saves dataset in a file.
    NOTE: If sink==True, all samples resulting in NaN will be removed. If possible, new samples that are not NaN will be generated in order to achieve a total number of `n_samples`
    samples. If all new samples compute to NaN as well, then the function terminates.
    :param file_name: name of the file to save dataset in.
    :param length: width/height of data samples.
    :param n_samples: total number of samples created.
    :param cost_exp: exponential of the distance in the cost matrix.
    :param batchsize: size of each batch created. Creates `n_samples//batchsize` batches.
    :param center: if True, centers all dual potentials such that each sums to zero.
    :param remove_zeros: if True, removes zeros in data (which are unlikely to occur anyways).
    :param mult: number of times samples are multiplied by themselves, leading to more sparse samples with greater Wasserstein distances.
    :param sink: if True, generates data using the Sinkhorn algorithm.
    :param iters: number of iterations for Sinkhorn algorithm.
    :param eps: regularizer for Sinkhorn algorithm.
    :param gauss: if True, uses absolute value of a multivariate Gaussian for sample creation instead.
    :param means: if None, uses zero mean for the Gaussian. Can also be a one dimensional tensor of length `length`**2, in which case it is used as mean. Can also be a list of multiple such means, in which case they're equally used for sample generation. Either BOTH `means` and `covs` are lists, or not.
    :param covs: if None, uses the identity matrix as covariance matrix for the multivariate Gaussian. Can also be a `length`**2 x `length`**2 dimensional tensor, in which case this is used as covariance matrix. Can also be a list of multiple such matrices, in which case they're equally used in sample generation.
    :param dtypein: dtype of tensors for sinkhorn computations.
    :param dtypeout: dtype in which data will be saved.
    """
    dataset = []
    dist_dim = length*length
    cost_matrix = euclidean_cost_matrix(length, length, cost_exp, True)
    if gauss and not isinstance(means, list):
        if means == None:
            means = torch.zeros(dist_dim)
        means = means.to(dtypein).to(device)
        if covs == None:
            covs = torch.eye(dist_dim)
        covs = covs.to(dtypein).to(device)
        gaussian = MultivariateNormal(means, covs)
    for i in tqdm(range(n_samples//batchsize)):
        data = []
        if gauss:
            if isinstance(means, list):
                gaussian = MultivariateNormal(means[int(i/((n_samples//batchsize)/len(means)))], covs[int(i/((n_samples//batchsize)/len(covs)))])
            a, b = gaussian.sample((batchsize,)).abs().to(dtypein).to(device), gaussian.sample((batchsize,)).abs().to(dtypein).to(device)
        elif random_shape:
            a = np.concatenate([np.expand_dims(random_shapes((length, length), channel_axis=None, max_shapes=20)[0], 0) for k in range(batchsize)])
            b = np.concatenate([np.expand_dims(random_shapes((length, length), channel_axis=None, max_shapes=20)[0], 0) for k in range(batchsize)])
            a = torch.from_numpy(a).to(dtypein).to(device)
            b = torch.from_numpy(b).to(dtypein).to(device)
            a = (255-a)/255
            b = (255-b)/255
            a = a.reshape(batchsize, dist_dim)
            b = b.reshape(batchsize, dist_dim)
        else:
            a, b = torch.rand(batchsize, dist_dim).to(dtypein).to(device), torch.rand(batchsize, dist_dim).to(dtypein).to(device)
        a = a**mult
        b = b**mult
        if remove_zeros:
            a /= a.sum(1)[:, None]
            b /= b.sum(1)[:, None]
            a += 2e-5*torch.ones(a.size()).to(dtypein).to(device)
            b += 2e-5*torch.ones(b.size()).to(dtypein).to(device)
        a /= a.sum(1)[:, None]
        b /= b.sum(1)[:, None]
        if not sink:
            log = ot.emd(a[0], b[0], cost_matrix, log=True)
            log[1]['u'] = log[1]['u'][None, :].to(dtypein).to(device)
            log[1]['cost'] = torch.tensor([[log[1]['cost']]]).to(dtypein).to(device)
            for k in range(1, batchsize):
                curr_log = ot.emd(a[k], b[k], cost_matrix, log=True)
                log[1]['u'] = torch.cat((log[1]['u'], curr_log[1]['u'][None, :].to(dtypein).to(device)), 0)
                log[1]['cost'] = torch.cat((log[1]['cost'], torch.tensor([[curr_log[1]['cost']]]).to(dtypein).to(device)), 0)
            batch = {'d1': a.to(dtypeout), 'd2': b.to(dtypeout), 'u': log[1]['u'].to(dtypeout), 'cost': log[1]['cost'].to(dtypeout)}
        else:
            log = (0, sinkhorn(a, b, cost_matrix, eps, max_iter=iters, log=True)) # cast to tuple s.t. this has the same format as the `ot.emd` output.
            log[1]['cost'] = log[1]['cost'][:, None]
            idx = torch.tensor([k for k in range(batchsize) if not torch.any(log[1]['u'][k].isnan()) and not log[1]['cost'][k].isnan()]).to(device)
            if len(idx) == 0:
                print("Warning! All samples compute to NaN. Try a smaller regularizing coefficient or less iterations.")
                return
            log[1]['u'] = log[1]['u'].index_select(0, idx)
            log[1]['cost'] = log[1]['cost'].index_select(0, idx)
            a = a.index_select(0, idx)
            b = b.index_select(0, idx)
            batch = {'d1': a.to(dtypeout), 'd2': b.to(dtypeout), 'u': log[1]['u'].to(dtypeout), 'cost': log[1]['cost'].to(dtypeout)}
            if not len(idx) == batchsize:
                counter = len(idx)
                while counter < batchsize:
                    if gauss:
                        if isinstance(means, list):
                            gaussian = MultivariateNormal(means[int(i/((n_samples//batchsize)/len(means)))], covs[int(i/((n_samples//batchsize)/len(covs)))])
                            a, b = gaussian.sample((batchsize,)).abs().to(dtypein).to(device), gaussian.sample((batchsize,)).abs().to(dtypein).to(device)
                    elif random_shape:
                        a = np.concatenate([np.expand_dims(random_shapes((length, length), channel_axis=None, max_shapes=20)[0], 0) for k in range(batchsize)])
                        b = np.concatenate([np.expand_dims(random_shapes((length, length), channel_axis=None, max_shapes=20)[0], 0) for k in range(batchsize)])
                        a = torch.from_numpy(a).to(dtypein).to(device)
                        b = torch.from_numpy(b).to(dtypein).to(device)
                        a = (255-a)/255
                        b = (255-b)/255
                        a = a.reshape(batchsize, dist_dim)
                        b = b.reshape(batchsize, dist_dim)
                    else:
                        a, b = torch.rand(batchsize, dist_dim).to(dtypein).to(device), torch.rand(batchsize, dist_dim).to(dtypein).to(device)
                        a = a**mult
                        b = b**mult
                    if remove_zeros:
                        a /= a.sum(1)[:, None]
                        b /= b.sum(1)[:, None]
                        a += 2e-5*torch.ones(a.size()).to(dtypein).to(device)
                        b += 2e-5*torch.ones(b.size()).to(dtypein).to(device)
                    a /= a.sum(1)[:, None]
                    b /= b.sum(1)[:, None]
                    log = (0, sinkhorn(a, b, cost_matrix, eps, max_iter=iters, log=True)) # cast to tuple s.t. this has the same format as the `ot.emd` output.
                    log[1]['cost'] = log[1]['cost'][:, None]
                    idx = torch.tensor([k for k in range(batchsize) if not torch.any(log[1]['u'][k].isnan()) and not log[1]['cost'][k].isnan()]).to(device)
                    if len(idx) == 0:
                        print("Warning! All samples compute to NaN. Try a smaller regularizing coefficient or less iterations.")
                        return
                    counter += len(idx)
                    log[1]['u'] = log[1]['u'].index_select(0, idx)
                    log[1]['cost'] = log[1]['cost'].index_select(0, idx)
                    a = a.index_select(0, idx)
                    b = b.index_select(0, idx)
                    batch = {'d1': torch.cat((a.to(dtypeout), batch['d1']), 0), 'd2': torch.cat((b.to(dtypeout), batch['d2']), 0), 'u': torch.cat((log[1]['u'].to(dtypeout), batch['u']), 0), 'cost': torch.cat((log[1]['cost'].to(dtypeout), batch['cost']), 0)}
            for key in batch.keys():
                batch[key] = batch[key][:batchsize, :]
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
    new_data = {'d1': torch.cat(([data[i]['d1'] for i in range(len(data))])).to(device), 'd2': torch.cat(([data[i]['d2'] for i in range(len(data))])).to(device), 'u': torch.cat(([data[i]['u'] for i in range(len(data))])).to(device), 'cost': torch.cat(([data[i]['cost'] for i in range(len(data))])).to(device)}
    return new_data
