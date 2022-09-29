import torch
import math
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from tqdm import tqdm

from datacreation import euclidean_cost_matrix
from utils import compute_c_transform


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sinkhorn(
                mu,
                nu,
                C,
                eps,
                max_iter = 100,
                start = None,
                log = False,
                stopThr = 0,
                tens_type = torch.float64,
                verbose = True,
                min_start = None,
                max_start = None
            ):
    """
    Sinkhorn's algorithm to compute the dual potentials and the dual problem value. Allows for parallelization.
    :param mu: first distribution. Two-dimensional tensor where first dimension corresponds to number of samples and second dimension to sample size. Can also be 1D for a single sample.
    :param nu: second distribution. Two-dimensional tensor as above.
    :param C: cost matrix. Two-dimensional tensor.
    :param eps: regularizer.
    :param max_iter: maximum number of iterations.
    :param start: first iteration's starting vector. If None, this is set to ones.
    :param log: if True, returns the optimal plan and dual potentials alongside the cost; otherwise, returns only the cost.
    :param stopThr: if greater than 0, the algorithm terminates if all approximations lie below this threshold, measured in terms of marginal constraint violation.
    :param tens_type: determines the dtype of all tensors involved in computations. Defaults to float64 as this allows for greater accuracy.
    :param verbose: if False, turns off all print statements.
    :param min_start: if given, sets all entries in the starting vector smaller than `min_start` equal to `min_start`.
    :param max_start: if given, sets all entries in the starting vector larger than `max_start` equal to `max_start`.
    """
    if mu.dim() == 1:
        mu = mu[None, :]
    if nu.dim() == 1:
        nu = nu[None, :]
    if start == None:
        start  = torch.ones(mu.size())
    start = start.detach()
    if max_start:
        start = torch.where(start<torch.tensor(max_start).to(start.dtype), start, torch.tensor(max_start).to(start.dtype))
    if min_start:
        start = torch.where(start>torch.tensor(min_start).to(start.dtype), start, torch.tensor(min_start).to(start.dtype))
    mu = mu.T.to(tens_type).to(device)
    nu = nu.T.to(tens_type).to(device)
    start = start.T.to(tens_type).to(device)
    K = torch.exp(-C/eps).to(tens_type).to(device)
    v = start
    it = max_iter
    for i in range(max_iter):
        u = mu/torch.matmul(K, v)
        v = nu/torch.matmul(K.T, u)
        if stopThr > 0:
            gamma = torch.matmul(torch.cat([torch.diag(u.T[j])[None, :] for j in range(u.size(1))]), torch.matmul(K, torch.cat([torch.diag(v.T[j])[None, :] for j in range(u.size(1))])))
            mu_star = torch.matmul(gamma, torch.ones(u.size(0)).to(tens_type).to(device))
            nu_star = torch.matmul(torch.ones(u.size(0)).to(tens_type).to(device), gamma)
            mu_err = (mu.T - mu_star).abs().sum(1)
            nu_err = (nu.T - nu_star).abs().sum(1)
            if (mu_err < stopThr).sum().item() == mu_err.size(0) and (nu_err < stopThr).sum().item() == nu_err.size(0):
                it = i + 1
                if verbose:
                    print(f'Accuracy below threshold. Early termination after {it} iterations.')
                break
    gamma = torch.matmul(torch.cat([torch.diag(u.T[j])[None, :] for j in range(u.size(1))]), torch.matmul(K, torch.cat([torch.diag(v.T[j])[None, :] for j in range(u.size(1))])))
    cost = (gamma * C).sum(1).sum(1)
    if not log:
        return cost
    else:
        mu_star = torch.matmul(gamma, torch.ones(u.size(0)).to(tens_type).to(device))
        nu_star = torch.matmul(torch.ones(u.size(0)).to(tens_type).to(device), gamma)
        mu_err = (mu.T - mu_star).abs().sum(1)
        nu_err = (nu.T - nu_star).abs().sum(1)
        mu_err = mu_err.sum()/mu_err.size(0)
        nu_err = nu_err.sum()/nu_err.size(0)
        return {'cost': cost, 'plan': gamma, 'iterations': it, 'u': eps * torch.log(u).T, 'v': eps * torch.log(v).T, 'average marginal constraint violation': (mu_err + nu_err)/2}

def average_accuracy(
                        data,
                        init,
                        nb_iters,
                        loss_f = F.l1_loss,
                        eps = .24,
                        min_start = None,
                        max_start = None,
                        conf = .95
                    ):
    """
    Computes the average accuracy of the Sinkhorn algorithm for a fixed number of iterations using an initialization specified by `init` on a batch of `data`.
    Splits `data` into ten subsets and computes the mean performance and a `conf` confidence interval.
    Accuracy measured in terms of `loss_f` error on the Wasserstein distance.
    :param data: data used for computation. Either a dict with keys 'd1', 'd2', 'u' and 'cost' or a list of such dicts in which case they are concatenated.
    :param init: initialization scheme. Takes functions that compute the first dual variable of the OT problem. If None, the default initialization is used instead.
    :param nb_iters: number of iterations used in the Sinkhorn algorithm.
    :param loss_f: loss function used to calculate error.
    :param eps: regularizer used for Sinkhorn algorithm.
    :param min_start: sets all values in Sinkhorn initialization smaller than `min_start` to `min_start`.
    :param max_start: sets all values in Sinkhorn initialization larger than `max_start` to `max_start`.
    """
    if type(data) == list:
        d = {'d1': None, 'd2': None, 'u': None, 'cost': None}
        for key in d.keys():
            d[key] = torch.cat([data[i][key] for i in range(len(data))])
    else:
        d = data
    l = int(math.sqrt(d['d1'].size(1)))
    c = euclidean_cost_matrix(l, l, 2, True)
    for i in tqdm(range(10)):
        if init == None:
            cost = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=nb_iters, max_start=max_start, min_start=min_start)
        else:
            c_transform = compute_c_transform(c, init(torch.cat((d['d1'], d['d2']), 1)))
            cost = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=nb_iters, start=torch.exp(c_transform/eps), max_start=max_start, min_start=min_start)
    return loss_f(cost, d['cost'].view(-1))


def compare_iterations(
                            data,
                            inits,
                            names,
                            acc = 'WS',
                            max_iter = 25,
                            eps = 0.24,
                            min_start = None,
                            max_start = None
                        ):
    """
    Compares the accuracy of the sinkhorn function with different intializations for `v` for a varying number of iterations.
    Accuracy either measured in terms of L1 error on WS distance or in terms of marginal constraint violation; can be controlled with the `acc` parameter.
    :param data: data used for computations. Either a dict with keys 'd1', 'd2', 'u' and 'cost' or a list of such dicts in which case they are concatenated.
    :param inits: list of initialization schemes. Takes functions that compute the first dual variable of the OT problem. If one is None, the default initialization is used instead.
    :param names: names of initialization schemes.
    :param acc: specifies how accuracy is computed. If set to 'WS' computes the average L1 error on the Wasserstein distance. If set to 'marg' computes the marginal constraint violations.
    :param max_iter: maximum number of iterations.
    :param eps: regularizer.
    :param min_start: sets all values in Sinkhorn initialization smaller than `min_start` to `min_start`.
    :param max_start: sets all values in Sinkhorn initialization larger than `max_start` to `max_start`.
    """
    if type(data) == list:
        data_dict = {'d1': None, 'd2': None, 'u': None, 'cost': None}
        for key in data_dict.keys():
            data_dict[key] = torch.cat([data[i][key] for i in range(len(data))])
    else:
        data_dict = data
    iters = [int(i*max_iter/25) + 1 for i in range(25)]
    errs = [[] for i in range(len(inits))]
    l = int(math.sqrt(data_dict['d1'].size(1)))
    c = euclidean_cost_matrix(l, l, 2, True)
    for i in range(25):
        for j in range(len(inits)):
            if acc == 'marg':
                if inits[j] == None:
                    err = sinkhorn(data_dict['d1'], data_dict['d2'], c, eps, max_iter=iters[i], log=True, max_start=max_start, min_start=min_start)['average marginal constraint violation'].item()
                else:
                    err = sinkhorn(data_dict['d1'], data_dict['d2'], c, eps, max_iter=iters[i], start=torch.exp(compute_c_transform(c, inits[j](torch.cat((data_dict['d1'], data_dict['d2']), 1)))/eps), log=True, max_start=max_start, min_start=min_start)['average marginal constraint violation'].item()
            elif acc == 'WS':
                if inits[j] == None:
                    cost = sinkhorn(data_dict['d1'], data_dict['d2'], c, eps, max_iter=iters[i], max_start=max_start, min_start=min_start)
                else:
                    cost = sinkhorn(data_dict['d1'], data_dict['d2'], c, eps, max_iter=iters[i], start=torch.exp(compute_c_transform(c, inits[j](torch.cat((data_dict['d1'], data_dict['d2']), 1)))/eps), max_start=max_start, min_start=min_start)
                err = ((cost - data_dict['cost'].view(-1)).abs().sum()/cost.size(0)).item()
            else:
                raise ValueError('Not a valid `acc` type! Try `WS` for the L1 error on the Wasserstein distance or `marg` for the average marginal constraint violation.')
            errs[j].append(err)
    for j in range(len(inits)):
        plt.plot(iters, errs[j], label = names[j])
    plt.xlabel('Iterations')
    plt.ylabel('Error on Wasserstein distance')
    plt.legend()
    plt.show()

def compare_time(
                    data,
                    inits,
                    names,
                    eps = 0.24,
                    acc = 'WS',
                    min_iter = 20,
                    max_iter = 100,
                    step_size = 1,
                    min_start = None,
                    max_start = None
                ):
    """
    Compares the time it takes for sinkhorn to achieve a certain level of accuracy in terms of L1 error on the Wasserstein distance or in terms of marginal constraint violation
    for various initialization schemes.
    :param data: data used for computations. Either a dict with keys 'd1', 'd2', 'u' and 'cost' or a list of such dicts in which case they are concatenated.
    :param inits: initialization schemes. Takes functions that compute the first dual variable of the OT problem. If one is None, the default initialization is used instead.
    :param names: names of initialization schemes.
    :param eps: regularizer.
    :param acc: specifies how accuracy is computed. If set to 'WS' computes the average L1 error on the Wasserstein distance. If set to 'marg' computes the marginal constraint violations.
    :param min_iter: minimum number of iterations performed. Interpolates between `min_iter` and `max_iter`.
    :param max_iter: maximum number of iterations performed in order to achieve required accuracy. Termination at `max_iter`.
    :param step_size: step size of iterations. Tries to achieve required accuracy starting with a single iteration and then increases number of iterations successively by `step_size`.
    :param min_start: sets all values in Sinkhorn initialization smaller than `min_start` to `min_start`.
    :param max_start: sets all values in Sinkhorn initialization larger than `max_start` to `max_start`.
    """
    if type(data) == list:
        d = {'d1': None, 'd2': None, 'u': None, 'cost': None}
        for key in d.keys():
            d[key] = torch.cat([data[i][key] for i in range(len(data))])
    else:
        d = data
    times = [[] for i in range(len(inits))]
    errs = [[] for i in range(len(inits))]
    l = int(math.sqrt(d['d1'].size(1)))
    c = euclidean_cost_matrix(l, l, 2, True)
    for j in range(len(inits)):
        for l in range(min_iter, max_iter, step_size):
            if acc == 'marg':
                if inits[j] == None:
                    t1 = time.time()
                    sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, max_start=max_start, min_start=min_start) # one run without computing the log for more accurate timing
                    t2 = time.time() - t1
                    err = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, log=True, max_start=max_start, min_start=min_start)['average marginal constraint violation'].item()
                else:
                    t1 = time.time()
                    sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, start=torch.exp(compute_c_transform(c, inits[j](torch.cat((d['d1'], d['d2']), 1)))/eps), max_start=max_start, min_start=min_start)
                    t2 = time.time() - t1
                    err = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, start=torch.exp(compute_c_transform(c, inits[j](torch.cat((d['d1'], d['d2']), 1)))/eps), log=True, max_start=max_start, min_start=min_start)['average marginal constraint violation'].item()
            elif acc == 'WS':
                if inits[j] == None:
                    t1 = time.time()
                    cost = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, max_start=max_start, min_start=min_start)
                    t2 = time.time() - t1
                else:
                    t1 = time.time()
                    cost = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, start=torch.exp(compute_c_transform(c, inits[j](torch.cat((d['d1'], d['d2']), 1)))/eps), max_start=max_start, min_start=min_start)
                    t2 = time.time() - t1
                err = ((cost - d['cost'].view(-1)).abs().sum()/cost.size(0)).item()
            else:
                raise ValueError('Not a valid `acc` type! Try `WS` for the L1 error on the Wasserstein distance or `marg` for the average marginal constraint violation.')
            errs[j].append(err)
            times[j].append(t2)
    print(f'Errs: {errs}')
    print(f'Times: {times}')
    for j in range(len(inits)):
        plt.plot(errs[j], times[j], label = names[j])
    if acc == 'WS':
        plt.xlabel('Average L1 error on the Wasserstein distance')
    else:
        plt.xlabel('Average marginal constraint violation')
    plt.ylabel('Time in seconds')
    plt.legend()
    plt.show()

def compare_accuracy(
                        data,
                        inits,
                        names,
                        max_acc = 2,
                        min_acc = 0.2,
                        eps = 0.24,
                        acc = 'WS',
                        max_iter = 100,
                        step_size = 1,
                        min_start = None,
                        max_start = None
                    ):
    """
    Compares the number of iterations needed for sinkhorn to achieve a specific accuracy threshold for varying initialization schemes.
    Accuracy can be measured in terms of average L1 error on Wasserstein distance or in terms of marginal constraint violation.
    :param data: data used for computations. Either a dict with keys 'd1', 'd2', 'u' and 'cost' or a list of such dicts in which case they are concatenated.
    :param inits: initialization schemes. Takes functions that compute the first dual variable of the OT problem. If one is None, the default initialization is used instead.
    :param names: names of initialization schemes.
    :param max_acc: maximum accuracy required. Interpolates between `max_acc` and `min_acc`.
    :param min_acc: minimum accuracy required.
    :param eps: regularizer.
    :param acc: specifies how accuracy is computed. If set to 'WS' computes the average L1 error on the Wasserstein distance. If set to 'marg' computes the marginal constraint violations.
    :param max_iter: maximum number of iterations performed in order to achieve required accuracy. Termination at `max_iter`.
    :param step_size: step size of iterations. Tries to achieve required accuracy starting with a single iteration and then increases number of iterations successively by `step_size`.
    :param min_start: sets all values in Sinkhorn initialization smaller than `min_start` to `min_start`.
    :param max_start: sets all values in Sinkhorn initialization larger than `max_start` to `max_start`.
    """
    if type(data) == list:
        d = {'d1': None, 'd2': None, 'u': None, 'cost': None}
        for key in d.keys():
            d[key] = torch.cat([data[i][key] for i in range(len(data))])
    else:
        d = data
    accs = [max_acc - i * (max_acc - min_acc)/25 for i in range(25)]
    iters = [[] for i in range(len(inits))]
    errs = [[] for i in range(len(inits))]
    l = int(math.sqrt(d['d1'].size(1)))
    c = euclidean_cost_matrix(l, l, 2, True)
    steps = [l for l in range(1, max_iter, step_size)]
    for j in range(len(inits)):
        for l in range(1, max_iter, step_size):
            if acc == 'marg':
                if inits[j] == None:
                    err = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, log=True, max_start=max_start, min_start=min_start)['average marginal constraint violation'].item()
                else:
                    err = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, start=torch.exp(compute_c_transform(c, inits[j](torch.cat((d['d1'], d['d2']), 1)))/eps), log=True, max_start=max_start, min_start=min_start)['average marginal constraint violation'].item()
            elif acc == 'WS':
                if inits[j] == None:
                    cost = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, max_start=max_start, min_start=min_start)
                else:
                    cost = sinkhorn(d['d1'], d['d2'], c, eps, max_iter=l, start=torch.exp(compute_c_transform(c, inits[j](torch.cat((d['d1'], d['d2']), 1)))/eps), max_start=max_start, min_start=min_start)
                err = ((cost - d['cost'].view(-1)).abs().sum()/cost.size(0)).item()
            else:
                raise ValueError('Not a valid `acc` type! Try `WS` for the L1 error on the Wasserstein distance or `marg` for the average marginal constraint violation.')
            errs[j].append(err)
            if err <= min_acc:
                break
    for j in range(len(inits)):
        for i in range(len(accs)):
            if errs[j][-1] <= accs[i]:
                for l in range(len(errs[j])):
                    if errs[j][l] <= accs[i]:
                        iters[j].append(steps[l])
                        break
            else:
                break
    for j in range(len(inits)):
        plt.plot(accs[:len(iters[j])], iters[j], label = names[j])
    if acc == 'WS':
        plt.xlabel('Average L1 error on the Wasserstein distance')
    else:
        plt.xlabel('Average marginal constraint violation')
    plt.ylabel('Iterations')
    plt.legend()
    plt.show()

def log_sinkhorn(
                    mu,
                    nu,
                    C,
                    eps,
                    max_iter = 100,
                    start_f = None,
                    start_g = None,
                    log = False,
                    tens_type = torch.float64
                ):
    """
    Sinkhorn's algorithm in log domain to compute the dual potentials and the dual problem value.
    :param mu: first distribution. One-dimensional tensor. Also supports two-dimensional tensor with an empty first dimension.
    :param nu: second distribution. One-dimensional tensor as above.
    :param C: cost matrix. Two-dimensional tensor.
    :param eps: regularizer.
    :param max_iter: maximum number of iterations.
    :param start: first iteration's starting vector. If None, this is set to ones.
    :param log: if True, returns the optimal plan and dual potentials alongside the cost; otherwise, returns only the cost.
    :param tens_type: determines the dtype of all tensors involved in computations. Defaults to float64 as this allows for greater accuracy.
    """
    if mu.dim() == 2:
        mu = mu.view(-1)
    if nu.dim() == 2:
        nu = nu.view(-1)
    mu = mu.to(tens_type).to(device)
    nu = nu.to(tens_type).to(device)
    if start_f == None:
        start_f  = torch.zeros(mu.size())
    if start_g == None:
        start_g  = torch.ones(mu.size())
    start_f = start_f.detach().to(tens_type).to(device)
    start_g = start_g.detach().to(tens_type).to(device)
    f = start_f
    g = start_g
    for i in range(max_iter):
        f = row_min(S(C, f, g),   eps) + f + eps * torch.log(mu)
        g = row_min(S(C, f, g).T, eps) + g + eps * torch.log(nu) # the column minimum function is equivalent to the row minimum function of the transpose
    gamma = torch.matmul(torch.diag(torch.exp(f/eps)), torch.matmul(torch.exp(-C/eps), torch.diag(torch.exp(g/eps))))
    cost = (gamma * C).sum().item()
    if not log:
        return cost
    else:
        return {'cost': cost, 'plan': gamma, 'u': f.T, 'v': g.T}

def S(cost, pot1, pot2):
    """
    Auxiliary function for log_sinkhorn.
    """
    ones = torch.ones(pot1.size())[None, :].to(pot1.dtype)
    return cost - torch.matmul(pot1[None, :].T, ones) - torch.matmul(ones.T, pot2[None, :])

def row_min(A, eps):
    """
    Auxiliary function for log_sinkhorn.
    """
    return -eps * torch.log(torch.exp(-A/eps).sum(1))


if __name__ == '__main__':
    from datacreation import load_data
    from DualOTComputation import *
    testdata=load_data('Data/random_28by28_10k_test_bs100_mult3_nozeros_centered.py')
    d=DualApproximator(28,FCNN3)
    d.load('savedModels/random_28by28_100k_mult3_nozeros_lr0p005_33mioparams_FCNN3_0.pt')
    c=euclidean_cost_matrix(28,28,2,True)
    s=testdata[0]
    f=d.net(torch.cat((testdata[0]['d1'][:20], testdata[0]['d2'][:20]), 1)).detach()
    g=compute_c_transform(c,f)
    fcc=compute_c_transform(c,g)
