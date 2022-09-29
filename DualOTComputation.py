from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.nn.functional as F
import random

from networks import *
from datacreation import load_data, euclidean_cost_matrix, data_to_list
from utils import compute_c_transform, compute_dual, compute_mean_conf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_max_ws(t, t2):
    """
    A simple loss function that can be fed to `learn_ws` to learn the WS by simple gradient ascent on the WS computed
    from the potential instead of using a loss of that guess and the ground truth.
    """
    return -t.sum()


class DualApproximator:

    def __init__(
                    self,
                    length,
                    networkclass,
                    lr = 0.005,
                    exponent = 2,
                    model = None
                ):
        """
        Creates an agent that learns the dual potential function.
        :param length: width and height dimension of the data. Dimension of the distribution is length*length.
        :param networkclass: class used for the neural network.
        :param exponent: exponent with which the euclidean distance can be exponentiated.
        :param model: Optional path to a torch model to be loaded.
        """
        self.length = length
        self.net = networkclass(length*length)
        if model != None:
            self.net.load_state_dict(torch.load(model))
        self.net.to(device)
        self.lr = lr
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.costmatrix = torch.tensor(euclidean_cost_matrix(length, length, exponent)).to(device)
        self.parnumber = sum(p.numel() for p in self.net.parameters())

    def load(self, path):
        """
        Loads a model saved in `path`.
        """
        self.net.load_state_dict(torch.load(path))

    def save(self, path):
        """
        Saves the current model in `path`.
        """
        torch.save(self.net.state_dict(),path)

    def reset_params(self):
        """
        Reset the network's parameters to default.
        """
        for c in d.net.children():
            for l in c:
                if hasattr(l, 'reset_parameters'):
                    l.reset_parameters()

    def learn_potential(
                            self,
                            data_filename,
                            loss_function = F.mse_loss,
                            epochs = 1,
                            batchsize = 100,
                            verbose = 0,
                            num_tests = 50,
                            test_data = None,
                            WS_perf = False
                        ):
        """
        Learns from data in file 'data_filename' using `loss_function` loss on the dual potential.
        :param data_filename: file with training data.
        :param loss_function: loss function to be used in backpropagation.
        :param epochs: number of epochs performed on data.
        :param batchsize: batch size used in all epochs.
        :param verbose: if >0, collects performance information and returns it after the last epoch.
        :param num_tests: number of times test data is collected if `verbose` >= 2.
        :param test_data: list of test data used for testing if verbose is True. Can contain various test data sets. If only one test data set is given, it needs to be nested into a 1-element list.
        :param WS_perf: if True, also collects performance on Wasserstein distance calculation in addition to potential approximation.
        """
        if test_data == None: # we oftentimes have a variable 'testdata' predefined.
            try:
                test_data = testdata
            except:
                pass
        if test_data != None:
            test_nb = len(test_data)
        else:
            test_nb = 0
        dataset = data_to_list(data_filename)
        if WS_perf:
            performance = {'WS': [[] for i in range(test_nb)], 'pot': [[] for i in range(test_nb)]}
        else:
            performance = {'pot': [[] for i in range(test_nb)]}
        if verbose >= 2:
            for j in range(test_nb):
                performance['pot'][j].append(self.test_potential(test_data[j]))
            if WS_perf:
                for j in range(test_nb):
                    performance['WS'][j].append(self.test_ws(test_data[j]))
        self.net.train()
        for e in range(epochs):
            perm = torch.randperm(dataset['d1'].size(0))
            for key in dataset.keys():
                dataset[key] = dataset[key][perm]
            for i in tqdm(range(dataset['d1'].size(0)//batchsize)):
                d1 = dataset['d1'][i*batchsize:(i+1)*batchsize]
                d2 = dataset['d2'][i*batchsize:(i+1)*batchsize]
                x = torch.cat((d1, d2), 1)
                out = self.net(x)
                self.optimizer.zero_grad()
                loss = loss_function(out, dataset['u'][i*batchsize:(i+1)*batchsize])
                loss.backward()
                self.optimizer.step()
                if verbose >= 2 and i % (dataset['d1'].size(0)//(num_tests * batchsize)) == 0 and i > 0:
                    if WS_perf:
                        for j in range(test_nb):
                            performance['WS'][j].append(self.test_ws(test_data[j]))
                    for j in range(test_nb):
                        performance['pot'][j].append(self.test_potential(test_data[j]))
                    self.net.train()
            if verbose == 1:
                if WS_perf:
                    for j in range(test_nb):
                        performance['WS'][j].append(self.test_ws(test_data[j]))
                for j in range(test_nb):
                    performance['pot'][j].append(self.test_potential(test_data[j]))
        return performance

    def average_performance(
                                self,
                                data,
                                learn_function,
                                nb_runs = 10,
                                loss = F.mse_loss,
                                num_tests = 30,
                                test_data = None,
                                WS_perf = False,
                                conf = .95,
                                save_models = False,
                                model_name = ''
                            ):
        """
        Runs a learning function and computes the average performance and a confidence interval w.r.t. a loss function `loss` across `nb_runs` runs.
        NOTE: resets all trainable parameters.
        :param data: training data file name.
        :param learn_function: function used for learning.
        :param nb_runs: number of runs, i.e. models trained.
        :param loss: loss function.
        :param num_tests: number of test results in each of the 10 runs.
        :param test_data: list containing test data. If just one test data set is given, it needs to be nested into a 1-element list.
        :param WS_perf: if True, also collects performance information on Wasserstein distance approximation in addition to potential approximation.
        :param conf: confidence for the confidence interval.
        :param save_models: optional boolean. If True, saves all trained networks after their training finished.
        :param model_name: file name of saved models. Models will be saved as `name`_0.pt to `name`_(`nb_runs`-1).pt.
        """
        test_nb = len(test_data)
        results = [[] for i in range(test_nb)]
        if WS_perf:
            WS_results = [[] for i in range(test_nb)]
        for i in range(nb_runs):
            self.reset_params()
            perf = learn_function(data_filename=data, loss_function=loss, num_tests=num_tests, test_data=test_data, verbose=2, WS_perf = WS_perf)
            if save_models:
                self.save(f'{model_name}_{i}.pt')
            for j in range(test_nb):
                results[j].append(perf['pot'][j])
            if WS_perf:
                for j in range(test_nb):
                    WS_results[j].append(perf['WS'][j])
        for j in range(test_nb):
            results[j] = compute_mean_conf(results[j], conf)
        if WS_perf:
            for j in range(test_nb):
                WS_results[j] = compute_mean_conf(WS_results[j], conf)
            return (results, WS_results)
        return results

    def run_tests(self, testdata, test_function):
        """
        A helper function that runs multiple sets of test data through a `test_function`.
        :param testdata: list containing test data.
        """
        results = []
        for d in testdata:
            results.append(test_function(d))
        return results

    def learn_ws(
                    self,
                    data_filename,
                    loss_function = loss_f,
                    epochs = 1,
                    batchsize = 100,
                    verbose = 0,
                    num_tests = 50,
                    test_data = None,
                    WS_perf = False
                ):
        """
        Learns from data in file 'data_filename' using `loss_function` on the _squared_ Wasserstein-2 distance (i.e. the actual OT cost).
        :param data_filename: file with training data.
        :param loss_function: loss function to be used in backpropagation.
        :param epochs: number of epochs performed on data.
        :param batchsize: batch size used in all epochs.
        :param verbose: if >0, collects performance information and returns it after the last epoch.
        :param num_tests: number of times test data is collected if `verbose` >= 2.
        :param test_data: test data used for testing if verbose is True.
        :param WS_perf: if True, also collects performance information on Wasserstein distance approximation in addition to potential approximation.
        """
        if test_data == None: # we oftentimes have a variable 'testdata' predefined.
            try:
                test_data = testdata
            except:
                pass
        if test_data != None:
            test_nb = len(test_data)
        else:
            test_nb = 0
        dataset = data_to_list(data_filename)
        if WS_perf:
            performance = {'WS': [[] for i in range(test_nb)], 'pot': [[] for i in range(test_nb)]}
        else:
            performance = {'pot': [[] for i in range(test_nb)]}
        if verbose >= 2:
            for j in range(test_nb):
                performance['pot'][j].append(self.test_potential(test_data[j]))
            if WS_perf:
                for j in range(test_nb):
                    performance['WS'][j].append(self.test_ws(test_data[j]))
        self.net.train()
        for e in range(epochs):
            perm = torch.randperm(dataset['d1'].size(0))
            for key in dataset.keys():
                dataset[key] = dataset[key][perm]
            for i in tqdm(range(dataset['d1'].size(0)//batchsize)):
                d1 = dataset['d1'][i*batchsize:(i+1)*batchsize]
                d2 = dataset['d2'][i*batchsize:(i+1)*batchsize]
                x = torch.cat((d1, d2), 1)
                u = self.net(x)
                v = compute_c_transform(self.costmatrix, u)
                ws_guess = compute_dual(d1, d2, u, v)
                self.optimizer.zero_grad()
                loss = loss_function(ws_guess, dataset['cost'][i*batchsize:(i+1)*batchsize])
                loss.backward()
                self.optimizer.step()
                if verbose >= 2 and i % (dataset['d1'].size(0)//(num_tests * batchsize)) == 0 and i > 0:
                    if WS_perf:
                        for j in range(test_nb):
                            performance['WS'][j].append(self.test_ws(test_data[j]))
                    for j in range(test_nb):
                        performance['pot'][j].append(self.test_potential(test_data[j]))
                    self.net.train()
            if verbose == 1:
                if WS_perf:
                    for j in range(test_nb):
                        performance['WS'][j].append(self.test_ws(test_data[j]))
                for j in range(test_nb):
                    performance['pot'][j].append(self.test_potential(test_data[j]))
        return performance

    def learn_multiple_files(
                                self,
                                filename,
                                start,
                                end,
                                f,
                                lr = None,
                                meta_epochs = 1,
                                loss_function = F.mse_loss,
                                epochs = 1,
                                batchsize = 100,
                                verbose = 0,
                                num_tests = 50,
                                test_data = None,
                                WS_perf = False,
                                save_points = None,
                                save_name = None
                            ):
        """
        Wraps a learning function `f` to call it on multiple files of format `filename`_`start`.py, `filename`_{`start`+1}', ..., `filename`_`end`.
        :param filename: determines files `filename`_`start`.py to `filename`_`end`.py to be used as files.
        :param start: index of first file.
        :param end: index of last file.
        :param f: learning function.
        :param lr: list of length `meta_epochs`*(`end`+1-`start`) containing learning rates. If None, learning rate remains unchanged.
        :param meta_epochs: number of times all files are used for training.
        :param loss_function: loss function to be used in backpropagation.
        :param epochs: number of epochs performed on data during each meta epoch.
        :param batchsize: batch size.
        :param verbose: verbose parameter passed to the learning function.
        :param num_tests: number of times test data is collected during each call of the learning function if `verbose` >= 2.
        :param test_data: test data used for testing if verbose is True.
        :param WS_perf: if True, also collects performance data on the WS distance approximation.
        :param save_points: optional list containing tuples indicating the points where the network will be saved. First entry indicates meta epoch, second indicates file number.
        :param save_name: file name for saving networks. Will be saved as `save_name`_`i`_`j`.pt where `i` is the meta epoch and `j` the file name number.
        """
        if save_points == None:
            save_points = []
        if test_data:
            test_nb = len(test_data)
        else:
            test_nb = 0
        if WS_perf:
            performance = {'WS': [[] for i in range(test_nb)], 'pot': [[] for i in range(test_nb)]}
        else:
            performance = {'pot': [[] for i in range(test_nb)]}
        for j in range(meta_epochs):
            for i in range(start, end + 1):
                if lr:
                    self.lr = lr[j*(end + 1 - start) + i]
                print(f'Metaepoch {j}, file {i} of {end}.')
                log = f(f'{filename}_{i}.py', loss_function=loss_function, epochs=epochs, batchsize=batchsize, verbose=verbose, num_tests=num_tests, test_data=test_data)
                if (j,i) in save_points:
                    d.save(f'{save_name}_{j}_{i}.pt')
                if j == 0 and i == 0:
                    if WS_perf:
                        for l in range(test_nb):
                            performance['WS'][l] += log['WS'][l]
                    for l in range(test_nb):
                        performance['pot'][l] += log['pot'][l]
                else:
                    if WS_perf:
                        for l in range(test_nb):
                            performance['WS'][l] += log['WS'][l][1:] # remove the first value as it's the same as the last value from the previous iteration
                    for l in range(test_nb):
                        performance['pot'][l] += log['pot'][l][1:]
        return performance

    def predict(self, a, b):
        """
        Concatenates input distributions `a` and `b` to compute the network's output.
        """
        self.net.eval()
        with torch.no_grad():
            x = torch.cat((a, b), 1)
            out = self.net(x)
        return out

    def test_potential(self, data, loss_function = F.mse_loss):
        '''
        Tests the network on test data 'data' which should be constructed from 'generate_simple_data' with 'n_samples'=='batchsize'.
        Returns the average `loss_function` error on the dual potential.
        '''
        self.net.eval()
        l = 0
        with torch.no_grad():
            for batch in data:
                loss = loss_function(self.predict(batch['d1'], batch['d2']), batch['u'])
                l += loss.item()
        l /= len(data)
        return l

    def test_ws(self, data, loss_function = F.mse_loss, rel = False, return_ws = False):
        '''
        Tests the network on test data 'data' which should be a list with dict entries of the format produced by 'generate_simple_data' or 'generate_structured_data'.
        Returns the average `loss_function` error on the squared Wasserstein-2 distance (i.e. the OT cost).
        :param rel: if True, returns the relative error, computed as `[\sum (a_i-b_i)^z]/[\sum b_i^z]`, where a is the prediction and b the ground truth.
        :param return_ws: if True, returns the approximated WS distances alongside the ground truth.
        '''
        self.net.eval()
        l = 0
        if rel:
            if loss_function == F.mse_loss:
                z = 2
            elif loss_function == F.l1_loss:
                z = 1
            else:
                raise ValueError(f'Loss function {loss_function} not supported for relative error computation.')
        if return_ws:
            ws_list = []
        with torch.no_grad():
            for batch in data:
                u = self.predict(batch['d1'], batch['d2'])
                v = compute_c_transform(self.costmatrix, u)
                ws_guess = compute_dual(batch['d1'], batch['d2'], u, v)
                loss = loss_function(ws_guess, batch['cost'])
                if rel:
                    loss *= len(ws_guess)/torch.tensor([abs(c)**z for c in batch['cost']]).sum()
                l += loss.item()
                if return_ws:
                    ws_list.append((ws_guess, batch['cost']))
        l /= len(data)
        if return_ws:
            return (l, ws_list)
        return l


if __name__ == '__main__':
    length = input("Input width of data: ")
    length = int(length)
    #lr = input("Input learning rate: ")
    #lr = float(lr)
    lr = 0.005
    d = DualApproximator(length=length, networkclass=FCNN3, lr=lr)
    testdata=[load_data('Data/test_random_28by28_10k_bs100_mult2_centered.py')]
    testdata2=load_data('Data/mnist_10k_test_300data_bs100_centered_0.py')
    testdata3=load_data('Data/real_mnist_28by28_10k_test_centered_withzeros.py')
    testdata4=[load_data('Data/test_10k_real_mnist_14by14_withzeros_centered_0.py'),load_data('Data/test_10k_real_mnist_14by14_withoutzeros_centered_0.py'),load_data('Data/mnist_10k_test_300data_bs100_14by14_nozeros_centered_0.py'),load_data('Data/test_10k_random_14by14_withzeros_mult0_centered.py'),load_data('Data/test_10k_random_14by14_withzeros_mult2_centered.py'),load_data('Data/test_10k_random_14by14_nozeros_mult2_centered.py'),load_data('Data/test_10k_cifar_nozeros_centered_14by14_0.py')]
    loc = 'Data/random_28by28_100k_bs100_centered_0.py'
    testdata5 = [load_data('Data/random_28by28_10k_test_bs100_mult3_nozeros_centered.py'), load_data('Data/test_10k_random_28by28_mult0_withzeros_centered.py'), load_data('Data/real_mnist_28by28_10k_test_centered_nozeros.py'), load_data('Data/test_10k_cifar_nozeros_centered_28by28.py')]
    #a = torch.tensor([.5,.25,.25,.0],dtype=torch.float)[None,:].to(device)
    #b = torch.tensor([.0,.5,.0,.5], dtype=torch.float)[None,:].to(device)
    #testdata = generate_simple_data(length=length, n_samples=10000, batchsize=10000)
    #d.predict(a, b)
