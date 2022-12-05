from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.nn.functional as F
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import LambdaLR
import ot


from src.networks import FCNN, genNet
from src.costmatrix import euclidean_cost_matrix
from src.datacreation import load_data, data_to_list
from src.utils import compute_c_transform, compute_dual, compute_mean_conf, visualize_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_max_ws(t, t2):
    """
    A simple loss function to learn the WS by gradient ascent on the WS computed
    from the potential instead of using a loss of that guess and the ground truth.
    :param t: tensor.
    :param t2: tensor.
    :return: one-element tensor.
    """
    return -t.sum()


class DualApproximator:

    def __init__(
                    self,
                    length = 28,
                    lr = 0.003,
                    gen_lr = 0.0003,
                    exponent = 2,
                    model = None,
                    gen_model = None,
                    norm_cost = False
                ):
        """
        Creates an agent that learns the dual potential function.
        :param length: width and height dimension of the data. Dimension of the distribution is length*length.
        :param lr: learning rate.
        :param gen_lr: generative model learning rate.
        :param exponent: exponent with which the euclidean distance can be exponentiated.
        :param model: Optional path to a torch model to be loaded for the approximator.
        :param gen_model: Optional path toa  torch model to be loaded for the generator.
        :param norm_cost: if set to True, uses the cost matrix in the unit square. Otherwise, the square size is determined by the `length`.
        """
        self.length = length
        self.dim = length*length
        self.net = FCNN(self.dim)
        if model != None:
            self.net.load_state_dict(torch.load(model))
        self.net.to(device)
        self.gen_net = genNet(self.dim)
        if gen_model != None:
            self.gen_net.load_state_dict(torch.load(gen_model))
        self.gen_net.to(device)
        self.lr = lr
        self.gen_lr = gen_lr
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.gen_optimizer = Adam(self.gen_net.parameters(), lr=gen_lr)
        self.lamb = lambda epoch: 0.99 ** epoch
        self.gen_lamb = lambda epoch: 0.99 ** epoch
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lamb)
        self.gen_scheduler = LambdaLR(self.gen_optimizer, lr_lambda=self.gen_lamb)
        self.costmatrix = torch.tensor(euclidean_cost_matrix(length, length, exponent)).to(device)
        if norm_cost:
            self.costmatrix /= self.dim
        self.parnumber = sum(p.numel() for p in self.net.parameters())
        self.gen_parnumber = sum(p.numel() for p in self.gen_net.parameters())

    def load(self, path1, path2):
        """
        Loads a model saved in `path1` and `path2`.
        """
        self.net.load_state_dict(torch.load(path1))
        self.net.to(device)
        self.gen_net.load_state_dict(torch.load(path2))
        self.gen_net.to(device)

    def save(self, path1, path2):
        """
        Saves the current model in `path1` and `path2`.
        """
        torch.save(self.net.state_dict(), path1)
        torch.save(self.gen_net.state_dict(), path2)

    def reset_params(self):
        """
        Reset both network's parameters to default.
        """
        self.reset_net()
        self.reset_gen_net()

    def reset_net(self):
        """
        Resets only the `net` parameters.
        """
        for c in self.net.children():
            for l in c:
                if hasattr(l, 'reset_parameters'):
                    l.reset_parameters()
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)

    def reset_gen_net(self):
        """
        Resets only the `gen_net` parameters.
        """
        for c in self.gen_net.children():
            for l in c:
                if hasattr(l, 'reset_parameters'):
                    l.reset_parameters()
        self.gen_optimizer = Adam(self.gen_net.parameters(), lr=self.gen_lr)

    def learn_potential(
                            self,
                            n_samples = 100000,
                            loss_function = F.mse_loss,
                            epochs = 5,
                            batchsize = 500,
                            minibatch = 100,
                            verbose = 0,
                            num_tests = 30,
                            test_data = None,
                            WS_perf = False,
                            rel_WS = True,
                            cost_norm_WS = 1,
                            gen_images = True,
                            learn_gen = 1,
                            update_gen_lr = True,
                            prints = False,
                            learn_WS = False
                        ):
        """
        Learns using `loss_function` loss on the dual potential. Can also learn using a loss on the transport distance with `learn_WS`=True.
        :param n_samples: number of unique samples to train on.
        :param loss_function: loss function to be used in backpropagation.
        :param epochs: number of epochs performed on data. Total number of training samples equals `n_samples`*`epochs`.
        :param batchsize: batch size used in all epochs.
        :param minibatch: size of minibatch for each gradient step. Each batch is split into multiple minibatches.
        :param verbose: if >0, collects performance information and returns it after the last epoch.
        :param num_tests: number of times test data is collected if `verbose` >= 2.
        :param test_data: list of test data used for testing if verbose is True. Can contain various test data sets. If only one test data set is given, it needs to be nested into a 1-element list.
        :param WS_perf: if True, also collects performance on Wasserstein distance estimation errors in addition to potential approximation errors.
        :param rel_WS: if True, computes the relative Wasserstein distance errors instead. (Only if WS_perf==True.)
        :param cost_norm_WS: an optional constant to scale the cost function by for Wassertein distance errors. (Only if WS_perf==True.)
        :param gen_images: if True, also collects 5 samples generated by the generator every time that test data is collected.
        :param learn_gen: in every `learn_gen`th iteration, the generating net will be updated. Can be set to `False` to turn off learning.
        :param update_gen_lr: if set to False, the generator's learning rate will remain constant.
        :param prints: if True, prints the losses of both networks during each iteration, along with sample images of the generator.
        :param learn_WS: if True, learns using a loss on the transport distance instead of one on the potentials.
        :return: dict with key 'pot', and also 'WS' if `WS_perf`==True. At each key is a list containing a list for each test dataset in `test_data`. Each list contains information on the respective error (MSE on potential resp. L1 on Wasserstein distance) over the course of learning.
        """
        prior = MultivariateNormal(torch.zeros(128).to(device), torch.eye(128).to(device))
        if num_tests > n_samples//batchsize:
            print("NOTE: `num_tests` exceeds number of iterations. Number of tests reduced to number of iterations.")
        num_tests = max(num_tests, n_samples//batchsize) # make sure the number of tests does not exceed the number of iterations.
        if test_data == None: # we oftentimes have a variable 'testdata' predefined.
            try:
                test_data = testdata
            except:
                pass
        if test_data != None:
            test_nb = len(test_data)
        else:
            test_nb = 0
        performance = {'pot': [[] for i in range(test_nb)]}
        if WS_perf:
            performance['WS'] = [[] for i in range(test_nb)]
        if gen_images:
            performance['ims'] = []
        if verbose >= 2:
            for j in range(test_nb):
                performance['pot'][j].append(self.test_potential(test_data[j]))
            if WS_perf:
                for j in range(test_nb):
                    performance['WS'][j].append(self.test_ws(test_data[j], rel=rel_WS, cost_norm=cost_norm_WS))
            if gen_images:
                with torch.no_grad():
                    samples = self.gen_net(prior.sample((5,)))
                    performance['ims'].append(torch.cat((samples[:, :self.dim][None, :], samples[:, self.dim:][None, :]), 0))

        self.net.train()
        self.gen_net.train()

        for i in tqdm(range(n_samples//batchsize)):

            x_0 = prior.sample((batchsize,))
            x = self.gen_net(x_0).detach()

            if not learn_WS:
                pot = ot.emd(x[0][:self.dim], x[0][self.dim:], self.costmatrix, log=True)[1]['u']
                pot = pot[None, :].to(torch.float32).to(device)
                for k in range(1, batchsize):
                    log = ot.emd(x[k][:self.dim], x[k][self.dim:], self.costmatrix, log=True)[1]
                    pot = torch.cat((pot, log['u'][None, :].to(torch.float32).to(device)), 0)
                x = x.to(torch.float32)
                pot = pot - pot.sum(1)[:, None]/pot.size(1)

            else:
                pot = torch.tensor([ot.emd(x[0][:self.dim], x[0][self.dim:], self.costmatrix, log=True)[1]['cost']]) # name pot a bit misleading but streamlines code
                pot = pot[None, :].to(torch.float32).to(device)
                for k in range(1, batchsize):
                    log = ot.emd(x[k][:self.dim], x[k][self.dim:], self.costmatrix, log=True)[1]
                    pot = torch.cat((pot, torch.tensor([log['cost']])[None, :].to(torch.float32).to(device)), 0)
                x = x.to(torch.float32)

            for e in range(epochs):
                perm = torch.randperm(batchsize).to(device)
                x_curr, pot_curr = x[perm], pot[perm]
                for j in range(batchsize//minibatch):
                    out = self.net(x_curr[j*minibatch:(j+1)*minibatch])
                    #dual = compute_dual(x_curr[j*minibatch:(j+1)*minibatch, :self.dim], x_curr[j*minibatch:(j+1)*minibatch, self.dim:], out)
                    #dual = dual.sum()/minibatch
                    self.optimizer.zero_grad()
                    if not learn_WS:
                        loss = loss_function(out, pot_curr[j*minibatch:(j+1)*minibatch])# - 20*dual
                    else:
                        ws_guess = compute_dual(x_curr[j*minibatch:(j+1)*minibatch, :self.dim], x_curr[j*minibatch:(j+1)*minibatch, self.dim:], out, c=self.costmatrix)
                        loss = loss_function(ws_guess, pot_curr[j*minibatch:(j+1)*minibatch])
                    if prints:
                        print("net loss, j="+str(j)+", loss="+str(loss.item()))
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    out_c = self.net(torch.cat((x_curr[j*minibatch:(j+1)*minibatch,self.dim:], x_curr[j*minibatch:(j+1)*minibatch,:self.dim]), 1))
                    self.optimizer.zero_grad()
                    if not learn_WS:
                        loss_c = loss_function(out_c, compute_c_transform(self.costmatrix, pot_curr[j*minibatch:(j+1)*minibatch], zero_sum=True))
                    else:
                        ws_guess = compute_dual(x_curr[j*minibatch:(j+1)*minibatch, self.dim:], x_curr[j*minibatch:(j+1)*minibatch, :self.dim], out_c, c=self.costmatrix)
                        loss_c = loss_function(ws_guess, pot_curr[j*minibatch:(j+1)*minibatch])
                    loss_c.backward()
                    self.optimizer.step()
            self.scheduler.step()

            if learn_gen != False and i % learn_gen == 0:
                x_gen = self.gen_net(x_0)
                x_gen = x_gen.to(torch.float32)
                #grad = torch.autograd.grad(x_gen, x_0)[0]
                #penalty = 0.1 * grad**2.sum()
                for j in range(batchsize//minibatch):
                    out = self.net(x_gen[j*minibatch:(j+1)*minibatch])
                    self.gen_optimizer.zero_grad()
                    #self.optimizer.zero_grad()
                    if not learn_WS:
                        gen_loss = -loss_function(out, pot[j*minibatch:(j+1)*minibatch])
                    else:
                        ws_guess = compute_dual(x_gen[j*minibatch:(j+1)*minibatch, :self.dim], x_gen[j*minibatch:(j+1)*minibatch, self.dim:], out, c=self.costmatrix)
                        gen_loss = -loss_function(ws_guess, pot[j*minibatch:(j+1)*minibatch])
                    if prints and j==0:
                        print("gen_net loss, i="+str(i)+", gen_loss="+str(gen_loss.item()))
                        visualize_data(torch.cat((x_gen.detach().cpu()[j*minibatch, :self.dim][None, :], x_gen.detach().cpu()[j*minibatch, self.dim:][None, :]), 0), 1, 2)
                    gen_loss.backward(retain_graph=True)
                    self.gen_optimizer.step()
                if update_gen_lr:
                    self.gen_scheduler.step()

            if verbose >= 2 and i % (n_samples//(num_tests * batchsize)) == 0 and i > 0:
                if WS_perf:
                    for j in range(test_nb):
                        performance['WS'][j].append(self.test_ws(test_data[j], rel=rel_WS, cost_norm=cost_norm_WS))
                for j in range(test_nb):
                    performance['pot'][j].append(self.test_potential(test_data[j]))
                self.net.train()
                if gen_images:
                    with torch.no_grad():
                        samples = self.gen_net(prior.sample((5,)))
                        performance['ims'].append(torch.cat((samples[:, :self.dim][None, :], samples[:, self.dim:][None, :]), 0))
        if verbose == 1:
            if WS_perf:
                for j in range(test_nb):
                    performance['WS'][j].append(self.test_ws(test_data[j], rel=rel_WS, cost_norm=cost_norm_WS))
            for j in range(test_nb):
                performance['pot'][j].append(self.test_potential(test_data[j]))
            if gen_images:
                with torch.no_grad():
                    samples = self.gen_net(prior.sample((5,)))
                    performance['ims'].append(torch.cat((samples[:, :self.dim][None, :], samples[:, self.dim:][None, :]), 0))
        if verbose >= 1:
            return performance

    def average_performance(
                                self,
                                learn_function,
                                nb_runs = 10,
                                conf = .95,
                                save_models = False,
                                model_name = '',
                                **kwargs
                            ):
        """
        Runs a learning function and computes the average performance and a confidence interval w.r.t. a loss function `loss` across `nb_runs` runs.
        NOTE: resets all trainable parameters.
        :param learn_function: function used for learning.
        :param nb_runs: number of runs, i.e. models trained.
        :param conf: confidence for the confidence interval.
        :param save_models: optional boolean. If True, saves all trained networks after their training finished.
        :param model_name: file name of saved models. Models will be saved as `name`_0.pt to `name`_(`nb_runs`-1).pt.
        :param **kwargs: keyword arguments passed to `learn_function`.
        :return: if WS_perf==True, returns (results, WS_results), otherwise results. For each test set in test_data, results contains a 3-tuple computed by `compute_mean_conf` which captures the average MSE error on the potential over the course of learning. WS_results is similar, but for the L1 error on the Wasserstein distance.
        """
        test_nb = len(test_data)
        results = [[] for i in range(test_nb)]
        if WS_perf:
            WS_results = [[] for i in range(test_nb)]
        for i in range(nb_runs):
            self.reset_params()
            print(f'Processing model {i+1} of {nb_runs}.')
            perf = learn_function(verbose=2, **kwargs)
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

    def run_tests(self, testdata, test_function, **kwargs):
        """
        A helper function that runs multiple sets of test data through a `test_function`.
        :param testdata: list containing test data.
        :param test_function: function used for testing data. Should be `self.test_potential` or `self.test_ws`.
        :return: a list containing the return values of `test_function` for each of the test sets in `testdata`.
        """
        results = []
        for d in testdata:
            results.append(test_function(d, **kwargs))
        return results

    def predict(self, a, b):
        """
        Concatenates input distributions `a` and `b` to compute the network's output.
        :return: `self.net` evaluated at `a` and `b`.
        """
        self.net.eval()
        with torch.no_grad():
            x = torch.cat((a, b), 1).to(device)
            out = self.net(x)
        return out

    def test_potential(self, data, loss_function = F.mse_loss, relative = False, scale = 1):
        '''
        Tests the network on test data 'data'.
        :param data: data used for testing. Should be a list with each item being a dictionary with keys `d1`, `d2` and `u` which contain the two distributions and the dual potential as two-dimensional tensors.
        :param relative: if True, computes the error relative to the distributions.
        :param scale: A scaling factor to scale the error by. Can e.g. be used to rescale a cost function to the cost in the unit square.
        :return: average `loss_function` error on the dual potential.
        '''
        self.net.eval()
        l = 0
        with torch.no_grad():
            for batch in data:
                if not relative:
                    loss = loss_function(self.predict(batch['d1'], batch['d2']), batch['u'])
                else:
                    approx_pot = self.predict(batch['d1'], batch['d2'])*batch['d1']*(batch['d1'].size(1))
                    true_pot = batch['u']*batch['d1']*(batch['d1'].size(1))
                    loss = loss_function(approx_pot, true_pot)
                l += loss.item()
        l /= len(data)
        return l

    def test_ws(self, data, loss_function = F.mse_loss, rel = False, return_ws = False, cost_norm = 1):
        '''
        Tests the network on test data 'data'.
        :param data: data used for testing. Should be a list with each item being a dictionary with keys `d1`, `d2` and `u` which contain the two distributions and the dual potential as two-dimensional tensors.
        :param loss_function: loss function.
        :param rel: if True, returns the relative error, computed as `[\sum (a_i-b_i)^z]/[\sum b_i^z]`, where a is the prediction and b the ground truth.
        :param return_ws: if True, additionally returns the approximated WS distances and the ground truth.
        :param cost_norm: a constant to multiply the cost function with. Can be used to normalize the cost function w.r.t. the dimension.
        :return: average `loss_function` error on the squared Wasserstein-2 distance (i.e. the OT cost). If `return_ws`==True, at second position also returns a list containing 2-tuples, where the first entry corresponds to the estimated Wasserstein distance and the second entry to the ground truth.
        '''
        self.net.eval()
        l = 0
        if return_ws:
            ws_list = []
        with torch.no_grad():
            for batch in data:
                u = self.predict(batch['d1'], batch['d2'])
                v = compute_c_transform(cost_norm*self.costmatrix, u)
                ws_guess = compute_dual(batch['d1'], batch['d2'], u, v)
                if not rel:
                    loss = loss_function(ws_guess, batch['cost'])
                else:
                    zeros = torch.zeros(batch['cost'].size()).to(device)
                    rel_error = (ws_guess - batch['cost'])/batch['cost']
                    loss = loss_function(zeros, rel_error)
                l += loss.item()
                if return_ws:
                    ws_list.append((ws_guess, batch['cost']))
        l /= len(data)
        if return_ws:
            return (l, ws_list)
        return l



if __name__ == '__main__':
    d = DualApproximator(model='Models/net100k.pt', gen_model='Models/gen100k.pt')
