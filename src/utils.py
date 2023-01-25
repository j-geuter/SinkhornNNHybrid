import matplotlib.pyplot as plt
import math
import ot
from tqdm import tqdm
import torch
from torch.optim import Adam
import numpy as np
from scipy.stats import t as t_scipy

from src.costmatrix import euclidean_cost_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCALE = 1
SMALL = SCALE*8
MEDIUM = SCALE*12
BIG = SCALE*16

plt.rc('font', size=MEDIUM)         # controls default text sizes
plt.rc('axes', titlesize=BIG)       # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM)   # legend fontsize
plt.rc('figure', titlesize=BIG)     # fontsize of the figure title


def visualize_data(data, row = None, column = None):
    """
    Visualizes data as black and white images. To visualize a single image, use visualize_image instead.
    :param data: list or array of size [n,dim] or [n, l, l], where n is the number of images to be visualized, and dim is the number of pixels per image (l is the height/width).
    :param row: optional parameter determining the number of rows.
    :param column: optimal parameter determining the number of columns.
    :return: None.
    """
    if len(data.size()) == 2:
        length = int(math.sqrt(data.size(1)))
        data = data.reshape(data.size(0), length, length)
    if not row:
        fig, axes = plt.subplots(len(data), figsize=(8,8))

    else:
        fig, axes = plt.subplots(row, column, figsize=(8,8))
    if not row:
        for i,ax in enumerate(axes):
            ax.imshow(data[i], cmap='Greys')
    else:
        if row == 1:
            axes = [axes]
        for j in range(row):
            for i,ax in enumerate(axes[j]):
                ax.imshow(data[j*column+i], cmap='Greys')
    fig.show()

def visualize_image(image):
    """
    Visualizes an single data sample.
    :param image: A square image; either a two-dimensional tensor, or a one-dimensional tensor where all points are concatenated.
    """
    if image.dim() == 1:
        length = int(math.sqrt(image.size(0)))
        image = image.reshape(length, length)
    plt.imshow(image, cmap='Greys')
    plt.show()

def check_accuracy(batch, method, reg, costmatrix = None):
    """
    Checks the accuracy of a `method`-Sinkhorn solver from the POT package compared to a batch of data.
    :param method: method of the Sinkhorn solver.
    :param reg: regularizer of the Sinkhorn solver.
    :param costmatrix: cost matrix to be used for OT computations. If 'None', defaults to squared Euclidean distance.
    :return: Average error.
    """
    if costmatrix == None:
        length = int(math.sqrt(batch['d1'].size()[1]))
        costmatrix = euclidean_cost_matrix(length, length, 2, True)
    n_sample = batch['d1'].size()[0]
    err = 0
    for i in tqdm(range(n_sample)):
        ws_sinkhorn = ot.sinkhorn2(batch['d1'][i], batch['d2'][i], costmatrix, reg, method=method)
        err += abs(ws_sinkhorn - batch['cost'][i]).item()
    err /= n_sample
    return err

def compute_c_transform(cost, sample, zero_sum = False):
    """
    Computes the c transform of 'sample' w.r.t. to cost matrix 'cost'. Supports multiple samples.
    :param cost: Two-dimensional tensor. Cost matrix.
    :param sample: Two-dimensional tensor. Sample(s). If just one sample is given, it needs to have an empty first dimension.
    :param zero_sum: if True, deducts a constant from each transform such that it sums to zero.
    :return: Two-dimensional tensor. c-transforms of `sample`.
    """
    lamext=sample.reshape(len(sample),len(sample[0]),1).expand(len(sample),len(sample[0]),len(sample[0])).transpose(2,1)
    lamstar=(cost-lamext).amin(dim=2).float()
    del lamext
    torch.cuda.empty_cache()
    if zero_sum:
        lamstar = lamstar - lamstar.sum(1)[:, None]/lamstar.size(1)
    return lamstar

def compute_dual(alpha, beta, u, v = None, c = None):
    """
    Computes the dual value of the OT problem as `\int u \, d\alpha + \int v \, d\beta`. Supports multiple samples in 'u' and 'v'.
    Either u or v can be None, in which case it is replaced by the c-transform of the other.
    :param alpha: Two-dimensional tensor. Source distribution(s).
    :param beta: Two-dimensional tensor. Target distribution(s).
    :param u: Two-dimensional tensor or None. First dual potential(s).
    :param v: Two-dimensional tensor or None. Second dual potential(s).
    :param c: Optional cost matrix.
    :return: Two-dimensional tensor. Values of the dual problem at (u,v).
    """
    if u == None:
        if c == None:
            l = int(math.sqrt(v.size(1)))
            c = euclidean_cost_matrix(l, l, 2, True)
        u = compute_c_transform(c, v)
    elif v == None:
        if c == None:
            l = int(math.sqrt(u.size(1)))
            c = euclidean_cost_matrix(l, l, 2, True)
        v = compute_c_transform(c, u)
    values = torch.sum(alpha*u, dim=1) + torch.sum(beta*v, dim=1)
    return values[None, :].T

def hilbert_metric(u, v):
    """
    Computes Hilbert projective metric of the Sinkhorn iterates on a batch of samples of dual potentials, which cancels out the logarithm. Note that for simplicity, we leave out the scaling factor 1/epsilon.
    :param u: two-dimensional tensor.
    :param v: two-dimensional tensor.
    """
    return (max_norm(u - v).sum())

def max_norm(x):
    """
    Auxiliary function for `hilbert_metric`.
    :param x: two-dimensional tensor.
    """
    return (x.max(1)[0] - x.min(1)[0])



def compute_mean_conf(data, conf):
    """
    Given a list of lists of datapoints, computes their mean values and a confidence interval.
    Each item in `data` corresponds to one set of samples taken over time, e.g. the length of `data` equals the sample size at each point in time and the length of each item in `data` equals the number of time steps that data was collected at.
    :param data: list of lists or list-like objects.
    :param conf: desired confidence between 0 and 1.
    :return: A 3-tuple. The first entry corresponds to the lower bound on the confidence interval, the second one to the mean, the third one to the upper bound.
    """
    n = len(data)
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    x     = np.array(data)
    means = x.mean(0)
    stds  = x.std(0)
    dof   = len(x) - 1
    t     = np.abs(t_scipy.ppf((1 - conf)/2, dof))
    lowers = means - stds * t/np.sqrt(len(x))
    uppers = means + stds * t/np.sqrt(len(x))
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    return(lowers, means, uppers)


def plot(
            y,
            x = None,
            labels = None,
            x_label = '',
            y_label = '',
            titles = None,
            separate_plots = None,
            rows = None,
            columns = None,
            slice = None,
            scale_y = 1
        ):
    """
    Plots data.
    :param y: List of lists, each containing data of one type. All lists in y have to be of the same length.
    :param x: List of lists. Values for x-axis. If `None`, defaults to [0, 1, 2, ..., len(y[0])] for each list in y.
    :param labels: List of labels for each list in y.
    :param x_label: String. Label on the x-axis.
    :param y_label: String. Label on the y-axis.
    :param titles: optional titles for each plot.
    :param separate_plots: optional parameter to split data into separate plots. If given, this should be a list of lists, each tuple containing the data indices for a plot.
    :param rows: number of rows for subplots. If None, all subplots will be in one row.
    :param columns: number of columns for subplots.
    :param slice: optional parameter with which one can determine a slice of each element in y to be used instead. If given, this is a tuple of two ints indicating the slice.
    :param scale_y: scales all `y` values by `scale_y`.
    :return: None.
    """
    if x == None:
        x = [i for i in range(len(y[0]))]
    elif isinstance(x, int):
        x = [i*x/(len(y[0])-1) for i in range(len(y[0]))]
    if not isinstance(x[0], list): # if x is just a single list, it needs to be cast to contain one list for each item in y.
        x = [x for i in range(len(y))]
    if slice != None:
        for i in range(len(y)):
            x[i] = x[i][slice[0]:slice[1]]
            y[i] = y[i][slice[0]:slice[1]]
    if separate_plots == None:
        for i in range(len(y)):
            if labels != None:
                plt.plot(x[i], scale_y*np.array(y[i]), label=labels[i])
            else:
                plt.plot(x[i], scale_y*np.array(y[i]))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
    else:
        nb_plots = len(separate_plots)
        if rows == None:
            rows = 1
            columns = nb_plots
        fig, axes = plt.subplots(rows, columns, sharex='col', figsize=(8,8))
        if rows == 1:
            axes = [axes]
        if columns == 1:
            axes = [[ax] for ax in axes]
        if titles:
            titles = iter(titles)
        for r in range(rows):
            for i, ax in enumerate(axes[r]):
                colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(separate_plots[r*columns+i]))))
                for j in separate_plots[r*columns+i]:
                    color = next(colors)
                    if labels != None:
                        ax.plot(x[i], scale_y*np.array(y[j]), label=labels[j], color=color)
                    else:
                        ax.plot(x[i], scale_y*np.array(y[j]), color=color)
                ax.set_ylim(bottom=0)
                if r == rows-1:
                    ax.set_xlabel(x_label)
                if i == 0:
                    ax.set_ylabel(y_label)
                if titles:
                    ax.set_title(next(titles))
                if labels != None:
                    ax.legend()
        fig.show()


def plot_conf(
                x,
                y,
                labels,
                x_label,
                y_label,
                titles = None,
                separate_plots = None,
                separate_legends = False,
                collect_legends = False,
                rows = None,
                columns = None,
                slice = None,
                scale_y = 1
            ):
    """
    Plots data containing average values alongside their confidence intervals as shaded areas.
    :param x: list of lists or arrays containing x values. Can also be an integer, in which case it is converted to a list of numbers interpolating between 0 and that integer.
    :param y: list of lists. Each contains three lists, the first corresponding to the lower confidence bound, the second to the average, the third to the upper confidence bound.
    :param labels: list of same length as `y` containing the labels for each plot.
    :param x_label: label for x axis.
    :param y_label: label for y axis.
    :param titles: optional titles for each plot.
    :param separate_plots: optional parameter to split data into separate plots. If given, this should be a list of lists, each tuple containing the data indices for a plot.
    :param seperate_legends: if True, creates a unique legend for each subplot, otherwise creates a single legend for all plots.
    :param collect_legends: if True, collects all legends for all subplots for the legend (only if separate_legends==False). Otherwise, assumes legends for subplots are identical and uses only one subplot's legend.
    :param rows: number of rows for subplots. If None, all subplots will be in one row.
    :param columns: number of columns for subplots.
    :param slice: optional parameter with which one can determine a slice of each element in y to be used instead. If given, this is a tuple of two ints indicating the slice.
    :param scale_y: scales all `y` values by `scale_y`, e.g. to rescale values to a cost function in the unit square.
    :return: None.
    """
    if isinstance(x, int):
        x = [i*x/(len(y[0][0])-1) for i in range(len(y[0][0]))]
    if not isinstance(x[0], list) and not isinstance(x[0], np.ndarray):
        x = [x for i in range(len(y))]
    n = len(y)
    if slice != None:
        for i in range(n):
            x[i] = x[i][slice[0]:slice[1]]
            for j in range(3):
                y[i] = list(y[i])
                y[i][j] = y[i][j][slice[0]:slice[1]]
    if not separate_plots:
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
        for i in range(n):
            color = next(colors)
            plt.fill_between(x[i], scale_y*np.array(y[i][0]), scale_y*np.array(y[i][2]), color=color/3, linewidth=0)
            plt.plot(x[i], scale_y*np.array(y[i][1]), label=labels[i], color=color)
        plt.ylim(bottom=0)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if titles:
            plt.title(titles)
        plt.legend()
        plt.show()
    else:
        nb_plots = len(separate_plots)
        if rows == None:
            rows = 1
            columns = nb_plots
        fig, axes = plt.subplots(rows, columns, sharex='col', figsize=(8,8))
        if rows == 1:
            axes = [axes]
        if columns == 1:
            axes = [[ax] for ax in axes]
        if titles:
            titles = iter(titles)
        for r in range(rows):
            for i, ax in enumerate(axes[r]):
                colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(separate_plots[r*columns+i]))))
                for j in separate_plots[r*columns+i]:
                    color = next(colors)
                    ax.fill_between(x[i], scale_y*np.array(y[j][0]), scale_y*np.array(y[j][2]), color=color/3, linewidth=0)
                    ax.plot(x[i], scale_y*np.array(y[j][1]), label=labels[j], color=color)
                ax.set_ylim(bottom=0)
                if r == rows-1:
                    ax.set_xlabel(x_label)
                if i == 0:
                    ax.set_ylabel(y_label)
                if titles:
                    ax.set_title(next(titles))
                if separate_legends:
                    ax.legend()
        if not separate_legends:
            if collect_legends:
                handles_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                handles, labels = [sum(lol, []) for lol in zip(*handles_labels)]
                fig.legend(handles, labels)
            else:
                handles, labels = fig.axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=len(labels))
        fig.show()

def interpolate_barycenters(
                        samples,
                        net,
                        iters = 30,
                        n_interpol = 5
                    ):
    """
    Creates a visualization of Wasserstein barycenter interpolations between four given samples using gradient descent and a network only without Sinkhorn.
    :param samples: an iterable containing the four samples. Each sample is either a one- or two-dimensional tensor of size [dim] or [height, width].
    :param net: network used for transport computations.
    :param iters: number of gradient steps.
    :param n_interpol: number of interpolations. Final output will contain `n_interpol`**2 images. CURRENTLY ONLY IMPLEMENTED FOR N_INTERPOL=5.
    """
    net.doubletransform = True
    if samples[0].dim() == 3:
        for i in range(4):
            l = samples[0].size(1)
            samples[i] = samples[i].reshape(l**2)
    dim = samples[0].size(0)
    cost_matrix = euclidean_cost_matrix(dim, dim, 2, True).to(device)
    images = [[0 for j in range(n_interpol)] for i in range(n_interpol)] # will contain output images for the visualization.
    images[0][0] = samples[0]
    images[-1][0] = samples[1]
    images[0][-1] = samples[2]
    images[-1][-1] = samples[3]
    for (i,j) in [(2,0), (2,4)]:
        images[i][j] = compute_barycenter([images[i-2][j], images[i+2],[j]], net, cost_matrix, iters)
    for (i,j) in [(0,2), (4,2)]:
        images[i][j] = compute_barycenter([images[i][j-2], images[i],[j+2]], net, cost_matrix, iters)
    images[2][2] = compute_barycenter([images[0][0], images[0][4], images[4][0], images[4][4]], net, cost_matrix, iters)
    for (i,j) in [(1,1), (1,3), (3,1), (3,3)]:
        images[i][j] = compute_barycenter([images[i-1][j-1], images[i+1][j-1], images[i-1][j+1], images[i+1][j+1]], net, cost_matrix, iters)
    for (i,j) in [(1,0), (3,0), (1,4), (2,4)]:
        images[i][j] = compute_barycenter([images[i-1][j], images[i+1],[j]], net, cost_matrix, iters)
    for (i,j) in [(0,1), (0,3), (4,1), (4,3)]:
        images[i][j] = compute_barycenter([images[i][j-1], images[i],[j+1]], net, cost_matrix, iters)
    for (i,j) in [(1,2), (2,1), (2,3), (3,2)]:
        images[i][j] = compute_barycenter([images[i-1][j], images[i+1][j], images[i][j+1], images[i][j-1]], net, cost_matrix, iters)

    fig, axes = plt.subplots(n_interpol, n_interpol)
    for i in range(n_interpol):
        for j in range(n_interpol):
            axes[i][j].imshow(images[i][j], cmap='Greys')
    fig.show()


def compute_barycenter(
                        samples,
                        net,
                        cost,
                        iters = 30,
                        prints = True,
                        start = None
                        ):
    """
    Computes a barycenter of `samples` using gradient descent and a network only without Sinkhorn.
    :param samples: an iterable containing the four samples. Each sample is a one-dimensional tensor.
    :param net: network used for transport computations.
    :param cost: cost matrix.
    :param iters: number of gradient steps.
    :param prints: if True, prints the loss in every iteration.
    :param start: if given, uses `start` as a starting distribution.
    :return: barycenter of input measures. One-dimensional tensor.
    """
    n = len(samples)
    archetypes = torch.cat(([sample[None, :] for sample in samples]), 0).to(device)
    if start == None:
        preBarycenter = torch.ones((784)).to(device) #initialize the variable
    else:
        preBarycenter = start.to(device)
    preBarycenter.requires_grad = True #making the variable gradiable
    optimizer = Adam([preBarycenter], lr=0.4)
    lamb = lambda epoch: 0.9 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamb)
    for k in range(iters):
      barycenter = torch.softmax(preBarycenter,dim=0) # Transforms the variable to become a probability measure
      barycenters = torch.cat(([barycenter[None, :] for i in range(n)]), 0)
      nninput = torch.cat((barycenters, archetypes), 1) # Makes the input for the NN
      optimizer.zero_grad()
      loss = torch.sum(compute_dual(barycenters, archetypes, net(nninput), v=None, c=cost))  #Loss is the average of the dual formula for the output of the network
      if prints:
          print("loss is:", loss, " k=", k)
      loss.backward()
      optimizer.step() # makes a gradient step on preBarycenter
      scheduler.step()
    net.doubletransform = False
    barycenter = torch.softmax(preBarycenter, dim=0)
    return barycenter

def visualize_barycenters(
                            samples,
                            iters,
                            net,
                            cost,
                            rows,
                            columns,
                            start = None
                            ):
    """
    Visualizes barycenters for a given list of sample batches.
    :param samples: list containing a batch of samples at each position. Each batch is a two-dimensional tensor containing the samples. Computes a barycenter for each batch.
    :param iters: number of gradient descent steps.
    :param net: network to use for barycenter computation.
    :param cost: cost matrix.
    :param rows: number of rows of visualization.
    :param columns: number of columns of visualization.
    :param start: if None, no `start` argument is passed to `compute_barycenter`. If set to 'single', uses each batch's first sample as `start`. If set to 'avg', uses each batch's average as `start`.
    """
    barycenters = torch.zeros((rows*columns, samples[0][0].size(0)))
    for i in range(len(samples)):
        if start == 'single':
            start_i = samples[i][0]
        elif start == 'avg':
            start_i = samples[i].sum(0)/len(samples[i])
        else:
            start_i = None
        barycenters[i] = compute_barycenter(samples[i], net, cost, iters=iters, prints=False, start=start_i).detach().to('cpu')
    visualize_data(barycenters, rows, columns)
