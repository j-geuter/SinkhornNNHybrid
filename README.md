# SinkhornNNHybrid
Welcome to the SinkhornNNHybrid repository! This repo provides the PyTorch implementation for the thesis 'A Sinkhorn-NN Hybrid Algorithm for Optimal
Transport', which is available as `Thesis.pdf`.
The code is structured in six files, `DualOTComputation`, `networks`, `utils`, `sinkhorn`, `costmatrix` and `datacreation`. The main files you'll need reproduce the results from the thesis are `DualOTComputation` and `sinkhorn`, and `utils` contains some useful functions that let you visualize or plot data. To generate training and testing data first, you'll need the
`datacreation` file. The `costmatrix` file contains a function to create cost matrices based on the Euclidean distance, and `networks` the neural network class; you won't need to actively use either of these files unless you want to define your own cost function or network structure, in which case it is easiest to alter the `euclidean_cost_matrix` function and the `FCNN` class therein.
The `requirements.txt` file lists all dependencies and their versions.  
The project is CUDA compatible.

## Code Overview
We will first have a quick look at the six files containing all code, before going over how to replicate the results from the thesis in the next section.

### costmatrix
Euclidean distance cost matrices can be constructed using the `euclidean_cost_matrix` function which takes as input the two image dimensions and an 'exponent' parameter, which is usually set to $2$ corresponding to the squared Euclidean distance. A fourth boolean parameter `tens` lets you choose whether the matrix should be a `torch.tensor` object or a `numpy.array`. In most cases, you will call this function as `euclidean_cost_matrix(28, 28, 2, True)`.

### datacreation
This file contains various functions that let you create and save datasets for training and testing. Usually, data `d` will come in the form of a list, where each item is a dictionary
with keywords `d1` and `d2` denoting batches of distributions, `cost` denoting the respective transport costs, and `u` denoting the dual potentials. A sample corresponds to
`(d['d1'][i], d['d1'][i], d['u'][i], d['cost'][i])`. The `load_data` and `save_data` functions are simple functions that let you open
and save data files using `pickle`. Another important function in this file is `compute_c_transform` which lets you compute the $c$-transform of a potential `sample`. This function
also needs as input the cost matrix `cost` and supports multiple samples at once.   
Most important for generating training data similar to the data used in the thesis is the
`generate_simple_data` function, which takes as input a `file_name` which determines the location where the data set will be saved, the side length `length` of the distributions
(e.g. $28$), number of samples in the file `n_samples` (note that setting `n_samples` too high might cause memory overload; a good approach is saving $100,000$ samples per file),
and a keyword argument `mult` which corresponds to the parameter $k_1$ from the thesis. It is also possible to pass a keyword argument `sink=True` in order to use the Sinkhorn algorithm for sample generation instead. This function can also be used for randomly generated test data.
To produce test data sets which are not random, but originate from data sets such as MNIST, use the `generate_dataset_data` function. It takes as input a `name` (location where
the data will be saved), a parameter `n_files` which controls how many data sets will be generated (and can most often be set to $1$), and, importantly, a `dataloader` parameter
which is a `torch.utils.data.DataLoader` object and can be wrapped around a `torchvision.datasets` data set such as `torchvision.datasets.MNIST`.  
Example (assuming there exists a subfolder `Data`):  

```python
generate_simple_data('Data/train_data.py', length=28, mult=3)
dataloader = DataLoader(MNIST(root='./Data/mnist_dataset', train=False, download=True, transform=torchvision.transforms.ToTensor()), batch_size=10000)
generate_dataset_data('Data/test_data.py', dataloader=dataloader, train=False, n_samples=10000)
```

### DualOTComputation
This file mainly contains the `DualApproximator` class which lets you create and train a network.
It takes as input the side length `length` of the image distributions and a network class, which defaults to the class `FCNN` from the file `networks`.  
The network learns using the function `learn_potential` which takes as input the file name of a training data file. This function learns using an error on the dual potential of the OT
problem. One can, however, also compute the OT value from this potential and use an error on the optimum (instead of the potential, i.e. the _optimizer_); this can be done
with `learn_ws`. The respective testing functions are `test_potential` and `test_ws`.  
Note that when run directly, `DualOTComputations` prompts you to enter an 'input width of data' and automatically creates a `DualApproximator` object `d`.  
Example:  

```python
d = DualApproximator(28)
testdata = load_data('Data/test_data.py') # loading the file `test_data` from the previous example

d.learn_potential('Data/train_data.py')
d.test_potential(testdata)

d.reset_params() # resets all trainable parameters to their default initialization
d.learn_ws('Data/train_data.py') # learn with a WS loss instead. learn_ws defaults to loss function `loss_max_ws` which does not need dual potentials in training data
d.test_ws(testdata)
```

### sinkhorn
This file contains the Sinkhorn algorithm and many functions accompanying it. The algorithm itself is implemented in the `sinkhorn` function, with the log domain version available
through the `log_sinkhorn` function.  
`sinkhorn` takes as input distributions `mu` and `nu`, which can be one-dimensional if just one sample is passed, or two-dimensional for multiple samples which are processed in a
parallelized fashion. An optional initialization can be passed via the `start` keyword argument. In case this start vector might contain very small or very large values,
they can be controlled using the `min_start` and `max_start` keyword arguments, which set values that are too small or too large equal to these thresholds.  
It also needs a cost matrix `C`, which in the case for euclidean distances can be produced using the `euclidean_cost_matrix` function in `datacreation`. The `compare_iterations`,
`compare_time` and `compare_accuracy` functions offer various functionality to compare the performance of different initialization schemes for the Sinkhorn algorithm.
They take as input test data `data` and initialization schemes `inits` which is a list of initializations. For the default initialization, set one of the items to `None`. For
using a `DualApproximator`'s network, simply pass the `net` attribute of the `DualApproximator` object.  
Example (with `d` as before):  

```python
from datacreation import load_data
testdata = load_data('Data/test_data.py')
c = euclidean_cost_matrix(28, 28, 2, True)
eps = .2
sinkhorn(testdata[0]['d1'][:5], testdata[0]['d2'][:5], c, eps, max_iter=1000, log=True) # computes the OT cost for the first five samples in test data. With `log=True`, also returns
                                                                                        # the transport plan, the dual potentials of the OT dual problem, and the marginal constraint
                                                                                        # violations
input = d.net(torch.cat((testdata[0]['d1'][:5], testdata[0]['d2'][:5]), 1))
input = torch.exp(input)/eps
sinkhorn(testdata[0]['d1'][:5], testdata[0]['d2'][:5], c, eps, max_iter=500, start=input, min_start=1e-35, max_start=1e35) # uses network's prediction for initialization
compare_iterations(testdata[0], [d.net, None], ['network', 'default'], accs=['WS'], max_iter=500, eps=eps, min_start=1e-35, max_start=1e35) # plots accuracy for default and network initialization
                                                                                                                               # w.r.t. the number of iterations
```

### utils
This file contains various functions for creating different types of plots from data.
`visualize_data` lets you visualize samples from your training and testing data, taking as input a tensor or array `data` of size $n\times dim$ or $n\times l\times l$ with $n$
being the number of samples, $dim$ the dimension of the samples, and $l$ the side length.  
`compute_dual` takes as input distributions `alpha` and `beta` and dual potentials `u` and `v` and computes the respective dual problem's value. All inputs need to be of dimension two and multiple samples can be processed at once.  
`compute_mean_conf` lets you compute a confidence interval from a data series in time. It takes as input `data`, which is a list of lists or list-like objects. Each item in `data` corresponds to one set of samples taken over time, e.g. the length of `data` equals the sample size at each point in time and the length of each item in `data` equals the number of time
steps that data was collected at. The function also takes as input a parameter `conf`, the desired confidence, i.e. this can e.g. be set to `.95` for a $95$ percent confidence interval.  
`plot` is a simple function that lets you plot multiple types of data at once, collected over the same `x`-values.  
`plot_conf` lets you plot various data time series alongside their confidence intervals, which will be shown as shaded areas around the plots. It takes as input `x`-values (which can also equal an integer in which case the x-values are interpolating between $0$ and that integer) and a list `y` where each item is a three-element list, where the first one corresponds to the lower values of
a confidence interval, the second one to the mean values, and the third one to the upper values of a confidence interval. This corresponds to the output generated by `compute_mean_conf`. It also takes as input `labels` corresponding to the labels of each item in `y`, `x_label` and `y_label` corresponding to the labels on the x- and y-axis, optional `titles` for each plot, and an optional parameter `separate_plots`. If this is not passed to the function, all `y` items will appear in a single plot. `separate_plots` can be a list of tuples, where each tuple corresponds to the indices from `y` which should appear in the same plot. If, e.g., the first three items should appear in one plot, and the fourth item in a second plot, set `separete_plots=[(0,1,2), (3)]`. In case `separete_plots` is given, the optional `rows` and `columns` parameters let you control how many rows and columns the plots should be placed in.  
Example (with `d` and `testdata` as before):  

```python
visualize_data(testdata[0]['d1'][:5])
f = d.net(torch.cat((testdata[0]['d1'][:20], testdata[0]['d2'][:20]), 1))
c = euclidean_cost_matrix(28, 28, 2, True)
g = compute_c_transform(c, f)
f = compute_c_transform(c, g) # sets f to its double-c-transform. Can also be achieved directly by setting d.net.doubletransform=True
dual_approx = compute_dual(testdata[0]['d1'][:20], testdata[0]['d2'][:20], f, g)
print((dual_approx - testdata[0]['cost'][:20]).abs().mean()) # average error on the dual OT problem value
```

### networks
This file contains the network class `FCNN`, which serves as the `net` attribute of the `DualApproximator` class in the `DualOTComputation` file and approximates a dual potential, given two input distributions. Each `FCNN` object has attributes `symmetry`, `doubletransform` and `zerosum` which default to `False`. If `symmetry` is set to `True`, the network will compute an output that is symmetric in the two input distributions. If `doubletransform` is set to `True`, it will instead compute the double- $c$-transform of the original output, which can be thought of as a $c$-concave approximation of the original output. If `zerosum` is set to `True`, it will enforce zero-sum outputs by subtracting a constant (note that this does not change the value of the dual problem as it is invariant under adding or subtracting constants to the potential).

## Results from the Thesis
The results from the thesis can be reproduced as follows.

### Data
Training data is generated using the `generate_simple_data` function, which was used to create ten datasets with 100,000 samples each (as one file with a million samples might cause memory issues). The test datasets were created using `generate_simple_data` and `generate_dataset_data`. Assuming the `datacreation.py` environment was loaded and that the Quick, Draw!
dataset was downloaded into a folder `./Data/QuickDraw` in the `.npy` version (see [this link](https://github.com/googlecreativelab/quickdraw-dataset) on more details):

```python
for i in range(10):
  generate_simple_data(f'Data/training_file_{i}.py', length=28, mult=3)
generate_simple_data('Data/test_file_0.py', length=28, mult=3, n_samples=10000)
generate_dataset_data('Data/test_file_1.py', dataloader=DataLoader(MNIST(root='./Data/mnist_dataset', train=False, download=True, transform=torchvision.transforms.ToTensor()), batch_size=10000), train=False, n_samples=10000)
generate_dataset_data('Data/test_file_2.py', dataloader=DataLoader(CIFAR10(root='./Data/CIFAR', train=False, download=True, transform=torchvision.transforms.ToTensor()), batch_size=10000), train=False, n_samples=10000)
teddies = load_files_quickdraw(categories=1, per_category=10000, rand=False, names=['teddy-bear'], dir='./Data/QuickDraw/')['teddy-bear']
teddies = torch.tensor(teddies)
generate_dataset_data('Data/test_file_3.py', train=False, n_samples=10000, data=teddies)
```

### Loss on Wasserstein Distance
To reproduce the results where a loss on the dual potential was compared to a loss on the Wasserstein distance for training the network, use the following code (assuming the `DualOTComputation.py` environment was loaded; note that when run directly, `DualOTComputations` prompts you to enter an 'input width of data' and automatically creates a `DualApproximator` object `d`):

```python
testdata = [load_data(f'Data/test_file_{i}.py') for i in range(4)]
d = DualApproximator(28)
pot_perf = d.average_performance('Data/training_file_0.py', d.learn_potential, test_data=testdata) # computes the average error on the dual potential across 10 instances of the network
WS_perf  = d.average_performance('Data/training_file_0.py', d.learn_ws, test_data=testdata) # computes the average error on the Wasserstein distance across 10 instances
```

Note that calling `average_performance` resets all learnable parameters of the `net` attribute, e.g. deletes all learning progress!
It returns, for each test dataset in `test_data`, three arrays, where the second one corresponds to the average performance over 10 instances of the network, measured at $30$ points in time during training (can be adjusted with the `num_tests` parameter).
Now the results can be plotted using the `plot_conf` function in `utils.py`:

```python
from utils import plot_conf
plot_conf(100000, pot_perf+WS_perf, ['pot']*4+['WS']*4, 'training samples', 'MSE error on potential', titles=['random','teddies', 'MNIST', 'CIFAR'], separate_plots=[[0,4], [2,5], [3,6], [4,7]], rows=2, columns=2)
```

### Train on MNIST Data
To reproduce the results where training on our regular training data was compared to training on MNIST data, produce an MNIST training dataset (assuming the `datacreation.py` environment was loaded):

```python
generate_dataset_data('Data/training_MNIST.py', dataloader=DataLoader(MNIST(root='./Data/mnist_dataset', train=True, download=True, transform=torchvision.transforms.ToTensor()), batch_size=60000), train=True, n_samples=100000)
```

and use this for training using `DualApproximator.average_performance` in the same way as in the previous example.

### Train Network
To train a network on one million samples as the network used in our main experiments, run the following (assuming the `DualOTComputation.py` environment was loaded):

```python
lr = [0.005 - i*0.0005 for i in range(10)]
d = DualApproximator(28)
d.learn_multiple_files('Data/training_file', 0, 9, d.learn_potential, lr=lr) # via the `verbose`, `num_tests` and `test_data` parameters, performance on test data can be collected
```

If you want to train for longer, you can do so using the `meta_epochs` parameter of the `learn_multiple_files` function, which allows you to go over the training files multiple times. If you do so, during each `meta_epoch` all files will be used once for training, and every time a file is loaded its samples are shuffled at random before used for training.

### Results
Now that the network has been trained, most of the results can be computed from a single call of the `compare_iterations` in `sinkhorn.py` for each test dataset. Assuming that the `sinkhorn.py` environment was loaded and `d` as before:

```python
testdata = [load_data(f'Data/test_file_{i}.py') for i in range(4)]
results = []
d.net.eval()
for t in testdata:
  results.append(compare_iterations(t[:10], [None, d.net], ['default', 'net'], max_iter=2500, eps=.2, min_start=1e-35, max_start=1e35, plot=False, timeit=True))
```
Now `r = results[i]` contains the results for the respective test dataset. `r[0]['WS']` and `r[0]['marg']` contain information on the Wasserstein distance and dual potential errors;
`r[1]` on the time it took for computations. In each of these locations, the values for the default initialization can be accessed at position `[0]` and for the network at position `[1]`. This will give you a 3-tuple, where each item is an array, the first one being the lower bound on the $95\%$ confidence interval, the second one the mean, and the third one the upper bound on the confidence interval. So for example, the upper bound on the $95\%$ confidence interval of the marginal constraint error of the default initialization for the first test dataset can be found at `results[0][0]['marg'][0][2]`. The results can be visualized with the `plot_conf` function from `utils.py` again, so for example for visualizing the errors on the marginal constraint alongside the confidence intervals across all four test datasets run:

```python
from utils import plot_conf
plot_conf(2500, results[0][0]['marg']+results[1][0]['marg']+results[2][0]['marg']+results[3][0]['marg'], ['default', 'net']*4, 'number of iterations', 'marginal constraint violation', titles=['random', 'teddies', 'MNIST', 'CIFAR'], separate_plots=[[0,1], [2,3], [4,5], [6,7]], rows=2, columns=2)
```
