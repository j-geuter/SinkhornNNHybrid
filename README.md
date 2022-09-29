# SinkhornNNHybrid
Welcome to the SinkhornNNHybrid repository! This repo provides the PyTorch implementation for the thesis 'A Sinkhorn-NN Hybrid Algorithm for Optimal
Transport'.  
The code is structured in five files, `DualOTComputation`, `networks`, `utils`, `sinkhorn` and `datacreation`. The main files you'll need reproduce the results from the thesis are `DualOTComputation` and `sinkhorn`, and `utils` contains some useful functions that let you visualize or plot data. To generate training and testing data first, you'll need the
`datacreation` file.

## datacreation
This file contains various functions that let you create and save datasets for training and testing. Usually, data `d` will come in the form of a list, where each item is a dictionary
with keywords `d1` and `d2` denoting batches of distributions, `cost` denoting the respective transport costs, and `u` denoting the dual potentials. A sample corresponds to
`(d['d1'][i], d['d1'][i], d['u'][i], d['cost'][i])`. The `load_data` and `save_data` functions are simple functions that let you open
and save data files using `pickle`. Another important function in this file is `compute_c_transform` which lets you compute the $c$-transform of a potential `sample`. This function
also needs as input the cost matrix `cost` and supports multiple samples at once. Euclidean distance cost matrices can be constructed using the `euclidean_cost_matrix` function which takes as input the two image dimensions and an 'exponent' parameter, which is usually set to $2$ corresponding to the squared Euclidean distance. A fourth boolean parameter `tens` lets you choose whether the matrix should be a `torch.tensor` object or a `numpy.array`. In most cases, you will call this function as `euclidean_cost_matrix(28, 28, 2, True)`.  
Most important for generating training data similar to the data used in the thesis is the
`generate_simple_data` function, which takes as input a `file_name` which determines the location where the data set will be saved, the side length `length` of the distributions
(e.g. $28$), number of samples in the file `n_samples` (note that setting `n_samples` too high might cause memory overload; a good approach is saving $100,000$ samples per file),
and a keyword argument `mult` which corresponds to the parameter $k_1$ from the thesis. This function can also be used for randomly generated test data.
To produce test data sets which are not random, but originate from data sets such as MNIST, use the `generate_dataset_data` function. It takes as input a `name` (location where
the data will be saved), a parameter `n_files` which controls how many data sets will be generated (and can most often be set to $1$), and, importantly, a `dataloader` parameter
which is a `torch.utils.data.DataLoader` object and can be wrapped around a `torchvision.datasets` data set such as `torchvision.datasets.MNIST`.  
Example:  
  generate_simple_data('train_data.py', length=28, mult=3)
  dataloader = DataLoader(MNIST(root='./Data/mnist_dataset', train=False, download=True, transform=torchvision.transforms.ToTensor()), batch_size=10000)
  generate_dataset_data('test_data', 1, dataloader=dataloader, train=False, n_samples=10000)

## DualOTComputation
This file mainly contains the `DualApproximator` class which lets you create and train a network.
It takes as input the side length `length` of the image distributions and a network class, which defaults to the class `FCNN` from the file `networks`.  
The network learns using the function `learn_potential` which takes as input the file name of a training data file. This function learns using an error on the dual potential of the OT
problem. One can, however, also compute the OT value from this potential and use an error on the optimum (instead of the potential, i.e. the _optimizer_); this can be done
with `learn_ws`. The respective testing functions are `test_potential` and `test_ws`.  
Example:  
  d = DualApproximator(28)
  d.learn_potential('train_data.py')
  d.test_potential('test_data_0.py')
  d.reset_params() # resets all trainable parameters to their default initialization
  d.learn_ws('train_data.py') # defaults to learn function `loss_max_ws` which does not need dual potentials in training data
  d.test_ws('test_data_0.py')

## sinkhorn
This file contains the Sinkhorn algorithm and many functions accompanying it. The algorithm itself is implemented in the `sinkhorn` function, with the log domain version available
through the `log_sinkhorn` function.  
`sinkhorn` takes as input distributions `mu` and `nu`, which can be one-dimensional if just one sample is passed, or two-dimensional for multiple samples which are processed in a
parallelized fashion. An optional initialization can be passed via the `start` keyword argument. In case this start vector might contain very small or very large values,
they can be controlled using the `min_start` and `max_start` keyword arguments, which set values that are too small or too large equal to these thresholds.  
It also needs a cost matrix `C`, which in the case for euclidean distances can be produced using the `euclidean_cost_matrix` function in `datacreation`. The `compare_iterations`,
`compare_time` and `compare_accuracy` functions offer various functionality to compare the performance of different initialization schemes for the Sinkhorn algorithm.
They take as input test data `data` and initialization schemes `inits` which is a list of initializations. For the default initialization, set one of the items to `None`. For
using a `DualApproximator`'s network, simply pass the `net` attribute of the `DualApproximator` object.  
Example:  

  ```python
  from datacreation import load_data
  testdata = load_data('test_data_0.py')
  c = euclidean_cost_matrix(28, 28, 2, True)
  eps = .2
  sinkhorn(testdata[0]['d1'][:5], testdata[0]['d2'][:5], c, eps, max_iter=1000, log=True) # computes the OT cost for the first five samples in testdata. With `log=True`, also returns
                                                                                          # the transport plan, the dual potentials of the OT dual problem, and the marginal constraint
                                                                                          # violations
  input = d.net(torch.cat((testdata[0]['d1'][:5], testdata[0]['d2'][:5]), 1))
  input = torch.exp(input)/eps
  sinkhorn(testdata[0]['d1'][:5], testdata[0]['d2'][:5], c, eps, max_iter=500, start=input, min_start=1e-30, max_start=1e30) # uses network's prediction for initialization
  compare_iterations(testdata[0], [d.net, None], ['network', 'default'], max_iter=500, eps=eps, min_start=1e-35, max_start=1e35) # plots accuracy for default and network initialization
                                                                                                                                 # w.r.t. the number of iterations
  ```

## utils
This file contains various functions for creating different types of plots from data.
`visualize_data` lets you visualize samples from your training and testing data, taking as input a tensor or array `data` of size $n\times dim$ or $n\times l\times l$ with $n$
being the number of samples, $dim$ the dimension of the samples, and $l$ the side length.  
`compute_dual` takes as input distributions `alpha` and `beta` and dual potentials `u` and `v` and computes the respective dual problem's value. All inputs need to be of dimension two and multiple samples can be processed at once.  
`compute_mean_conf` lets you compute a confidence interval from a data series in time. It takes as input `data`, which is a list of lists or list-like objects. Each item in `data` corresponds to one set of samples taken over time, e.g. the length of `data` equals the sample size at each point in time and the length of each item in `data` equals the number of time
steps that data was collected at. The function also takes as input a parameter `conf`, the desired confidence, i.e. this can e.g. be set to `.95` for a $95$ percent confidence interval.  
`plot` is a simple function that lets you plot multiple types of data at once, collected over the same `x`-values.  
`plot_conf` lets you plot various data time series alongside their confidence intervals, which will be shown as shaded areas around the plots. It takes as input `x`-values (which can also equal an integer in which case the x-values are interpolating between $0$ and that integer) and a list `y` where each item is a three-element list, where the first one corresponds to the lower values of
a confidence interval, the second one to the mean values, and the third one to the upper values of a confidence interval. This corresponds to the output generated by `compute_mean_conf`. It also takes as input `labels` corresponding to the labels of each item in `y`, `x_label` and `y_label` corresponding to the labels on the x- and y-axis, optional `titles` for each plot, and an optional parameter `separate_plots`. If this is not passed to the function, all `y` items will appear in a single plot. `separate_plots` can be a list of tuples, where each tuple corresponds to the indices from `y` which should appear in the same plot. If, e.g., the first three items should appear in one plot, and the fourth item in a second plot, set `separete_plots=[(0,1,2), (3)]`. In case `separete_plots` is given, the optional `rows` and `columns` parameters let you control how many rows and columns the plots should be placed in.  
Example:  
  visualize_data(testdata[0]['d1'][:5])
  f = d.net(torch.cat((testdata[0]['d1'][:20], testdata[0]['d2'][:20]), 1))
  c = euclidean_cost_matrix(28, 28, 2, True)
  g = compute_c_transform(c, f)
  f = compute_c_transform(c, g) # sets f to its double-c-transform. Can also be achieved directly by setting d.net.doubletransform=True
  dual_approx = compute_dual(testdata[0]['d1'][:20], testdata[0]['d2'][:20], f, g)
  print((dual_approx - testdata[0]['cost'][:20].view(-1)).abs().sum()/20) # average error on the dual OT problem value


## networks
This file contains the network class `FCNN`, which serves as the `net` attribute of the `DualApproximator` class in the `DualOTComputation` file and approximates a dual potential, given two input distributions. Each `FCNN` object has attributes `symmetry`, `doubletransform` and `zerosum` which default to `False`. If `symmetry` is set to `True`, the network will compute an output that is symmetric in the two input distributions. If `doubletransform` is set to `True`, it will instead compute the double- $c$-transform of the original output, which can be thought of as a $c$-concave approximation of the original output. If `zerosum` is set to `True`, it will enforce zero-sum outputs by subtracting a constant (note that this does not change the value of the dual problem as it is invariant under adding or subtracting constants to the potential).
