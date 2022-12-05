# Generative Adversarial Learning of Sinkhorn Algorithm Initializations
Welcome to the repository of the paper [Generative Adversarial Learning of Sinkhorn Algorithm Initializations](https://arxiv.org/abs/2212.00133)!
The paper aims at warm-starting the Sinkhorn algorithm with initializations computed by a neural network, which is trained in an adversarial fashion similar to a GAN using a second, generating neural network.
It is based on the Master's thesis 'A Sinkhorn-NN Hybrid Algorithm' by Jonathan Geuter, but differs from the thesis in many aspects. The thesis, along with its codebase and a comprehensive README, can be found in the [thesis branch](https://github.com/j-geuter/SinkhornNNHybrid/tree/thesis) of this repository. The main branch contains the codebase of the paper (which differs significantly from the thesis' codebase).

The code is structured in six files, `DualOTComputation.py`, `networks.py`, `utils.py`, `sinkhorn.py`, `costmatrix.py` and `datacreation.py`. The main files you'll need reproduce the results from the paper are `DualOTComputation.py` and `sinkhorn.py`, and `utils.py` contains some useful functions that let you visualize or plot data. To generate testing data, you'll need `datacreation.py`; however, all test datasets used in the paper are available in the `Data` folder. `costmatrix.py` contains a function to create cost matrices based on the Euclidean distance, and `networks.py` the neural network classes for the approximator and generator; you won't need to actively use either of these files unless you want to define your own cost function or network structure.  
The `requirements.txt` file lists all dependencies and their versions.  
The project is CUDA compatible.


# Reproduce Results

## Test Data
The folder `Data` contains all four test datasets used in the paper: 'random', 'teddies', 'MNIST' and 'CIFAR'. If you wish to produce your own test datasets, you can do so using the `generate_dataset_data` function in `datacreation.py`.
You can then load all test datafiles with the `load_data` function from `datacreation.py`:

```python
from src.datacreation import load_data
testdata = [load_data('Data/random.py'), load_data('Data/teddies.py'), load_data('Data/MNIST.py'), load_data('Data/CIFAR.py')]
```

## Create a Model
Unfortunately, the file containing the fully trained network's weights needs to be uploaded with Git LFS due to its size, and Git LFS seems to corrupt the file. Hence, the fully trained approximator and generator used in the paper are available for download in [this Google Drive folder](https://drive.google.com/drive/folders/1My0jXBqjDs4LVJtSX8gi45z0v9WMicNV?usp=sharing).
Assuming you saved the files as `net100k.pt` and `gen100k.pt`, you can then create a model with the two fully trained nets by running:

```python
from src.DualOTComputation import DualApproximator
d = DualApproximator(model='net100k.pt', gen_model='gen100k.pt')
```

If you wish to train your own model, you can do so using using the `learn_potential` function in `DualOTComputation.py`:

```python
from src.DualOTComputation import DualApproximator
d = DualApproximator()
d.learn_potential(n_samples=10000) # trains on 10,000 samples for 5 epochs, i.e. 50,000 samples total
```

The `learn_potential` function in `DualOTComputation.py` offers various optional arguments. If you wish to print the loss alongside sample images of the generator during training, pass `prints=True`.
If you want to learn using a loss on the transport distance (as outlined in Section 5.2 of the paper) instead of one on the dual potential, pass `learn_WS=True`.
You can also collect performance information on the test datasets using the `verbose`, `num_tests`, and `test_data` arguments, where you can pass `test_data=testdata` with `testdata` defined as above. The function will then return performance information upon completion.

## Obtain Results
To obtain the results from the paper, you'll need to run the `compare_iterations` function from `sinkhorn.py` for each test dataset. The results can then be saved using `save_data` from `datacreation.py`. I.e. with `testdata` as above:

```python
from src.sinkhorn import compare_iterations
from src.datacreation import save_data
d.net.eval()
results = []
for t in testdata:
  results.append(compare_iterations(t[:10], [None, d.net], ['default', 'net'], max_iter=2500, eps=.2, min_start=1e-35, max_start=1e35, plot=False, timeit=True))
save_data(results, 'results.py')
```

Now `r = results[i]` contains the results for the respective test dataset. `r[0]['WS']` and `r[0]['marg']` contain information on the Wasserstein distance and dual potential errors;
`r[1]` on the time it took for computations. In each of these locations, the values for the default initialization can be accessed at position `[0]` and for the network at position `[1]`. This will give you a 3-tuple, where each item is an array, the first one being the lower bound on the 95% confidence interval, the second one the mean, and the third one the upper bound on the confidence interval. So for example, the upper bound on the 95% confidence interval of the marginal constraint error of the default initialization for the first test dataset can be found at `results[0][0]['marg'][0][2]`.

## Plot Results
You can plot various results using the `plot_conf` function from `utils.py`.
Load results with `load_anydata` from `datacreation.py`:

```python
from src.datacreation import load_anydata
results = load_anydata('results.py')
```

Plot error on the marginal constraints:

```python
from src.utils import plot_conf
plot_conf(2500, results[0][0]['marg']+results[1][0]['marg']+results[2][0]['marg']+results[3][0]['marg'], ['default', 'net']*4, 'number of iterations', 'marginal constraint violation', titles=['random', 'teddies', 'MNIST', 'CIFAR'], separate_plots=[[0,1], [2,3], [4,5], [6,7]], rows=2, columns=2, slice=(5,24), scale_y=1/784**2)
```

Plot relative Wasserstein distance errors w.r.t. the number of Sinkhorn iterations:

```python
from src.utils import plot_conf
plot_conf(2500, results[0][0]['WS']+results[1][0]['WS']+results[2][0]['WS']+results[3][0]['WS'], ['default', 'net']*4, 'number of iterations', 'relative L1 error on WS distance', titles=['random', 'teddies', 'MNIST', 'CIFAR'], separate_plots=[[0,1], [2,3], [4,5], [6,7]], rows=2, columns=2, slice=(5,24))
```

Plot relative Wasserstein distance errors w.r.t. the computation time:

```python
from src.utils import plot_conf
plot_conf([results[0][1][0][1], results[0][1][1][1], results[1][1][0][1], results[1][1][1][1], results[2][1][0][1], results[2][1][1][1], results[3][1][0][1], results[3][1][1][1]], results[0][0]['WS']+results[1][0]['WS']+results[2][0]['WS']+results[3][0]['WS'], ['default', 'net']*4, 'time in s', 'relative L1 error on WS distance', titles=['random', 'teddies', 'MNIST', 'CIFAR'], separate_plots=[[0,1], [2,3], [4,5], [6,7]], rows=2, columns=2, slice=(5,24))
```

To compute the number of iterations needed for a particular bound on the marginal constraint violation, run the `iterations_per_marginal` function in `sinkhorn.py`:

```python
from src.sinkhorn import iterations_per_marginal
iters = iterations_per_marginal(1e-2, testdata, [d.net, None], stepsize=25)
# for a 1e-2 marginal constraint violation. This function runs a lot faster if you specify the start_iter argument
```

Barycenters can be computed and visualized using the `visualize_barycenters` function in `utils.py`. Note that typically, between 15 and 30 gradient steps are sufficient.
