# SinkhornNNHybrid
Welcome to the SinkhornNNHybrid repository! This repo provides the PyTorch implementation for the thesis 'A Sinkhorn-NN Hybrid Algorithm for Optimal
Transport'.
The code is structured in five files, `DualOTComputation`, `networks`, `utils`, `sinkhorn` and `datacreation`. The main files you'll need reproduce the results from the thesis are `DualOTComputation` and `sinkhorn`, and `utils` contains some useful functions that let you visualize or plot data.

## DualOTComputation
This file mainly contains the `DualApproximator` class which lets you create and train a network
