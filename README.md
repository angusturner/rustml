# Rust Machine Learning

Some basic ML algorithms implemented in Rust.

## Logistic Regression
Binary classification with a log-likelihood cost function, and batch gradient descent.
Usage - `cargo run ex2data1.txt <iters> <learning rate>`
e.g 150,000 iters and alpha=0.001 seems to work

## Kmeans Clustering
An implementation of unsupervised clustering with kmeans.

### Progress

* [X] Import data from CSV
* [X] Random centroid initialization
* [X] Cluster assignment step
* [X] Centroid calculation
* [ ] Random data splitting (test/cv/train)
* [ ] kmeans++ initialization
* [ ] Minimize cost function over multiple centroid initializations
* [ ] Try cluster assignment with correct labels, and check prediction accuracy
* [ ] Add unit tests


## Neural Network

A basic multilayer perceptron for solving classification problems. Originally ported from Matlab,
as a solution to a project from Andrew Ngs Machine Learning course. I am now in the process of
generalising the solution to allow arbitrary network architectures, and configurable activation
functions. More sophisticated optimisation algorithms (other than batch gradient descent), may
also be considered.

### Progress

* [X] Allow CSV Importing of Pre-Trained Network Weights
* [X] Unrolling / rolling of feature vectors into matrices
* [X] Basic 1-layer architecture with forward propagation to classify data
* [X] Implement cost function
* [X] Backpropagation to get gradients
* [X] Gradient descent to minimize cost function
* [X] Add regularization to prevent overfitting
* [X] Refactor to allow arbitrary number of layers and neurons
* [ ] Performance Enhancements for Generalized Algorithm
* [ ] Implement gradient checking to verify backprop implementation.
* [ ] Ability to save trained weights in CSV
* [ ] Configurable activation functions (per layer?)
* [ ] Better optimisation functions (Levenberg-Marquardt?)
* [ ] Add unit tests
