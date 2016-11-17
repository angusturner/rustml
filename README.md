# Rust Machine Learning

Some basic ML algorithms implemented in Rust, primarily as solutions to Andrew Ng's ML course on Coursera.

## Logistic Regression
Binary classification with a log-likelihood cost function, and batch gradient descent.
Usage - `cargo run ex2data1.txt <iters> <learning rate>` 
e.g 150,000 iters and alpha=0.001 seems to work

## Neural Network (WIP)

### Plan

* ~~Allow CSV Importing of Pre-Trained Network Weights~~
* ~~Unrolling / rolling of feature vectors into matrices~~
* ~~Basic architecture with forward propagation to classify data~~
* Implement cost function
* Backpropagation to get gradients
* Gradient descent to minimize cost function
* Implement gradient checking to test if backprop is working.
* Add regularization to prevent overfitting
* Ability to save trained weights in CSV (or similar)
* Add better optimisation functions (Levenberg-Marquardt)
