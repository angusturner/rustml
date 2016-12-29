#![feature(rand)]

extern crate rulinalg as rl;
extern crate kmeans;
extern crate rand;

use rand::distributions::{IndependentSample, Range};
use rand::Rng;
use rl::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};
use kmeans::utils::*;
use kmeans::io::read_csv;
use std::f64;

#[allow(non_snake_case)]
fn main() {
    // design matrix m x n, where m = training examples, n = number of features
    let mut X = read_csv("input.csv");

    // response vector m x 1
    // y(i) corresponds to i-th training example, and is an integer between {1..r},
    // where r is the number of classes
    let y = read_csv("output.csv");

    // randomly initialize 10 cluster centroids, by setting to co-ords to random training example
    let mut p: Matrix<f64> = initialize_centroids(10, &X);

    // assign every training example to its nearest cluster
    let c: Vec<usize> = assign_to_clusters(&p, &X);

    // recompute the centroids based on the assigned training examples
    // NOTE: if no training examples are assigned, randomly re-initialize ?
    p = compute_centroids(&c, &X);
}

/// compute the cluster centroids
fn compute_centroids(c: &Vec<usize>, X: &Matrix<f64>) -> Matrix<f64> {
    Matrix::zeros(0,0)
}

/// assign each training example to its nearest centroid
fn assign_to_clusters(p: &Matrix<f64>, X: &Matrix<f64>) -> Vec<usize> {
    // iterate over rows in X
    X.row_iter()
    .map(|x_i| {
        // compute the euclidean distance to every centroid, and get the index k of the minimum
        p.row_iter()
        .map(|p_i| euclid(*p_i, *x_i))
        .fold((0usize, 0usize, f64::INFINITY), |(k, min_index, min_val), val| {
            match val < min_val {
                true => (k+1, k, val),
                false => (k+1, min_index, min_val)
            }
        })
    })
    .map(|tup| tup.1) // extract k from the tuple
    .collect::<Vec<usize>>()
}

/// compute the euclidean distance between two vectors (encoded as MatrixSlices)
fn euclid(a: MatrixSlice<f64>, b: MatrixSlice<f64>) -> f64 {
    (a - b).iter()
    .map(|x| x*x)
    .fold(0f64, |acc, val| acc+val)
    .sqrt()
}

/// compute the euclidean distance between two vectors
fn euclid_vec(a: Vec<f64>, b: Vec<f64>) -> f64 {
    a.iter().zip(b)
    .map(|(a, b)| (a - b).powi(2))
    .fold(0f64, |acc, val| acc+val)
    .sqrt()
}

/// randomly initialise K centroids, yielding a K x n matrix
/// K - number of centroids
/// n - number of features on input data
fn initialize_centroids(K: usize, X: &Matrix<f64>) -> Matrix<f64> {
    // get training set dimensions
    let (m, n) = dims(&X);

    // initialise rng with bounds [0, m)
    let between = Range::new(0usize, m);
    let mut rng = rand::thread_rng();

    // generate K centroids
    let mut centroids = Matrix::zeros(0, n);
    for _ in 0..K {
        // select a random row i from the input dataset, and append to centroids
        let i: usize = between.ind_sample(&mut rng);
        centroids = centroids.vcat(&X.row(i));
    }
    centroids
}
