/// this module contains helper functions for matrix manipulation
extern crate rulinalg as rl;

use std::cmp;
use self::rl::matrix::{Matrix, BaseMatrix};

/// maps each value in an m x 1 matrix into a one-hot vector represenation, yielding an m x n
/// matrix, where n is the number of output classes
/// NOTE: assumes input classes are in f64 format, and indexed from 1
pub fn one_hot(input: &Matrix<f64>) -> Matrix<f64> {
    // determine the number of output classes by iterating over the input
    let n: usize = input.iter().fold(0usize, |acc, &val| cmp::max(acc, val as usize) as usize);

    // get the input dimensions
    let (m, _) = dims(&input);

    // initialize a matrix of zeros
    let mut out: Matrix<f64> = Matrix::zeros(m, n);

    // fill in the ones
    let mut j: usize = 0;
    for i in 0..m {
        j = (input[[i, 0]] - 1f64) as usize; // NOTE: assumes classes start at one...
        out[[i, j]] = 1f64;
    }

    out
}

/// compute the natural logarithm of each matrix element
pub fn log(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, n) = dims(&mat);
    let col_vec = mat.iter().map(|x| x.ln()).collect::<Vec<f64>>();
    Matrix::new(m, n, col_vec)
}

/// reduce a matrix into a vector containing the index of the maximum in every row
pub fn row_max(mat: &Matrix<f64>) -> Vec<i64> {
    let (m, n) = dims(&mat);

    // transpose, then convert to vector (column-major order)
    let (_i, _v, res) = mat.iter()
        // how to abuse reduce patterns
        .fold((0usize, 0f64, vec![0i64; m]), |acc, &val| {
            let (mut i, mut v, mut vec) = acc;
            if &i % &n == 0 {
                v = 0f64;
            }
            if val > v {
                v = val;
                let ind = &i/&n;
                vec[ind] = (&i % &n) as i64;
            }
            i += 1;
            (i, v, vec)
        });
    res
}

/// zero first column of matrix
pub fn zero_first_col(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, _n) = dims(&mat);
    let mut out = mat.clone();
    for i in 0..m {
        out[[i, 0]] = 0f64;
    }
    out
}

/// add a column of ones to the start of a matrix
pub fn add_ones(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, n) = dims(&mat);

    let mut col_vec = mat.transpose().iter().map(|x| *x).collect::<Vec<f64>>();
    let mut ones = vec![1f64; m];
    ones.append(&mut col_vec);

    Matrix::new(n + 1, m, ones).transpose()
}

/// get matrix dimensions
pub fn dims(mat: &Matrix<f64>) -> (usize, usize) {
    (mat.rows(), mat.cols())
}

/// function to unroll supplied matrices into a single 1-dimensional vector
pub fn unroll_matrices(matrices: Vec<&Matrix<f64>>) -> Vec<f64> {
    let mut out = vec![];
    for val in matrices {
        out.append(&mut val.data().clone());
    }
    out
}

/// function to re-roll vector into its contituent matrices
/// vector - all the network parameters unrolled into a single vector
/// dimensions - a vector of dimension tuples, describing how to reconstruct the matrices
pub fn roll_matrices(vector: Vec<f64>, dimensions: Vec<(usize, usize)>) -> Vec<Matrix<f64>> {
    let mut out = vec![];
    let mut bounds = (0usize, 0usize); // moving reference to the lower and upper slice bounds
    for i in 0..dimensions.len() {
        // get the dimensions
        let dims = dimensions[i];

        // update the upper-bound
        bounds.1 = bounds.0 + (dims.0 * dims.1);

        // construct the matrix
        out.push(Matrix::new(dims.0, dims.1, &vector[bounds.0..bounds.1]));

        // update lower bound
        bounds.0 += dims.0 * dims.1;
    }
    out
}
