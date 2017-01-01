extern crate rulinalg as rl;
extern crate kmeans;
extern crate rand;
extern crate num_traits;

use rand::distributions::{IndependentSample, Range};

use rl::matrix::{Axes, Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};
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
    let mut y_raw: Matrix<f64> = read_csv("output.csv");

    // randomly shuffle the input data and labels
    shuffle_rows(&mut vec![&mut X, &mut y_raw], 5000);

    // split the input data into a training, cross-validation and test set
    let (X_train, X_cv, X_test) = split_data(&X, (0.6, 0.2, 0.2));
    //let mut X_train = X;

    // input labels for digit 0 have been mapped to 10 due to ease-of-use with Matlab's
    // ridiculous 1-indexing of vectors. This step reverses that, and also casts f64 -> usize.
    let y_vec = y_raw.data()
        .iter()
        .map(|y_i| {
            let yusize = *y_i as usize;
            match yusize {
                10 => 0,
                _ => yusize,
            }
        })
        .collect::<Vec<usize>>();

    // split the labels into training, cv and test sets
    let y_mat = Matrix::new(y_vec.len(), 1, y_vec.clone());
    let (y_train, y_cv, y_test) = split_data(&y_mat, (0.6, 0.2, 0.2));
    //let mut y_train = y_mat;

    // randomly initialize 10 cluster centroids, by setting to co-ords to random training example
    let mut p: Matrix<f64> = initialize_centroids(10, &X_train);

    // compute centroids directly from training set labels, to get an indication of max accuracy
    // p = compute_centroids(&y_train.data(), &X_train, 10);

    // assign every training example to its nearest cluster
    let mut c: Vec<usize> = assign_to_clusters(&p, &X_train);

    // complete 50 iterations of kmeans algorithm
    for i in 0..50 {
        println!("Iter {}", i);
        p = compute_centroids(&c, &X_train, 10);
        c = assign_to_clusters(&p, &X_train);
        // println!("{:?}, {:?}", c.len(), dims(&p));
    }

    // compute the label composition of each cluster
    let mut composition: Matrix<f64> = Matrix::zeros(10, 10);
    composition = c.iter()
        .zip(y_train.data().iter())
        .fold(composition, |mut acc, (c_i, y_i)| {
            acc[[*c_i, *y_i]] += 1.0;
            acc
        });

    // determine which cluster (0 -> K) to associate with each label
    let inference = row_max(&composition);
    println!("{:?}", inference);

    //
    //for row in composition.row_iter() {
    //    println!("{:?}", (*row).into_iter().collect::<Vec<_>>());
    //}

    // generate predictions on the training set and compute accuracy
    let c_train: Vec<usize> = assign_to_clusters(&p, &X_train);
    let pred = c_train.iter().map(|c_i| inference[*c_i] as usize).collect::<Vec<usize>>();
    let train_acc = accuracy(&pred, &(y_train.data()));
    println!("Training set accuracy {}", train_acc);

    // generate predictions on the test set and compute accuracy
    let c_test: Vec<usize> = assign_to_clusters(&p, &X_test);
    let pred = c_test.iter().map(|c_i| inference[*c_i] as usize).collect::<Vec<usize>>();
    let test_acc = accuracy(&pred, &(y_test.data()));
    println!("Test set accuracy {}", test_acc);
}

/// compute the accuracy by mapping the labels and corresponding predictions
/// to an iter of booleans, and then computing the fraction of `true` entries
fn accuracy(pred: &Vec<usize>, y: &Vec<usize>) -> f64 {
    let (correct, total) = pred.iter()
        .zip(y)
        .map(|(a, b)| a == b)
        .fold((0f64, 0f64), |(correct, total), val| {
            match val {
                true => (correct + 1.0, total + 1.0),
                false => (correct, total + 1.0),
            }
        });
    // println!("Correct {}, Total {}", &correct, &total);
    correct / total
}

/// function to split the data into (train, cv, test) sets
/// NOTE: assumes the data is already randomly shuffled
fn split_data<T>(mat: &Matrix<T>, split: (f64, f64, f64)) -> (Matrix<T>, Matrix<T>, Matrix<T>)
    where T: std::clone::Clone + num_traits::identities::Zero + std::marker::Copy
{
    // get the input data length
    let m = mat.rows();

    // determine the number of elems in each set
    let (a, b, _c) = split;
    let split1 = (m as f64 * a).round() as usize;
    let split2 = (m as f64 * b).round() as usize;

    // split first matrix
    let (train, residual) = mat.split_at(split1, Axes::Row);
    let (cv, test) = residual.split_at(split2, Axes::Row);

    // convert all slices into owned Matrix structs, and return
    (train.into_matrix(), cv.into_matrix(), test.into_matrix())
}

/// function to shuffle matrix rows
/// takes a vector of mutable matrix references, and will apply same tranformation to all matrices
/// (i.e to apply same shuffling to labels and data)
/// NOTE: all input matrices must have same row count
fn shuffle_rows<T>(vec: &mut Vec<&mut Matrix<T>>, swaps: usize) {
    // get the number of input rows
    let num_rows = vec[0].rows();

    // initialise random number generator, with bounds [0, num_rows)
    let between = Range::new(0usize, num_rows);
    let mut rng = rand::thread_rng();

    // repeatedly select and swap two random rows
    for _ in 0..swaps {
        let a = between.ind_sample(&mut rng);
        let b = between.ind_sample(&mut rng);
        for i in 0..vec.len() {
            vec[i].swap_rows(a, b);
        }
    }
}

/// for each cluster construct a matrix of the associated training examples
/// then average over the rows to obtain the centroids
fn compute_centroids(c: &Vec<usize>, X: &Matrix<f64>, K: usize) -> Matrix<f64> {
    let n = X.cols();
    let mut p: Matrix<f64> = Matrix::zeros(0, n);
    let mut clusters: Vec<Matrix<f64>> = vec![Matrix::zeros(0, n); K];
    // sort every training example into a matrix for its assigned cluster
    X.row_iter()
        .zip(c)
        .fold(clusters, |mut acc, (x_i, c_i)| {
            acc[*c_i] = acc[*c_i].vcat(&x_i);
            acc
        })
        .iter()
        .map(|mat| Matrix::new(1, n, (mat.sum_rows() / mat.rows() as f64)))
        .fold(p, |acc, val| acc.vcat(&val))
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
    (a - b)
        .iter()
        .map(|x| x * x)
        .fold(0f64, |acc, val| acc + val)
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
