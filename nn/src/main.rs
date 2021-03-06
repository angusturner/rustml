extern crate rulinalg as rl;
extern crate rand;
extern crate nn;

use rand::Rng;
use rl::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};
use nn::utils::*;
use nn::io::read_csv;
use nn::nn::NN;

#[allow(non_snake_case)]
fn main() {
    // design matrix m x n, where m = training examples, n = number of features
    let mut X = read_csv("input.csv");

    // response vector m x 1
    // y(i) corresponds to i-th training example, and is an integer between {1..r},
    // where r is the number of classes
    let y = read_csv("output.csv");

    // response matrix m x r, mapping y(i) to a vector of r zeroes, with a 1 in the y(i)-th position
    // i.e. each row mapped to a one-hot vector representation
    // let y2 = read_csv("output_map.csv");
    let y2 = one_hot(&y);

    // weights mapping input features to the second (hidden) layer s2 x n+1,
    // where s2 is the number of neurons in layer 2, n+1 is the number of features plus an extra
    // value corresponding to the input bias neuron
    let theta1 = read_csv("theta1.csv");

    // weights mapping hidden layer to output layer s3 x s2 + 1
    // where s3 = r is the number of outputs, and s2 + 1 is the number of hidden layer neurons
    // plus an extra value corresponding the hidden layer bias neuron
    let theta2 = read_csv("theta2.csv");

    // store a copy of the design matrix without the added bias units
    let X_1 = X.clone();

    // add a column of 1's to the design matrix
    X = add_ones(&X);

    // get the dimensions of the input and weights
    let (m, n) = dims(&X);
    let (s2, _) = dims(&theta1);
    let (s3, _) = dims(&theta2);

    // define network and training parameters
    let alpha = 2.0_f64; // learning rate
    let lambda = 1.0_f64; // regularization parameter
    let iters = 2000_i32; // number of updates for gradient descent
    let hidden_units = 25_usize; // # neurons in hidden layer

    // train the network with gradient descent
    let (theta1_t, theta2_t) = grad_desc(&X, &y2, &alpha, &lambda, hidden_units, iters);

    // compute the final training cost
    let (cost_t, _, _) = cost_fn(&X, &y2, &theta1_t, &theta2_t, &lambda);

    // compute the predictions
    let p = predict(&X, &theta1_t, &theta2_t);

    // compute the accuracy
    let y_vec: &Vec<f64> = y.data();
    let accuracy: f64 = accuracy(&p, y_vec);

    // NOTE: this is the same as above, but implemented as a struct in nn.rs
    // let mut test_net = NN::new(400, 10)
    // .add_layer(25) // add a 25 neuron hidden layer
    // .finalize();
    // .train(&X, &y, &alpha, &lambda, 400);
    //
    // train the network
    // test_net.train(&X_1, &y2, &alpha, &lambda, 200); //vec![theta1_t, theta2_t]);
    //
    // compute predictions
    // let p2 = test_net.predict(&X_1);
    //
    // compute the accuracy
    // let accuracy: f64 = accuracy(&p2, y_vec);

    // assert_eq!(&p2, &p);

    // println!("{:?}", test_net.get_weights());

    // debug output
    println!("Loaded m={} training examples with n={} features.", m, n - 1);
    println!("{} neurons in the hidden layer, and {} outputs", s2, s3);
    println!("Training set accuracy: {:?}", accuracy);
    println!("Training set error: {:?}", cost_t);
}

/// compute the accuracy by mapping the labels and corresponding predictions
/// to an iter of booleans, and then computing the fraction of `true` entries
fn accuracy(pred: &Vec<i64>, y: &Vec<f64>) -> f64 {
    let (correct, total) = pred.iter()
        .zip(y)
        .map(|(a, b)| (*a as f64) == *b)
        .fold((0f64, 0f64), |(correct, total), val| {
            match val {
                true => (correct + 1.0, total + 1.0),
                false => (correct, total + 1.0),
            }
        });
    correct / total
}

/// vanilla gradient descent with early stopping, for a 3 layer neural net.
/// X - design matrix
/// y - response matrix, with each row entry encoded as a one-hot vector
/// alpha - learning rate
/// lambda - regularization parameter
/// s2 - number of units (excluding bias unit) in the hidden layer
/// iters - the number of update iterations to perform
fn grad_desc(X: &Matrix<f64>,
             y: &Matrix<f64>,
             alpha: &f64,
             lambda: &f64,
             s2: usize,
             iters: i32)
             -> (Matrix<f64>, Matrix<f64>) {

    // determine number of input and output neurons from design / reponse matrices
    let (_, n) = dims(&X); // n = number of features + 1
    let (_, s3) = dims(&y);

    // initialize weight matrices mapping input->hidden and hidden->output
    let (mut theta1, mut theta2) = init_weights(n - 1, s2, s3);
    println!("Dimensions of parameter matrices: {:?}, {:?}",
             dims(&theta1),
             dims(&theta2));

    // perform gradient descent
    let mut tup: (f64, Matrix<f64>, Matrix<f64>) = cost_fn(&X, &y, &theta1, &theta2, &lambda);
    println!("Dimensions of gradient matrices: {:?}, {:?}",
             dims(&tup.1),
             dims(&tup.2));
    for _ in 0..iters {
        tup = cost_fn(&X, &y, &theta1, &theta2, &lambda);
        let (theta1_grad, theta2_grad) = (tup.1, tup.2);
        theta1 += -(theta1_grad * alpha);
        theta2 += -(theta2_grad * alpha);
        println!("cost: {}", tup.0);
    }

    (theta1, theta2)
}

/// generate predictions with neural net
#[allow(non_snake_case)]
fn predict(X: &Matrix<f64>, theta1: &Matrix<f64>, theta2: &Matrix<f64>) -> Vec<i64> {
    // compute activations on the hidden layer
    let z2 = X * theta1.transpose();
    let mut a2 = z2.apply(&sigmoid);
    a2 = add_ones(&a2); // add bias units

    // compute activations on the output layer
    let z3 = a2 * theta2.transpose();
    let h = z3.apply(&sigmoid);

    // get predictions by taking index of max val. in every row
    row_max(&h).iter().map(|x| x + 1).collect::<Vec<i64>>()
}

/// for training, the parameter matrices theta1 and theta2 must be initialised with random values
/// in the range [-epsilon, epsilon] where,
/// epsilon = sqrt(6) / sqrt(# input neurons + # output neurons)
fn init_weights(num_inputs: usize,
                num_hidden_layer: usize,
                num_outputs: usize)
                -> (Matrix<f64>, Matrix<f64>) {
    // compute epsilon
    let epsilon: f64 = 6f64.sqrt() / ((num_inputs + num_outputs) as f64).sqrt();

    // define function to generate rand value in desired range
    let rand = |_, _| rand::thread_rng().gen::<f64>() * 2.0 * epsilon - epsilon;

    // use rand function to populate matrices
    let theta1 = Matrix::from_fn(num_hidden_layer, num_inputs + 1, &rand);
    let theta2 = Matrix::from_fn(num_outputs, num_hidden_layer + 1, &rand);

    (theta1, theta2)
}

/// regularized log-likelihood cost function
/// returns the training error and the gradients wrt to each weight
#[allow(non_snake_case)]
fn cost_fn(X: &Matrix<f64>,
           y: &Matrix<f64>,
           theta1: &Matrix<f64>,
           theta2: &Matrix<f64>,
           lambda: &f64)
           -> (f64, Matrix<f64>, Matrix<f64>) {

    // compute activations on the hidden layer
    let z2 = X * theta1.transpose();
    let mut a2 = z2.clone().apply(&sigmoid);
    let a2_no_bias = a2.clone();
    a2 = add_ones(&a2); // add bias units

    // compute activations on the output layer
    let z3 = &a2 * theta2.transpose();
    let h = z3.apply(&sigmoid);

    // get dimensions of y
    let (m, r) = dims(&y);

    // compute the cost: -1/m * [ sum ( y.*log(h) + (1-y).*log(1-h) ) ]
    let ones: Matrix<f64> = Matrix::new(m, r, vec![1f64; m*r]); // (f64 - Matrix) is invalid :(
    let cost = y.elemul(&log(&h)) + (&ones - y).elemul(&log(&(&ones - &h)));
    let mut J: f64 = -cost.sum() / (m as f64); // switch signs, take the mean

    // zero the bias unit columns in the parameter matrices
    let theta1_0 = zero_first_col(&theta1);
    let theta2_0 = zero_first_col(&theta2);

    // add regularization: lambda/2m * [ sum(theta1.^2) + sum(theta2.^2) ]
    J += (lambda / (2f64 * (m as f64))) *
         (theta1_0.elemul(&theta1_0).sum() + theta2_0.elemul(&theta2_0).sum());

    // compute the errors on the output
    let d3 = h - y;

    // get a slice into theta2, with the first column (pertaining to bias units) removed
    let theta2_1 = MatrixSlice::from_matrix(&theta2, [0, 1], theta2.rows(), theta2.cols() - 1);

    // compute errors on the hidden layer (does not include bias unit)
    // let d2 =
    //  (theta2_1.transpose() * d3.transpose()).transpose().elemul(&(z2.apply(&sigmoid_gradient)));
    let sigmoid_test = a2_no_bias.apply(&sigmoid_gradient2);
    let d2 = (theta2_1.transpose() * d3.transpose()).transpose().elemul(&sigmoid_test);

    // compute gradients
    let mut theta1_grad = d2.transpose() * (X / (m as f64));
    // println!("{:?}, {:?}", dims(&d2), dims(&theta1_grad));
    let mut theta2_grad = d3.transpose() * (&a2 / (m as f64));

    // regularize gradients
    theta1_grad += theta1_0 * (lambda / (m as f64));
    theta2_grad += theta2_0 * (lambda / (m as f64));

    (J, theta1_grad, theta2_grad)
}

/// sigmoid activation function g(z)
fn sigmoid(n: f64) -> f64 {
    1.0f64 / (1.0f64 + ((-n).exp()))
}

/// sigmoid gradient function
/// g'(z) = a.*(1-a), where a = g(z)
fn sigmoid_gradient(n: f64) -> f64 {
    let a = sigmoid(n);
    a * (1.0 - a)
}

fn sigmoid_gradient2(a: f64) -> f64 {
    a * (1.0 - a)
}
