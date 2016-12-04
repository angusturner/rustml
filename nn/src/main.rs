extern crate csv;
extern crate rustc_serialize;
extern crate rulinalg as rl;
extern crate rand;

use rand::Rng;
use rl::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};

#[allow(non_snake_case)]
fn main() {
    // design matrix m x n, where m = training examples, n = number of features
    let mut X = read_csv("input.csv");

    // response vector m x 1
    // y(i) corresponds to i-th training example, and is an integer between {1..r},
    // where r is the number of classes
    let y = read_csv("output.csv");

    // response matrix m x r, mapping y(i) to a vector of r zeroes, with a 1 in the y(i)-th position
    let y2 = read_csv("output_map.csv");

    // weights mapping input features to the second (hidden) layer s2 x n+1,
    // where s2 is the number of neurons in layer 2, n+1 is the number of features plus an extra
    // value corresponding to the input bias neuron
    let theta1 = read_csv("theta1.csv");

    // weights mapping hidden layer to output layer s3 x s2 + 1
    // where s3 = r is the number of outputs, and s2 + 1 is the number of hidden layer neurons
    // plus an extra value corresponding the hidden layer bias neuron
    let theta2 = read_csv("theta2.csv");

    // add a column of 1's to the design matrix
    X = add_ones(&X);

    // get the dimensions of the input and weights
    let (m, n) = dims(&X);
    let (s2, _) = dims(&theta1);
    let (s3, _) = dims(&theta2);

    // define network and training parameters
    let alpha = 2.0_f64; // learning rate
    let lambda = 1.0_f64; // regularization parameter
    let iters = 450_i32; // number of updates for gradient descent
    let hidden_units = 25_usize; // # neurons in hidden layer

    // train the network with gradient descent
    let (theta1_t, theta2_t) = grad_desc(&X, &y2, &alpha, &lambda, hidden_units, iters);

    // compute the final training cost
    let (cost_t, _, _) = cost_fn(&X, &y2, &theta1_t, &theta2_t, &lambda);

    // compute the predictions
    let p = predict(&X, &theta1_t, &theta2_t);

    // compare predictions with the true values
    let mut q = 0;
    for i in 0..p.len() {
        // compare ith column of the row-vector y with the associated prediction
        // note: since
        if y[[i,0]] == p[i] as f64 {
            q += 1;
        }
    }

    // calculate the accuracy
    let accuracy = (q as f64)/(m as f64);

    // debug output
    println!("Loaded m={} training examples with n={} features.", m, n-1);
    println!("Network has {} neurons in the hidden layer, and {} outputs", s2, s3);
    println!("Training set accuracy (should be about 95%): {:?}", accuracy);
    println!("Training set error (should be about 0.5): {:?}", cost_t);
}

/// vanilla gradient descent with early stopping, for a 3 layer neural net.
/// X - design matrix
/// y - response matrix, with each row entry encoded as a one-hot vector
/// alpha - learning rate
/// lambda - regularization parameter
/// s2 - number of units (excluding bias unit) in the hidden layer
/// iters - the number of update iterations to perform
fn grad_desc(X: &Matrix<f64>, y: &Matrix<f64>, alpha: &f64, lambda: &f64, s2: usize, iters: i32)
    -> (Matrix<f64>, Matrix<f64>) {

    // determine number of input and output neurons from design / reponse matrices
    let (_, n) = dims(&X); // n = number of features + 1
    let (_, s3) = dims(&y);

    // initialize weight matrices mapping input->hidden and hidden->output
    let (mut theta1, mut theta2) = init_weights(n-1, s2, s3);

    // perform gradient descent
    let mut tup: (f64, Matrix<f64>, Matrix<f64>) = cost_fn(&X, &y, &theta1, &theta2, &lambda);
    for _ in 0..iters {
        tup = cost_fn(&X, &y, &theta1, &theta2, &lambda);
        let (theta1_grad, theta2_grad) = (tup.1, tup.2);
        theta1 += - (theta1_grad * alpha);
        theta2 += - (theta2_grad * alpha);
        println!("cost: {}", tup.0);
    }

    ( theta1, theta2 )
}

/// levenberg-marquardt optimization (TODO)
//fn levenberg() {
//
//}

/// generate predictions with neural net
#[allow(non_snake_case)]
fn predict(X: &Matrix<f64>, theta1: &Matrix<f64>, theta2: &Matrix<f64>) -> Vec<i64> {
    // compute activations on the hidden layer
    let z2 = X*theta1.transpose();
    let mut a2 = z2.apply(&sigmoid);
    a2 = add_ones(&a2); // add bias units

    // compute activations on the output layer
    let z3 = a2*theta2.transpose();
    let h = z3.apply(&sigmoid);

    // get predictions by taking index of max val. in every row
    row_max(&h).iter().map(|x| x+1).collect::<Vec<i64>>()
}

/// for training, the parameter matrices theta1 and theta2 must be initialised with random values
/// in the range [-epsilon, epsilon] where epsilon = sqrt(6) / sqrt(# input neurons + # output neurons)
fn init_weights(num_inputs: usize, num_hidden_layer: usize, num_outputs: usize) -> (Matrix<f64>, Matrix<f64>)  {
    // compute epsilon
    let epsilon: f64 = 6f64.sqrt() / ((num_inputs+num_outputs) as f64).sqrt();

    // define function to generate rand value in desired range
    let rand = |_, _| {
        rand::thread_rng().gen::<f64>() * 2.0 * epsilon - epsilon
    };

    // use rand function to populate matrices
    let theta1 = Matrix::from_fn(num_hidden_layer, num_inputs+1, &rand);
    let theta2 = Matrix::from_fn(num_outputs, num_hidden_layer+1, &rand);

    (theta1, theta2)
}

/// regularized log-likelihood cost function
/// returns the training error and the gradients wrt to each weight
#[allow(non_snake_case)]
fn cost_fn(X: &Matrix<f64>, y: &Matrix<f64>, theta1: &Matrix<f64>, theta2: &Matrix<f64>, lambda: &f64)
    -> (f64, Matrix<f64>, Matrix<f64>) {

    // compute activations on the hidden layer
    let z2 = X*theta1.transpose();
    let mut a2 = z2.clone().apply(&sigmoid);
    a2 = add_ones(&a2); // add bias units

    // compute activations on the output layer
    let z3 = &a2*theta2.transpose();
    let h = z3.apply(&sigmoid);

    // get dimensions of y
    let (m, r) = dims(&y);

    // compute the cost: -1/m * [ sum ( y.*log(h) + (1-y).*log(1-h) ) ]
    let ones: Matrix<f64> = Matrix::new(m, r, vec![1f64; m*r]); // needed because f64 - Matrix is not valid :(
    let cost = y.elemul(&log(&h)) + (&ones-y).elemul(&log(&(&ones-&h)));
    let mut J: f64 = - cost.sum() / (m as f64); // switch signs, take the mean

    // zero the bias unit columns in the parameter matrices
    let theta1_0 = zero_first_col(&theta1);
    let theta2_0 = zero_first_col(&theta2);

    // add regularization: lambda/2m * [ sum(theta1.^2) + sum(theta2.^2) ]
    J += (lambda/(2f64 * (m as f64))) * (theta1_0.elemul(&theta1_0).sum() + theta2_0.elemul(&theta2_0).sum());

    // compute the errors on the output
    let d3 = h - y;

    // get a slice into theta2, with the first column (pertaining to bias units) removed
    let theta2_1 = MatrixSlice::from_matrix(&theta2, [0, 1], theta2.rows(), theta2.cols()-1);

    // compute errors on the hidden layer (does not include bias unit)
    let d2 = (theta2_1.transpose()*d3.transpose()).transpose().elemul(&(z2.apply(&sigmoid_gradient)));

    // compute gradients
    let mut theta1_grad = d2.transpose()*(X / (m as f64));
    let mut theta2_grad = d3.transpose()*(a2 / (m as f64));

    // regularize gradients
    theta1_grad += theta1_0 * (lambda/(m as f64));
    theta2_grad += theta2_0 * (lambda/(m as f64));

    (J, theta1_grad, theta2_grad)
}

/// compute log of each matrix element
fn log(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, n) = dims(&mat);
    let col_vec = mat.iter().map(|x| x.ln()).collect::<Vec<f64>>();
    Matrix::new(m, n, col_vec)
}

/// reduce a matrix into a vector containing the index of the maximum in every row
fn row_max(mat: &Matrix<f64>) -> Vec<i64> {
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

/// sigmoid activation function g(z)
fn sigmoid(n: f64) -> f64 {
    1.0f64/(1.0f64+((-n).exp()))
}

/// sigmoid gradient function
/// g'(z) = a.*(1-a), where a = g(z)
fn sigmoid_gradient(n: f64) -> f64 {
    let a = sigmoid(n);
    a*(1.0-a)
}

/// zero first column of matrix
fn zero_first_col(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, _n) = dims(&mat);
    let mut out = mat.clone();
    for i in 0..m {
        out[[i, 0]] = 0f64;
    }
    out
}

/// add a column of ones to the start of a matrix
fn add_ones(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, n) = dims(&mat);

    let mut col_vec = mat.transpose().iter().map(|x| *x).collect::<Vec<f64>>();
    let mut ones = vec![1f64; m];
    ones.append(&mut col_vec);

    Matrix::new(n+1, m, ones).transpose()
}

/// get matrix dimensions
fn dims(mat: &Matrix<f64>) -> (usize, usize) {
    (mat.rows(), mat.cols())
}

/// read CSV into a matrix
fn read_csv(file_path: &str) -> Matrix<f64> {
    // initialise some stuff
    let mut rdr = csv::Reader::from_file(file_path).unwrap().has_headers(false);
    let mut out: Vec<f64> = vec![]; // output vector
    let mut m: usize = 0; // number of rows
    let mut n: usize = 0; // number of cols

    // iterate through records
    for record in rdr.decode() {
        // decode row into a vector
        let mut row: Vec<f64> = match record {
            Ok(res) => res,
            Err(err) => panic!("failed to read {}: {}", m, err)
        };

        // compute number of columns
        n = row.len();

        // append row to output vector
        out.append(&mut row);

        // increment number of rows
        m += 1usize;
    }

    // reshape data into matrix
    Matrix::new(m, n, out)
}

// use two sided difference with the cost function to compare actual gradients with numerical estimation
// cost'(theta) = [ cost(theta+epsilon)-cost(theta-epsilon) ] / (2* epsilon), where epsilon ~ 10^-4
//fn grad_checking() {
    // TODO
//}
