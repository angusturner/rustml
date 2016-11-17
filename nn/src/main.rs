extern crate csv;
extern crate rustc_serialize;
extern crate rulinalg as rl;

use rl::matrix::{Matrix, BaseMatrix};

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

    // compute activations of the hidden layer
    let z2 = &X*&theta1.transpose();
    let mut a2 = apply(&z2, sigmoid);
    a2 = add_ones(&a2); // add bias units

    // compute activations on the output layer
    let z3 = &a2*&theta2.transpose();
    let a3 = apply(&z3, sigmoid);

    // compute the predictions
    let p = row_max(&a3);

    // compare predictions with the true values
    let mut q = 0;
    for i in 0..p.len() {
        if y[[i,0]] == (p[i] + 1) as f64 {
            q += 1;
        }
    }

    // compute accuracy
    let accuracy = (q as f64)/(m as f64);

    // compute the cost / training error
    let lambda = 1f64;
    let cost = cost_fn(&a3, &y2, lambda, &theta1, &theta2);

    // debug output
    println!("Loaded m={} training examples with n={} features.", m, n-1);
    println!("Network has {} neurons in the hidden layer, and {} outputs", s2, s3);
    println!("Training set accuracy (should be about 95%): {:?}", accuracy);
    println!("Training set error (should be about 0.5): {:?}", cost);
}

/// regularized log-likelihood cost function
#[allow(non_snake_case)]
fn cost_fn(h: &Matrix<f64>, y: &Matrix<f64>, lambda: f64, theta1: &Matrix<f64>, theta2: &Matrix<f64>) -> f64 {
    let (m, n) = dims(&y);

    // compute the cost: -1/m * [ sum ( y.*log(h) + (1-y).*log(1-h) ) ]
    let ones: Matrix<f64> = Matrix::new(m, n, vec![1f64; m*n]); // needed because f64 - Matrix is not valid :(
    let cost = y.elemul(&log(&h)) + (&ones-y).elemul(&log(&(&ones-h)));
    let J: f64 = - cost.sum() / (m as f64); // switch signs, take the mean

    // zero the bias units in the parameter matrices
    let theta1_0 = zero_first_col(&theta1);
    let theta2_0 = zero_first_col(&theta2);

    // add regularization: lambda/2m * [ sum(theta1.^2) + sum(theta2.^2) ]
    J + (lambda/(2f64 * (m as f64))) * (theta1_0.elemul(&theta1_0).sum() + theta2_0.elemul(&theta2_0).sum())
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

/// sigmoid activation function
fn sigmoid(n: &f64) -> f64 {
    1.0f64/(1.0f64+((-n).exp()))
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

/// apply a function to every element in a matrix
fn apply(mat: &Matrix<f64>, f: fn(n: &f64) -> f64) -> Matrix<f64> {
    // get dimensions
    let (m, n) = dims(&mat);

    // convert to column vector, iter, apply function, recollect into vec
    let col_vec = mat.data().iter().map(|x| f(x)).collect::<Vec<f64>>();

    // reshape into a matrix
    Matrix::new(m, n, col_vec)
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
