extern crate csv;
extern crate rustc_serialize;
extern crate rulinalg as rl:

use rl::matrix::{Matrix, BaseMatrix};

fn main() {
    // Read in the data
    let mut X = read_csv("input.csv", true); // design matrix
    let mut y = read_csv("output.csv", false); // response vector
    let mut y2 = read_csv("output_map.csv", false); // response matrix (class k -> [0..1..r])
    let mut theta1 = read_csv("theta1.csv", false); // weights mapping input to hidden layer
    let mut theta2 = read_csv("theta2.csv", false); // weights mapping hidden layer to output layer

    // get the dimensions of the training set
    let (m, n) = dims(&X);

    // compute activations of the hidden layer
    let z2 = &X*&theta1.transpose();
    let mut a2 = apply(&z2, sigmoid);
    a2 = add_ones(&a2); // add bias units

    // compute activations on the output layer
    let mut z3 = &a2*&theta2.transpose();
    let mut a3 = apply(&z3, sigmoid);

    // compute the predictions
    let p = row_max(&a3);

    // compare predictions with the true values
    let mut q = 0;
    for i in 0..p.len() {
        if y[(i,0)] == (p[i] + 1) as f64 {
            q += 1;
        }
    }

    // compute accuracy
    let accuracy = (q as f64)/(m as f64);

    // compute the cost / training error
    let cost = cost_fn(&y2, &a3);

    // compute activations of the output
    println!("Training set accuracy: {}", accuracy);
}

/// log-likelihood cost function
fn cost_fn(h: &Matrix<f64>, y: &Matrix<f64>) -> f64 {
    let cost = y.data().dot(log(&h).data())+(y.add(-1f64)).data().dot((1f64-log(&h).data()));
    cost
}

/// compute log on a matrix
fn log(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, n) = dims(&mat);
    let col_vec = mat.data().iter().map(|x| x.ln()).collect::<Vec<f64>>();
    Matrix::new(m, n, col_vec);
}

/// reduce a matrix into a vector containing the index of the maximum in every row
fn row_max(mat: &Matrix<f64>) -> Vec<i64> {
    let (m, n) = dims(&mat);

    // transpose, then convert to vector (column-major order)
    let (_i, _v, res) = mat.transpose().data().iter()
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

/// add a column of ones to the start of a matrix
fn add_ones(mat: &Matrix<f64>) -> Matrix<f64> {
    let (m, n) = dims(&mat);

    // convert to DVector, iterate, dereference vals, collect into Vec (must be a better way...)
    let mut col_vec = mat.data().iter().map(|x| *x).collect::<Vec<f64>>();
    let mut ones = vec![1f64; m];
    ones.append(&mut col_vec);

    Matrix::new(m, n+1, &ones)
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
    (mat.nrows(), mat.ncols())
}

/// Sigmoid Function
fn sigmoid(n: &f64) -> f64 {
    1.0/(1.0+((-n).exp()))
}

/// Read CSV into a matrix
fn read_csv(file_path: &str, add_ones: bool) -> Matrix<f64> {
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

        // prepend a one (for design matrix)
        if add_ones {
            let mut one: Vec<f64> = vec![1f64];
            one.append(&mut row);
            row = one;
        }

        // compute number of columns
        n = row.len();

        // append row to output vector
        out.append(&mut row);

        // increment number of rows
        m += 1usize;
    }

    // reshape data into matrix
    Matrix::from_row_vector(m, n, &out)
}
