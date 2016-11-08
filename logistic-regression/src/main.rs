extern crate csv;
extern crate rustc_serialize;

fn main() {
    // collect user inputs
    let fpath = ::std::env::args().nth(1).unwrap();
    let iters = ::std::env::args().nth(2).unwrap().parse::<i32>().unwrap();
    let alpha = ::std::env::args().nth(3).unwrap().parse::<f64>().unwrap();

    // read data
    let (x, y) = read_csv(fpath);

    // initial parameter vector to zeros
    let mut theta: Vec<f64> = vec![0f64; x[0].len()];

    // compute the initial cost
    let (mut cost, mut grad) = cost_function(&x, &y, &theta);

    // print the initial cost and parameter gradients
    println!("Initial Cost: {:?}\nGrad: {:?}", &cost, &grad);

    // try to optimize
    for _ in 0..iters {
        let res = cost_function(&x, &y, &theta);
        cost = res.0;
        grad = res.1;
        gradient_descent(&mut theta, &grad, &alpha);
        println!("Cost: {:?}\nTheta: {:?}\nGrad: {:?}", &cost, &theta, &grad);
    }
}

/// Read the CSV data into a design matrix and response vector
///
/// m = number of training examples
/// n = number of features
///
/// Response vector y (1 x m)
///
/// Design matrix x_ij (m * n+1)
///
/// 1 x_0,1 ... x_1,n
/// 1 x_1,1 ... x_1,n
/// ...
/// 1 x_(m-1),1 ... x_(m-1),n
///
fn read_csv(fpath: String) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rdr = csv::Reader::from_file(fpath).unwrap();
    let mut x: Vec<Vec<f64>> = vec![]; // design matrix
    let mut y: Vec<f64> = vec![]; // response vector
    for record in rdr.decode() {
        // read the line into a vector
        let mut line: Vec<f64> = record.unwrap();

        // push final element to response vector
        let y_i: f64 = line.pop().unwrap();
        y.push(y_i);

        // construct ith training example by prepending a row of 1's
        let mut x_i: Vec<f64> = vec![1f64];
        x_i.append(&mut line);

        // add row to design matrix
        x.push(x_i);
    }
    (x, y)
}

/// Sigmoid / Logistic Function
fn sigmoid(num: f64) -> f64 {
    1.0/(1.0+((-num).exp()))
}

/// Log-likelihood cost function
///
/// theta contains the model parameters we are trying to fit
/// x is the design matrix (2D vector)
/// y is the response vector
///
/// Returns a tuple with the current cost (log-likelihood)
/// and the gradient vector w.r.t to each feature
fn cost_function(x: &Vec<Vec<f64>>, y: &Vec<f64>, theta: &Vec<f64>) -> (f64, Vec<f64>) {
    // compute number of training examples (m) and model features k = n+1
    let m: usize = y.len();
    let k: usize = theta.len(); // n+1 because of x0 = 1 column

    // initialize the cost and the gradient vector
    let mut J = 0f64;
    let mut grad: Vec<f64> = vec![0f64; k];

    // iterate through the training examples
    for i in 0..m {
        // compute the hypothesis for the ith training example
        let h = sigmoid(dot(&x[i], &theta));

        // increment cost according to log-likelihood function
        J += y[i]*h.ln() + (1f64-y[i])*(1f64-h).ln();

        // update the gradients
        for j in 0..k {
            grad[j] += (h-y[i])*x[i][j];
        }
    }

    J = -J/(m as f64); // scale cost by number of training examples
    (J, grad)
}

/// Update parameter values by subtracting some fraction (alpha) of the gradients
fn gradient_descent(theta: &mut Vec<f64>, grad: &Vec<f64>, alpha: &f64) {
    for i in 0..theta.len() {
        theta[i] -= alpha*grad[i];
    }
}

// compute dot product of two vectors
fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut res = 0f64;
    for i in 0..a.len() {
        res += a[i as usize]*b[i as usize];
    }
    res
}
