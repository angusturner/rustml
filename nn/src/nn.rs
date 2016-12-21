extern crate rulinalg as rl;

use self::rl::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};
use utils::{dims, add_ones, row_max, log, zero_first_col};

pub struct NN {
    num_inputs: usize,
    num_outputs: usize,
    num_hidden_layers: usize,
    weights: Vec<Matrix<f64>>,
    gradients: Vec<Matrix<f64>>,
    activations: Vec<Matrix<f64>>,
    raw_activations: Vec<Matrix<f64>>,
}

impl NN {
    /// static constructor function
    pub fn new(num_inputs: usize, num_outputs: usize) -> NN {
        NN {
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            num_hidden_layers: 0usize,
            weights: vec![],
            gradients: vec![],
            activations: vec![],
            raw_activations: vec![]
        }
    }

    /// return the weights
    pub fn get_weights(&self) -> Vec<Matrix<f64>> {
        self.weights.clone()
    }

    /// add a hidden layer with the specified number of neurons
    pub fn add_layer(&self, neurons: usize) -> NN {
        // initialise a new weight matrix, mapping the previous layer to the current one,
        // adding a column for the bias unit
        let mut n = 0usize;
        if self.num_hidden_layers == 0 {
            n = self.num_inputs + 1;
        } else {
            // get the number of neurons in the previous layer
            let i = self.num_hidden_layers - 1;
            n = self.weights[i].rows() + 1;
        }

        // create a new vector of weight matrices
        let mut new_weights = self.weights.clone();
        new_weights.push(Matrix::zeros(neurons, n));

        // return a new NN with the new weight vector
        // NOTE: avoids mutability and allows method chaining
        NN {
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            num_hidden_layers: self.num_hidden_layers + 1,
            weights: new_weights,
            gradients: self.gradients.clone(),
            activations: self.activations.clone(),
            raw_activations: self.raw_activations.clone()
        }
    }

    /// connect the last hidden layer to the output
    pub fn finalize(&self) -> NN {
        // get the dimensions of the final hidden layer
        let i = self.num_hidden_layers - 1;
        let n = self.weights[i].rows() + 1;

        // create a new vector of weight matrices
        let mut new_weights = self.weights.clone();
        new_weights.push(Matrix::zeros(self.num_outputs, n));

        // return a new NN with the new weight vector
        // NOTE: avoids mutability and allows method chaining
        // TODO: avoid code duplication!
        NN {
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            num_hidden_layers: self.num_hidden_layers + 1,
            weights: new_weights,
            gradients: self.gradients.clone(),
            activations: self.activations.clone(),
            raw_activations: self.raw_activations.clone()
        }
    }

    /// train the neural net using batch gradient descent
    /// X - the design matrix of training examples (excluding the bias unit column)
    /// y - the response matrix (output values mapped to one-hot vectors)
    /// lambda - regularization parameter
    /// alpha - learning rate
    /// max_iters - number of iterations of gradient descent to perform
    pub fn train(&mut self, X: &Matrix<f64>, y: &Matrix<f64>, alpha: &f64, lambda: &f64, max_iters: usize) { //, weights: Vec<Matrix<f64>>) -> Vec<i64> {
        // <DEBUG>
        // self.weights = weights;

        // compute network activations and get the hypothesis
        let h = self.forward_prop(&X);

        // get the dimensions of the response matrix
        let (m, r) = dims(&y);

        // compute the log-likelihood cost: -1/m * [ sum ( y.*log(h) + (1-y).*log(1-h) ) ]
        let ones: Matrix<f64> = Matrix::new(m, r, vec![1f64; m*r]);
        let cost = y.elemul(&log(&h)) + (&ones-y).elemul(&log(&(&ones-&h)));
        let mut J: f64 = - cost.sum() / (m as f64); // switch signs, take the mean

        // take the sum of the all the network parameters squared (excluding bias units)
        let sum_square_params = self.weights.iter().fold(0f64, |acc, theta| {
            let theta_0 = zero_first_col(&theta); // zero bias unit column
            acc + theta_0.elemul(&theta_0).sum()
        });

        // add regularization to cost: lambda/2m * [ sum(theta1.^2) + sum(theta2.^2) ... etc. ]
        J += (lambda/(2f64 * (m as f64))) * sum_square_params;

        // compute the gradients using back_prop
        self.back_prop(&y, &h);

        //row_max(&self.activations.last().unwrap()).iter().map(|x| x+1).collect::<Vec<i64>>()
    }

    // compute the error gradients for the network
    // y - response matrix
    // h - hypothesis
    fn back_prop(&mut self, y: &Matrix<f64>, h: &Matrix<f64>) {
        // compute errors on output
        let mut d = y - h;

        // get number of training examples
        let (m, _) = dims(&y);

        // propagate backwards through network
        let mut errors = vec![d.clone()];

        // iterate backwards through network to compute hidden layer errors
        //let mut theta: Matrix<f64>;
        //let mut z: Matrix<f64>;
        for i in self.weights.len()-1..0 {
            // get the weights, and remove the first column pertaining to bias units
            let theta = self.weights[i].clone();
            let theta_1 = MatrixSlice::from_matrix(&theta, [0, 1], theta.rows(), theta.cols()-1);

            // get the linear activations
            let z = self.raw_activations[i].clone();

            // compute the errors
            d = (theta_1.transpose() * d.clone().transpose())
                .transpose().elemul(&(z.apply(&sigmoid_gradient)));

            errors.push(d.clone());
        }

        // reverse the error vector and iterate back through, computing the parameter gradients
        errors.reverse();
        for i in 0..errors.len() {
            let d = errors[i].clone();
            let a = self.activations[i].clone();
            let theta_grad = d.transpose() * (a / (m as f64));
            self.gradients.push(theta_grad);
        }
    }

    /*fn cost_fn(X: &Matrix<f64>, y: &Matrix<f64>, theta1: &Matrix<f64>, theta2: &Matrix<f64>, lambda: &f64)
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
    }*/

    /// use the trained weights to generate predictions
    pub fn predict(&self, test_set: Matrix<f64>) {

    }

    // given a matrix of training examples, compute activations for the network using forward prop
    fn forward_prop(&mut self, X: &Matrix<f64>) -> Matrix<f64> {
        let mut a = X.clone(); // reference to activations of previous layer
        self.activations.push(a.clone());
        for theta in self.weights.iter() {
            // add bias column to activations, and compute linear activation with weights
            let z = add_ones(&a)*theta.transpose();

            // push raw activations to vector
            self.raw_activations.push(z.clone());

            // apply activation function
            a = z.clone().apply(&sigmoid);
            self.activations.push(a.clone());
        }

        // return the final layer (the hypothesis)
        a
    }
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
