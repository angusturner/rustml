extern crate rulinalg as rl;
extern crate rand;

use self::rl::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};
use utils::{dims, add_ones, row_max, log, zero_first_col};
use self::rand::Rng;

pub struct NN {
    num_inputs: usize,
    num_outputs: usize,
    num_hidden_layers: usize,
    weights: Vec<Matrix<f64>>,
    gradients: Vec<Matrix<f64>>,
    activations: Vec<Matrix<f64>>,
    raw_activations: Vec<Matrix<f64>>,
    epsilon: f64
}

impl NN {
    /// static constructor function
    pub fn new(num_inputs: usize, num_outputs: usize) -> NN {
        // compute epsilon, which is used as a lower and upper bounds
        // in the random weight initialisation
        let epsilon = 6f64.sqrt() / ((num_inputs+num_outputs) as f64).sqrt();

        NN {
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            num_hidden_layers: 0usize,
            weights: vec![],
            gradients: vec![],
            activations: vec![],
            raw_activations: vec![],
            epsilon: epsilon
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
        new_weights.push(self.init_weights(neurons, n));

        // return a new NN with the new weight vector
        // NOTE: avoids mutability and allows method chaining
        NN {
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            num_hidden_layers: self.num_hidden_layers + 1,
            weights: new_weights,
            gradients: self.gradients.clone(),
            activations: self.activations.clone(),
            raw_activations: self.raw_activations.clone(),
            epsilon: self.epsilon
        }
    }

    /// connect the last hidden layer to the output
    pub fn finalize(&self) -> NN {
        // get the dimensions of the final hidden layer
        let i = self.num_hidden_layers - 1;
        let n = self.weights[i].rows() + 1;

        // create a new vector of weight matrices
        let mut new_weights = self.weights.clone();
        new_weights.push(self.init_weights(self.num_outputs, n));

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
            raw_activations: self.raw_activations.clone(),
            epsilon: self.epsilon
        }
    }

    /// parameter matrices must be initialised with random values in the range [-epsilon, epsilon]
    /// where epsilon = sqrt(6) / sqrt(# input neurons + # output neurons)
    fn init_weights(&self, rows: usize, cols: usize) -> Matrix<f64> {
        // get epsilon
        let epsilon: f64 = self.epsilon;

        // define closure to generate rand value in desired range, capturing epsilon
        let rand = |_, _| {
            rand::thread_rng().gen::<f64>() * 2.0 * epsilon - epsilon
        };

        // use rand function to populate matrices
        Matrix::from_fn(rows, cols, &rand)
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

        // gradient descent
        for _ in 0..max_iters {
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

            println!("The cost is {}", J);

            // compute the gradients using back_prop
            self.back_prop(&y, &h, &lambda);

            println!("{}", self.gradients.clone().len());

            // update the weights
            /*for i in 0..self.weights.len() {
                self.weights[i] += - (self.gradients[i].clone() * alpha);
            }*/
        }
    }

    // compute the error gradients for the network
    // y - response matrix
    // h - hypothesis
    fn back_prop(&mut self, y: &Matrix<f64>, h: &Matrix<f64>, lambda: &f64) {
        // compute errors on output
        let mut d = y - h;

        // get number of training examples
        let (m, _) = dims(&y);

        // propagate backwards through network
        let mut errors = vec![d.clone()];

        println!("{}", self.weights.len());

        // iterate backwards through network to compute hidden layer errors
        for i in self.weights.len()..0 {
            println!("made it here");


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
            let mut theta_grad = d.transpose() * (a / (m as f64));
            let theta_0 = zero_first_col(&self.weights[i]);
            theta_grad += theta_0 * (lambda/(m as f64));
            self.gradients.push(theta_grad);
        }
    }

    /// use the trained weights to generate predictions
    pub fn predict(&mut self, X: Matrix<f64>) -> Vec<i64> {
        let h = self.forward_prop(&X);
        row_max(&h).iter().map(|x| x+1).collect::<Vec<i64>>()
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
