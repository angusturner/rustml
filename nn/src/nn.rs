extern crate rulinalg as rl;

use self::rl::matrix::{Matrix, BaseMatrix};
use utils::{dims};

pub struct NN {
    num_inputs: usize,
    num_outputs: usize,
    num_hidden_layers: usize,
    weights: Vec<Matrix<f64>>
}

impl NN {
    /// static constructor function
    pub fn new(num_inputs: usize, num_outputs: usize) -> NN {
        NN {
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            num_hidden_layers: 0usize,
            weights: vec![]
        }
    }

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
            weights: new_weights
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
            weights: new_weights
        }
    }

    /// train the neural net using gradient descent
    /// X - the design matrix of training examples
    /// y - the response matrix (output values mapped to one-hot vectors)
    /// lambda - regularization parameter
    /// alpha - learning rate
    /// max_iters - number of iterations of gradient descent to perform
    fn train(&self, X: Matrix<f64>, y: Matrix<f64>, lambda: f64, alpha: f64, max_iters: usize) -> bool {
        true
    }

    /// use the trained weights to generate predictions
    fn predict(&self, test_set: Matrix<f64>) {

    }
}
