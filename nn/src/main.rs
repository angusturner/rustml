extern crate csv;
extern crate rustc_serialize;
extern crate nalgebra as na;

use na::{DMatrix};

fn main() {
    let X = read_csv("input.csv".to_string());
    println!("{:?}", X);
}

/// read CSV into a matrix
fn read_csv(file_path: String) -> DMatrix<f64> {
    // read CSV into a row vector
    let mut rdr = csv::Reader::from_file(file_path).unwrap();
    let mut X: Vec<f64> = vec![]; // design matrix
    let mut m: usize = 0; // number of training examples
    let mut n: usize = 0;
    //
    for record in rdr.decode() {
        // construct ith training example by prepending a row of 1's
        let mut line: Vec<f64> = record.unwrap();
        let mut x_i: Vec<f64> = vec![1f64];
        x_i.append(&mut line); // prepend each row with a 1
        n = x_i.len();
        X.append(&mut x_i);
        m += 1usize;
    }

    DMatrix::from_row_vector(m, n, &X)
}
