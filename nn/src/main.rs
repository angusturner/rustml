extern crate csv;
extern crate rustc_serialize;
extern crate nalgebra as na;

use na::{DMatrix};

fn main() {
    // Read in the data
    let X = read_csv("input.csv", true); // design matrix
    let y = read_csv("output.csv", false); // response vector
    let y2 = read_csv("output_map.csv", false); // response matrix (class k -> [0..1..r])
    let theta1 = read_csv("theta1.csv", false); // weights mapping input to hidden layer
    let theta2 = read_csv("theta2.csv", false); // weights mapping hidden layer to output layer

    println!("{:?}", theta1);
}

/// Read CSV into a matrix
fn read_csv(file_path: &str, add_ones: bool) -> DMatrix<f64> {
    // initialise some stuff
    let mut rdr = csv::Reader::from_file(file_path).unwrap();
    let mut out: Vec<f64> = vec![]; // output vector
    let mut m: usize = 0; // number of rows
    let mut n: usize = 0; // number of cols

    // iterate through records
    for record in rdr.decode() {
        // decode row into a vector
        let mut row: Vec<f64> = record.unwrap();

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
    DMatrix::from_row_vector(m, n, &out)
}
