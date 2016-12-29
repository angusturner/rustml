#[macro_use]
extern crate rulinalg as rl;

use rl::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut, MatrixSliceMut};

#[allow(non_snake_case)]
fn main() {
    // construct a 3x3 matrix of f64
    let mut mat = matrix!(1.0, 2.0, 3.0;
                    4.0, 5.0, 6.0;
                    7.0, 8.0, 9.0);

    // take a slice into the matrix
    {
        // take a mutable matrix slice
        let mut slice = MatrixSliceMut::from_matrix(&mut mat, [1, 1], 2, 2);

        // edit the mutable slice
        slice[[0,1]] = 1f64;
    }

    // observe effect on original matrix
    println!("{:?}", mat[[0,0]]);
}
