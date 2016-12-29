#![feature(test)]

extern crate test;
extern crate rulinalg as rl;

use rl::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};

// clone a Matrix
pub fn matrix_clone() {
    let mut ones: Matrix<f64> = Matrix::ones(5000, 400);
    let mut zeros: Matrix<f64> = Matrix::zeros(5000, 400);
    ones.clone();
}

// re-assign a matrix using the set_to() method
pub fn matrix_set_to() {
    let mut ones: Matrix<f64> = Matrix::ones(5000, 400);
    let mut zeros: Matrix<f64> = Matrix::zeros(5000, 400);
    zeros.set_to(ones);
}

// directly re-assign a Matrix
pub fn matrix_assign() {
    let mut ones: Matrix<f64> = Matrix::ones(5000, 400);
    let mut zeros: Matrix<f64> = Matrix::zeros(5000, 400);
    zeros = ones;
}

// get a MatrixSlice into a Matrix
pub fn matrix_get_slice() {
    let mut ones: Matrix<f64> = Matrix::ones(5000, 400);
    let mut zeros: Matrix<f64> = Matrix::zeros(5000, 400);
    let slice = MatrixSlice::from_matrix(&ones, [0, 0], ones.rows(), ones.cols());
}

// simply get a reference into a Matrix
pub fn matrix_get_ref() {
    let mut ones: Matrix<f64> = Matrix::ones(5000, 400);
    let mut zeros: Matrix<f64> = Matrix::zeros(5000, 400);
    let a = &ones;
}

// determine the time it takes to retrieve the number of matrix rows
pub fn matrix_get_dims() {
    let mut ones: Matrix<f64> = Matrix::ones(5000, 400);
    let mut zeros: Matrix<f64> = Matrix::zeros(5000, 400);
    ones.data().len();
    // ones.cols();
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use rl::matrix::Matrix;

    #[test]
    fn it_works() {}

    #[bench]
    fn bench_matrix_clone(b: &mut Bencher) {
        b.iter(|| matrix_clone());
    }

    #[bench]
    fn bench_matrix_set_to(b: &mut Bencher) {
        b.iter(|| matrix_set_to());
    }

    #[bench]
    fn bench_matrix_assign(b: &mut Bencher) {
        b.iter(|| matrix_assign());
    }

    #[bench]
    fn bench_matrix_get_slice(b: &mut Bencher) {
        b.iter(|| matrix_get_slice());
    }

    #[bench]
    fn bench_matrix_get_ref(b: &mut Bencher) {
        b.iter(|| matrix_get_ref());
    }

    #[bench]
    fn bench_matrix_get_dims(b: &mut Bencher) {
        b.iter(|| matrix_get_dims());
    }

}
