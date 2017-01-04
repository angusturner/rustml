/// this module will contain functions for importing and exporting data
extern crate csv;
extern crate rustc_serialize;
extern crate rulinalg as rl;

use std::fs::File;
use std::io::prelude::*;
use std::io::LineWriter;

use self::rl::matrix::{Matrix, BaseMatrix};
use self::rustc_serialize::{Encodable, Decodable};

/// read CSV into a matrix
pub fn read_csv<T>(file_path: &str) -> Matrix<T>
    where T: Decodable
{
    // initialise some stuff
    let mut rdr = csv::Reader::from_file(file_path).unwrap().has_headers(false);
    let mut out: Vec<T> = vec![]; // output vector
    let mut m: usize = 0; // number of rows
    let mut n: usize = 0; // number of cols

    // iterate through records
    for record in rdr.decode() {
        // decode row into a vector
        let mut row: Vec<T> = match record {
            Ok(res) => res,
            Err(err) => panic!("failed to read {}: {}", m, err),
        };

        // compute number of columns
        n = match n {
            0 => row.len(),
            _ => n
        };

        // append row to output vector
        out.append(&mut row);

        // increment number of rows
        m += 1usize;
    }

    // reshape data into matrix
    Matrix::new(m, n, out)
}

/// write a matrix row-by-row into a CSV
pub fn matrix_to_csv<T>(mat: &Matrix<T>, file_path: &str)
    where T: Encodable
{
    // create/overwrite the file
    let file = File::create(file_path).unwrap();

    // create a LineWriter, which buffers output and writes it on every new-line
    let mut file = LineWriter::new(file);

    // encode the matrix as csv
    let mut wtr = csv::Writer::from_memory();
    for row in mat.row_iter() {
        let record = (*row).iter().collect::<Vec<&T>>();
        let result = wtr.encode(record);
        assert!(result.is_ok());
    }

    // write to file
    for &byte in wtr.as_bytes() {
        file.write(&[byte]).unwrap();
    }
}
