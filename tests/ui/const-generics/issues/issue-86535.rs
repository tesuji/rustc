//@ run-pass
#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features, unused_variables)]

#[allow(dead_code)]
struct F<const S: &'static str>;
impl<const S: &'static str> X for F<{ S }> {
    const W: usize = 3;

    fn d(r: &[u8; Self::W]) -> F<{ S }> {
        let x: [u8; Self::W] = [0; Self::W];
        F
    }
}

pub trait X {
    const W: usize;
    fn d(r: &[u8; Self::W]) -> Self;
}

fn main() {}
