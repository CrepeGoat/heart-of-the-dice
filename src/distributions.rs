use std::ops::Mul;

#[derive(PartialEq, Clone)]
pub struct Data<X, Y> {
    x: X,
    y: Y,
}

fn convolve<T>(v1: &[T], v2: &[T], result: &mut [T]) -> ()
where
    T: Mul<T>,
{
    if !(result.len() == v1.len() + v2.len() - 1) {
        panic!("incorrect array lengths")
    }
    // TODO
    v1.windows(v2.len())
        .map(|window| window.iter().zip(v2).map(|(x, y)| x * y).sum())
        .collect()
}

/// A sequence of numbers, offset from zero by a set amount.
///
/// The offset avoids having to manually offset data by explicitly storing zeros
/// in the arrays, and also allows arrays to start before zero.
#[derive(PartialEq)]
pub struct SequenceWithOffset<X, Y> {
    seq: Vec<Y>,
    offset: X,
}

impl<X, Y> SequenceWithOffset<X, Y> {
    pub fn new() -> Self {}

    fn index_end(self) -> X {
        self.offset + self.seq.len()
    }

    pub fn add(self, other: Self) -> Self {}

    pub fn bias_by(self, value: X) -> Self {}

    pub fn scale_by(self, value: X) -> Self {}
}

pub fn roll_0<X, Y>() -> SequenceWithOffset<X, Y> {}
