//! Implements `ThreadRandGen` wrapper(s) for randomness sources.

use rug::rand::ThreadRandGen;
use openssl::rand::rand_bytes;


/// Random Number Generator using OpenSSL for randomness
#[derive(Debug, Copy, Clone)]
pub struct GeneratorOpenSSL;
impl ThreadRandGen for GeneratorOpenSSL {
    fn gen(&mut self) -> u32 {
        let mut buf = [0; 4]; // empty buffer for four bytes
        rand_bytes(& mut buf).unwrap();
        let result = u32::from_be_bytes(buf);
        return result;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::{Float, rand::ThreadRandState};
    #[test]
    fn test_rand_basic() {
        let p = 10;
        let mut rng = GeneratorOpenSSL {};
        let mut rand_state = ThreadRandState::new_custom(& mut rng);
        let a = Float::with_val(p, Float::random_bits(& mut rand_state));
        let b = Float::with_val(p,&a*2);
        assert!(b <= 2);
        assert!(a <= 1);
    }

}