//! Naive Mechanism Implementations for comparison
use rug::{Float, rand::ThreadRandGen, rand::ThreadRandState};
pub fn naive_exponential_mechanism<T, R: ThreadRandGen + Copy>(epsilon: f64, 
                                                               outcomes: &Vec<T>, 
                                                               utility: fn(&T) -> f64,
                                                               mut rng: R) -> Result<&T, &'static str> 
    {
        let mut weights: Vec<f64> = Vec::new();
        let mut rand_state = ThreadRandState::new_custom(&mut rng);
        // Compute Weights
        for i in 0..outcomes.len() {
            let mut u = -utility(&outcomes[i]);
            u *= epsilon;
            weights.push(u.exp());
        }

        // Normalize weights
        let total_weight: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|x| x/total_weight).collect();


        // Draw a random number
        let t = Float::with_val(53,Float::random_cont(&mut rand_state)).to_f64();
        
        // Find the associated index
        let mut cumulative_weight = 0.0;
        let mut index: Option<usize> = None;
        for i in 0..outcomes.len(){
            cumulative_weight += normalized_weights[i];
            if cumulative_weight >= t {
                if index.is_none() {index = Some(i);}
            }
        }
        if index.is_none(){
            return Err("Failed to sample");
        }
        return Ok(&outcomes[index.unwrap()]);

   }


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::randomness::GeneratorOpenSSL;

    #[test]
    fn test_naive(){
        let epsilon = 1.3;
        let outcomes = (0..10).collect();
        let rng = GeneratorOpenSSL {};
        fn utility_fn(x: &i64)->f64 { *x as f64 }
        let result = naive_exponential_mechanism(epsilon, &outcomes, utility_fn, rng);
        assert!(result.is_ok());
    }
}