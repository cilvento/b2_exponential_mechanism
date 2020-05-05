//! Implements an approximation of the Laplace mechanims via the base-2 exponential mechanism. 

use rug::rand::ThreadRandGen;
use crate::mechanisms::exponential::exponential_mechanism;
use crate::utilities::params::Eta;

/// Implements clamped Laplace mechanism via the base-2 exponential mechanism
/// Outputs an `f64` in the range `[lower_bound, upper_bound]` of the form
/// `lower_bound + gamma*i`.
/// ## Arguments
///   * `eta`: the base-2 privacy parameter
///   * `lower_bound`: the lower bound of the range for the Laplace mechanism
///   * `upper_bound`: the upper bound of the range for the Laplace mechanism
///   * `target`: the target value (value to approximate)
///   * `gamma`: the granularity of the outcome space
///   * `rng`: a random number generator
/// ## Returns
/// Returns an `f64` within the specified bounds. 
/// ## Errors
/// Returns Err if any of the parameters are configured incorrectly or if inexact arithmetic
/// occurs. 
/// ## Example
/// Sample from the Laplace mechanism targeting `0.0` clamped to the range `[-10,10]` with
/// granularity `0.25`. 
/// ```
/// use b2dp::{mechanisms::laplace::clamped_laplace_mechanism, Eta, GeneratorOpenSSL};
/// 
/// let target = 0.0;
/// let eta = Eta::new(1,1,1).unwrap(); 
/// let lower_bound = -10.0;
/// let upper_bound = 10.0;
/// let gamma = 0.25;
/// let rng = GeneratorOpenSSL {};
/// let optimize = false;
/// let empirical_precision = false;
/// let sample = clamped_laplace_mechanism(eta, lower_bound, upper_bound, target, gamma, rng, optimize, empirical_precision).unwrap();
/// ```
pub fn clamped_laplace_mechanism<R: ThreadRandGen + Copy>(eta: Eta, lower_bound: f64, upper_bound: f64, target: f64, gamma: f64,
                                rng: R, optimize: bool, empirical_precision: bool) -> Result<f64, &'static str> 
    {
    // Check Parameters
    eta.check()?;
    if lower_bound >= upper_bound {return Err("lower_bound must be smaller than upper_bound");} 

    // Clamp the target
    let mut clamp_target = target;
    if clamp_target > upper_bound {clamp_target = upper_bound;}
    if clamp_target < lower_bound {clamp_target = upper_bound;}

        
    // Generate exponential mechanism parameters
    let mut outcomes: Vec<f64> = Vec::new();

    let mut x = lower_bound;

    while x <= upper_bound {
        outcomes.push(x);
        x += gamma;
    }
    let max_outcomes = outcomes.len();


    let laplace_utility = |x: &f64| -> f64 { (clamp_target - *x).abs() };

    let utility_max = upper_bound - lower_bound;

    let sample = exponential_mechanism(eta, &outcomes,
                                        laplace_utility, 
                                        0,
                                        utility_max as i64, 
                                        max_outcomes as u32,
                                        rng, 
                                        optimize, empirical_precision).unwrap();

    
    Ok(*sample)
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::randomness::GeneratorOpenSSL;
    #[test]
    fn test_laplace_mechanism_basic() {
        let eta = Eta::new(1,1,1).unwrap();
        let rng = GeneratorOpenSSL {};
        let sample = clamped_laplace_mechanism(eta,-10.0,10.0,1.0,0.25,rng,true, false);
        assert!(sample.is_ok());   
    }
}