//! Implements the base-2 exponential mechanism. 

use rug::{Float, ops::Pow, rand::ThreadRandGen};
use crate::utilities::exactarithmetic::{ArithmeticConfig, normalized_sample, randomized_round};
use crate::utilities::params::Eta;
/// The exponential mechanism configuration. Includes all parameters 
/// and information needed to derive the appropriate precision for the
/// mechanism.
#[derive(Debug)]
struct ExponentialConfig {
    /// The privacy parameter
    pub eta: Eta,
    /// The maximum utility (signed)
    pub utility_min: i64,
    /// The minimum utility (signed)
    pub utility_max: i64,
    /// The maximum size of the outcome space
    pub max_outcomes: u32,
    /// The arithmetic configuration
    arithmetic_config: ArithmeticConfig,
}

// Constructors
impl ExponentialConfig {
    /// Create a new context for the exponential mechanism.
    /// ## Arguments
    ///   * `eta`: the base-2 privacy parameter
    ///   * `utility_min`: the minimum utility permitted by the mechanism (highest possible weight)
    ///   * `utility_max`: the maximum utility permitted by the mechanism (lowest possible weight)
    ///   * `max_outcomes`: the maximum number of outcomes this instance exponential mechanism permits.
    /// ## Returns 
    /// An `ExponentialConfig` from the specified parameters or an error.
    /// ## Errors
    /// Returns an error if any of the parameters are mis-specified, or if sufficient precision cannot
    /// be determined.
    pub fn new(eta: Eta, utility_min: i64, utility_max: i64, 
               max_outcomes: u32, empirical_precision: bool)//, min_sampling_precision: u32)
            -> Result<ExponentialConfig, &'static str> {
        
        
        // Parameter sanity checking
        if utility_min > utility_max {
            return Err("utility_min must be smaller than utility_max");
        }
        if max_outcomes == 0 {
            return Err("Must provide a positive value for max_outcomes");
        }

        let arithmetic_config =  ArithmeticConfig::for_exponential(&eta, utility_min, utility_max, 
                                                                   max_outcomes, empirical_precision)?;

        // Construct the configuration with the precision we determined above
        let config = ExponentialConfig {
            eta, 
            utility_min, 
            utility_max, 
            max_outcomes, 
            arithmetic_config
        };
        Ok(config)
    }

    pub fn get_base(&self) -> Float {
        self.eta.get_base(self.arithmetic_config.precision).unwrap()
    }
}



/// Implements the base-2 exponential mechanism.
/// Utility convention is to take `-utility(o)`, and `utility_min` is therefore the highest
/// possible weight/maximum probability outcome. This mechanism does not scale based on
/// the sensitivity of the utility function. For a utility function with sensitivity `alpha`,
/// the mechanism is `2*alpha*eta` base-2 DP, and `2*alpha*ln(2)*eta` base-e DP. 
/// This function calls `enter_exact_scope()` and 
/// `exit_exact_scope()`, and therefore clears the `mpfr::flags` and does not preserve the 
/// incoming flag state. **The caller must ensure that `utility_min`, `utility_max`, `max_outcomes`
/// and `outcomes` are determined independently of the `utility` function.**
/// ## Arguments
///   * `eta`: the base-2 privacy parameter
///   * `outcomes`: the set of outcomes the mechanism chooses from
///   * `utility`: utility function operating on elements of `outcomes`
///   * `utility_min`: the minimum utility permitted by the mechanism (highest possible weight)
///   * `utility_max`: the maximum utility permitted by the mechanism (lowest possible weight)
///   * `max_outcomes`: the maximum number of outcomes permitted by the mechanism
///   * `rng`: a random number generator
/// ## Returns
/// Returns a reference to an element in `outcomes` sampled according to the base-2 exponential 
/// mechanism. 
/// ## Known Timing Channels
/// **This mechanism has known timing channels.** Please see 
/// [normalized_sample](../../utilities/exactarithmetic/fn.normalized_sample.html#known-timing-channels).
/// ## Errors
/// Returns Err if any of the parameters are configured incorrectly or if inexact arithmetic
/// occurs. 
/// ## Example
/// ```
/// use b2dp::{exponential_mechanism, Eta, GeneratorOpenSSL};
/// 
/// fn util_fn (x: &u32) -> f64 {
///     return ((*x as f64)-0.0).abs();
/// }
/// let eta = Eta::new(1,1,1).unwrap(); 
/// let utility_min = 0;
/// let utility_max = 10;
/// let max_outcomes = 10;
/// let rng = GeneratorOpenSSL {};
/// let optimize = true;
/// let empirical_precision = false;
/// let outcomes: Vec<u32> = (0..max_outcomes).collect();
/// let result = exponential_mechanism(eta, &outcomes, util_fn, 
///                                     utility_min, utility_max, 
///                                     max_outcomes,
///                                     rng, optimize, empirical_precision);
/// ```
pub fn exponential_mechanism<T, R: ThreadRandGen + Copy, F: Fn(&T)->f64>(eta: Eta, outcomes: &Vec<T>, utility: F,
                                utility_min: i64, utility_max: i64, 
                                max_outcomes: u32,
                                rng: R, optimize: bool, empirical_precision: bool) -> Result<&T, &'static str> {
    // Check Parameters
    eta.check()?;
    if (max_outcomes as usize) < outcomes.len() {
        return Err("Number of outcomes exceeds max_outcomes.");
    }

    // Generate an ExponentialConfig
    let mut exponential_config = ExponentialConfig::new(eta, utility_min, utility_max, 
                                                        max_outcomes, empirical_precision)?;
    
    // Compute Utilities
    let mut utilities = Vec::new();
    for o in outcomes.iter() {
        let mut u = utility(o);
        // clamp the utility to the allowed range
        if u > exponential_config.utility_max as f64 {
            u = exponential_config.utility_max as f64;
        }
        else if u < exponential_config.utility_min as f64 {
            u = exponential_config.utility_min as f64;
        }
        utilities.push(randomized_round(u, & mut exponential_config.arithmetic_config, rng));
    }

    // Enter exact scope
    exponential_config.arithmetic_config.enter_exact_scope()?;

    // get the base
    let base = &exponential_config.get_base();

    // Generate weights vector
    let mut weights = Vec::new();
    for u in utilities.iter() {
        let w = Float::with_val(exponential_config.arithmetic_config.precision, base.pow(u));
        weights.push(w);
    }

    // Sample
    let sample_index = normalized_sample(&weights, & mut exponential_config.arithmetic_config, rng, optimize)?;
    let sample = &outcomes[sample_index];

    // Exit exact scope
    exponential_config.arithmetic_config.exit_exact_scope()?;

    Ok(sample)
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::randomness::GeneratorOpenSSL;
    #[test]
    fn test_exponential_mechanism_basic() {
        let eta = Eta::new(1,1,1).unwrap();
        let utility_min = -100;
        let utility_max = 0;
        let max_outcomes = 10;
        let rng = GeneratorOpenSSL {};

        let num_samples = 1000;
        let num_outcomes = 5;
        let outcomes: Vec<u32> = (0..num_outcomes).collect();
        println!("Outcomes: {:?}", outcomes);
    
        fn util_fn (x: &u32) -> f64 {
            return (*x as f64)*2.0 ;
        }

        let outcome = exponential_mechanism(eta, &outcomes, util_fn, 
                                            utility_min, utility_max, 
                                            max_outcomes,
                                            rng, false, false);
        println!("Outcome: {:?}", outcome);

        let mut samples = [0;5];
        for _i in 0..num_samples {
            let sample = exponential_mechanism(eta, &outcomes, util_fn, 
                utility_min, utility_max, 
                max_outcomes,
                rng, false, false).unwrap();
            samples[*sample as usize] += 1;
        }
        println!("{:?}", samples);   
    }
}