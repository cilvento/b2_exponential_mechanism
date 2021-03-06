//! Implements methods requiring exact arithmetic, and encapsulates all
//! `unsafe` code to access `mpfr::flags` to determine whether coputations
//! are exact.


use rug::{ops::Pow, rand::ThreadRandGen, rand::ThreadRandState, Float};

use super::params::Eta;
use gmp_mpfr_sys::mpfr;

/// Randomized Rounding
/// ## Arguments
///   * `x`: the value to round
///   * `arithmetic_config`: the arithmetic configuration to use
/// ## Returns
/// `x` rounded to the nearest smaller or larger integer by drawing a random value
/// `rho` in `[0,1]` and rounding up if `rho > x_fract`, rounding down otherwise.
pub fn randomized_round<R: ThreadRandGen>(
    x: f64,
    arithmetic_config: &mut ArithmeticConfig,
    mut rng: R,
) -> i64 {
    // if x is already integer, return it
    if x.trunc() == x { return x as i64; }
    // Get fractional part of x
    let x_fract = x.fract();
    let x_trunc = x.trunc() as i64;
    // Draw a random value
    let mut rand_state = ThreadRandState::new_custom(&mut rng);
    let rho = Float::with_val(
        arithmetic_config.precision,
        Float::random_bits(&mut rand_state),
    );
    if rho > x_fract {
        return x_trunc;
    } else {
        return x_trunc + 1;
    }
}

/// Determine smallest `k` such that `2^k >= total_weight`.
fn get_power_bound(total_weight: &Float, arithmetic_config: &mut ArithmeticConfig) -> i32 {
    let mut k: i32 = 0;
    if *total_weight > 1 {
        // increase `k` until `2^k >= total_weight`.
        let mut two_exp_k = Float::i_pow_u(2, k as u32);
        while Float::with_val(arithmetic_config.precision, two_exp_k) < *total_weight {
            k += 1;
            two_exp_k = Float::i_pow_u(2, k as u32);
        }
    } else {
        let mut w = Float::with_val(arithmetic_config.precision, total_weight);
        while w <= 1 {
            k -= 1;
            w *= 2;
        }
        k += 1;
    }
    k
}

/// Normalized Weighted Sampling
/// Returns the index of the element sampled according to the weights provided.
/// Uses optimized sampling if `optimize` set to true.
/// ## Arguments
///   * `weights`: the set of weights to use for sampling
///   * `arithmetic_config`: the arithmetic config specifying precision
///   * `rng`: source of randomness.
///   * `optimize`: whether to optimize sampling, introducing a timing channel.
/// ## Known Timing Channels
/// This method has a known (but difficult to expoit) timing channel. To determine
/// the index corresponding to sample, the method iterates through cumulative weights
/// and terminates the loop when the index is found. This can be exploited in three ways:
///   * **Rejection probability:** if the adversary can control the total weight of the utilities
///     such that the probability of rejection in the random index generation stage changes,
///     the time needed for sampling will vary between adjacent databases. The difference in time
///     will depend on the speed of random number generation.
///   * **Optimized sampling:**
///     * **Ordering of weights:** if the adversary can change the ordering of the weights such
///       that the largest weights (most probable) weights are first under a certain condition,
///       and the largest weights are last if that condition doesn't hold, then the adversary
///       can use the time of normalized_sample to guess whether the condition holds.
///     * **Size of weights:** if the adversary can change the size of the weights such that if
///       a certain condition holds, the weight is more concentrated and if not the weight is less
///       concentrated, then the adversary can use the time taken by normalized_sample as a signal
///       for whether the condition holds.
pub fn normalized_sample<R: ThreadRandGen>(
                                            weights: &Vec<Float>,
                                            arithmetic_config: &mut ArithmeticConfig,
                                            mut rng: R,
                                            optimize:bool,
                                        ) -> Result<usize, &'static str> {
    // Compute the total weight
    let total_weight = Float::with_val(arithmetic_config.precision, Float::sum(weights.iter()));
    // Determine smallest `k` such that `2^k > total_weight`
    let k = get_power_bound(&total_weight, arithmetic_config);
    // Initialize the random state
    let mut rand_state = ThreadRandState::new_custom(&mut rng);

    let mut t = Float::with_val(arithmetic_config.precision, &total_weight);
    t += 1; // ensure that the initial `t` is larger than `total_weight`.
    while t > total_weight {
        t = Float::with_val(
            arithmetic_config.precision,
            Float::random_bits(&mut rand_state),
        );
        // Multiply by 2^k to scale
        let two_pow_k = Float::with_val(arithmetic_config.precision, Float::i_exp(1, k));
        t = t * two_pow_k;
    }
    let mut cumulative_weight = Float::with_val(arithmetic_config.precision, 0);

    let mut index: Option<usize> = None;
    // Iterate through the weights until the cumulative weight is greater than or equal to `t`
    for i in 0..weights.len() {
        let next_weight = Float::with_val(arithmetic_config.precision, &weights[i]);
        cumulative_weight += next_weight;
        if cumulative_weight >= t {
            // This is the index to return
            if index.is_none() { index = Some(i); if optimize {return Ok(i);}}
        }

    }

    if index.is_some() { return Ok(index.unwrap()); }

    // Return an error if we are unable to sample
    // Caller can choose an index at random if needed
    Err("Unable to sample.")
}


/// The exact arithmetic configuration. Includes the precision of all
/// mechanism arithmetic and status bits indicating if any inexact
/// arithmetic has been performed.
/// The ArithmeticConfig implementation encapsulates all `unsafe` calls to
/// `mpfr`.
#[derive(Debug)]
pub struct ArithmeticConfig {
    /// The required precision (computed based on other parameters)
    pub precision: u32,
    /// Whether an inexact operation has been performed in the scope of
    /// this config
    pub inexact_arithmetic: bool, 
    /// Whether the code is currently in an exact scope
    exact_scope: bool,
}

impl ArithmeticConfig {
    /// A basic arithmetic_config with default precision
    pub fn basic() -> Result<ArithmeticConfig, &'static str> {
        let p ;//= 53;
        unsafe {p = mpfr::get_default_prec() as u32;}
        let config = ArithmeticConfig {precision: p, inexact_arithmetic: false, exact_scope: false};
        Ok(config)
    }

    pub unsafe fn get_empirical_precision(eta: &Eta,
                                utility_min: i64,
                                utility_max: i64,
                                max_outcomes: u32,
                                max_precision: u32,) 
                                -> Result<u32,&'static str> 
    {
    
        let mut p = mpfr::get_default_prec() as u32;
        // Get the base with the default precision
        let mut _base_test = eta.get_base(p);

        while ArithmeticConfig::check_mpfr_flags().is_err() {
            p *= 2;
            // Check if the precision has exceeded the maximum allowed
            if p > max_precision {
                return Err("Maximum precision exceeded.");
            }
            _base_test = eta.get_base(p);
        }
        // Check that we can compute the base with the current precision.
        mpfr::clear_flags();
        let base_result = eta.get_base(p);

        ArithmeticConfig::check_mpfr_flags()?;
        let base = &base_result.unwrap();

        // Loop until we can successfully evaluate the test function.
        mpfr::clear_flags(); // clear flags
        mpfr::set_inexflag(); // set the inexact flag
        let mut opt_p: Option<u32> = None;
        while ArithmeticConfig::check_mpfr_flags().is_err() {
            mpfr::clear_flags();
            // Increase the precision and update the base to the new precision
            // Only double if we haven't tried at this precision yet.
            if opt_p.is_none() { opt_p = Some(p); } 
            else { p *= 2; }
            let new_base = &Float::with_val(p, base);
            // Check if the precision has exceeded the maximum allowed
            if p > max_precision {
                return Err("Maximum precision exceeded.");
            }

            for u in utility_min..(utility_max+1) {
                let max_weight = Float::with_val(p, new_base.pow(utility_min)).ceil();
                let u_weight = Float::with_val(p, new_base.pow(u));
                let _combination = Float::with_val(p, max_weight * max_outcomes + u_weight);
            }
            
        }

        ArithmeticConfig::check_mpfr_flags()?;
        Ok(p)
    }


    /// Initialize an ArithmeticConfig for the base-2 exponential mechanism.
    /// This method empirically determines the precision required to compute a linear
    /// combination of at most `max_outcomes` weights in the provided utility range.
    /// Note that the precision to create Floats in rug/mpfr is given as a `u32`, but the
    /// sizes (min, max, etc) of precision returned (e.g. `mpfr::PREC_MAX`) are `i64`.
    /// We handle this by explicitly checking that `mpfr::PREC_MAX` does not exceed the
    /// maximum value for a `u32` (this should never happen, but we check anyway).
    /// ## Arguments
    ///   * `eta`: the base-2 privacy parameter
    ///   * `utility_min`: the minimum utility permitted by the mechanism (highest possible weight)
    ///   * `utility_max`: the maximum utility permitted by the mechanism (lowest possible weight)
    ///   * `max_outcomes`: the maximum number of outcomes permitted by this instance of the exponential
    ///                     mechanism.
    /// ## Returns
    /// Returns an ArithmeticConfig with sufficient precision to carry out the operations for the 
    /// exponential mechanism with the given parameters.
    /// ## Errors
    /// Returns an error if sufficient precision cannot be determined.
    pub fn for_exponential(
                            eta: &Eta,
                            utility_min: i64,
                            utility_max: i64,
                            max_outcomes: u32,
                            empirical_precision: bool,
                        ) -> Result<ArithmeticConfig, &'static str> 
    {
        let p: u32;
        //let empirical_precision = true;

        unsafe {
            // Clear the flags
            mpfr::clear_flags();

            // Check that the maximum precision does not exceed the maximum value of a
            // u32. Precision for Float::with_val(precision: u32, val) requires a u32.
            let mut max_precision = u32::max_value();
            if mpfr::PREC_MAX < max_precision as i64 {
                max_precision = mpfr::PREC_MAX as u32;
            }
            
            if !empirical_precision{
                let mut bx =  (eta.x as f32).log2().ceil() as u32;
                if bx < 1 { bx = 1;}
                let mut um = utility_max.abs();
                if um < 1 { um = 1; }
                if utility_min.abs() < 1 { um += 1; }
                else { um += utility_min.abs(); }
                p = (um as u32)*(eta.z*(eta.y+bx)) as u32 + max_outcomes;
                if p > max_precision {return Err("Maximum precision exceeded."); }
            }
            else 
            {
                p = ArithmeticConfig::get_empirical_precision(&eta, utility_min, utility_max, max_outcomes, max_precision)?;
            }
        } // end unsafe block

        let config = ArithmeticConfig {
            precision: p,
            inexact_arithmetic: false,
            exact_scope: false,
        };
        Ok(config)
    }

    /// Check the current state of the flags
    pub fn check_mpfr_flags() -> Result<(), &'static str> {
        unsafe {
            let flags = mpfr::flags_save();
            if flags > 0 {
                if mpfr::inexflag_p() > 0 {
                    return Err("Inexact arithmetic.");
                } else {
                    return Err("Arithmetic error other than inexact (see mpfr::flags)");
                }
            }
        }
        Ok(())
    }

    /// Returns whether the config is currently in a valid state
    pub fn is_valid(&self) -> bool {
        return !self.inexact_arithmetic;
    }

    /// Invalidates the config
    pub fn invalidate(&mut self) {
        self.inexact_arithmetic = true;
    }

    /// Enter exact arithmetic scope.
    /// This method clears `mpfr` flags if not currently in an `exact_scope`.
    /// # Returns
    ///   * `OK(())` if the scope is successfully entered
    ///   * `Err` if the scope is alread invalid
    pub fn enter_exact_scope(&mut self) -> Result<(), &'static str> {
        if self.is_valid() == false {
            // inexact arithmetic has already occurred
            return Err("ArithmeticConfiguration invalid.");
        }
        if !self.exact_scope {
            unsafe {
                mpfr::clear_flags();
            }
            // set the exact_scope flag
            self.exact_scope = true;
        }

        return ArithmeticConfig::check_mpfr_flags();
    }

    /// Exit the exact arithmetic scope.
    /// **Must be called after any arithmetic operations are performed which should be exact.**
    /// **Must be paired with `enter_exact_scope` to ensure that flags aren't misinterpreted.**
    /// This method checks the `mpfr` flag state, and returns whether
    /// the scope is still valid. Also sets the `inexact` property.
    /// This method does **not** reset the `mpfr` flags.
    /// ## Returns
    ///   * `OK(())` if the configuration reports than no inexact arithmetic was performed
    ///   * `Err` if the configuration is invalid (inexact arithmetic performed)
    pub fn exit_exact_scope(&mut self) -> Result<(), &'static str> {
        if self.is_valid() == false {
            // Error has already occurred
            return Err("ArithmeticConfiguration invalid.");
        }
        if self.exact_scope == false {
            return Err("Not in exact scope.");
        }

        let result = ArithmeticConfig::check_mpfr_flags();
        if result.is_err() {
            self.invalidate();
        }
        // set the exact_scope status to false
        self.exact_scope = false;
        return result;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::randomness::GeneratorOpenSSL;
    use rug::{ops::Pow};

    #[test]
    fn test_power_bound() {
        // Generate an arithmetic config
        let eta = &Eta::new(1, 1, 1).unwrap();
        let utility_min = 0;
        let utility_max = 100;
        let max_outcomes = 10;
        let arithmetic_config_result = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            false,
        );
        assert!(arithmetic_config_result.is_ok());
        let mut arithmetic_config = arithmetic_config_result.unwrap();

        arithmetic_config.enter_exact_scope().unwrap();
        let mut x = Float::with_val(arithmetic_config.precision, 1.25);
        let mut r = get_power_bound(&x, &mut arithmetic_config);
        assert_eq!(r, 1);

        let y = Float::with_val(arithmetic_config.precision, 0.35);
        let s = get_power_bound(&y, &mut arithmetic_config);
        assert_eq!(s, -1);

        x = Float::with_val(arithmetic_config.precision, 0.75);
        r = get_power_bound(&x, &mut arithmetic_config);
        assert_eq!(r, 0);

        x = Float::with_val(arithmetic_config.precision, 5.75);
        r = get_power_bound(&x, &mut arithmetic_config);
        assert_eq!(r, 3);

        x = Float::with_val(arithmetic_config.precision, 0.0625); // 1/16 = 2^(-4)
        r = get_power_bound(&x, &mut arithmetic_config);
        assert_eq!(r, -4);

        x = Float::with_val(arithmetic_config.precision, 16);
        r = get_power_bound(&x, &mut arithmetic_config);
        assert_eq!(r, 4);

        arithmetic_config.exit_exact_scope().unwrap();
    }

    /// Test flag behavior of mpfr
    #[test]
    fn test_flags() {
        let precision = 53;
        // Use an unsafe block
        unsafe {
            // clear the flags
            mpfr::clear_flags();
            let mut flags = mpfr::flags_save();
            assert_eq!(flags, 0);

            // divide 1 by 3 to get an inexact result
            let x = Float::with_val(precision, 1.0);
            let y = Float::with_val(precision, 3.0);
            let _z = x / y;

            // Test the specific flag value
            flags = mpfr::flags_save();
            assert_eq!(flags, 8);

            // Test the inexflag directly
            assert_eq!((mpfr::inexflag_p() > 0), true);
            // Confirm other flags not set
            assert_eq!(
                (mpfr::underflow_p()
                    + mpfr::overflow_p()
                    + mpfr::divby0_p()
                    + mpfr::nanflag_p()
                    + mpfr::erangeflag_p())
                    > 0,
                false
            );

            // Do some exact arithmetic
            let a = Float::with_val(precision, 5.0);
            let b = Float::with_val(precision, 6.0);
            let c = a + b;
            // Test the specific flag value is preserved
            flags = mpfr::flags_save();
            assert_eq!(flags, 8);

            // Test the inexflag directly
            assert_eq!((mpfr::inexflag_p() > 0), true);
            // Confirm other flags not set
            assert_eq!(
                (mpfr::underflow_p()
                    + mpfr::overflow_p()
                    + mpfr::divby0_p()
                    + mpfr::nanflag_p()
                    + mpfr::erangeflag_p())
                    > 0,
                false
            );

            // Clear the flags and do some exact arithmetic
            mpfr::clear_flags();
            let d = Float::with_val(precision, 7.0);
            let _e = d * c;
            flags = mpfr::flags_save();
            assert_eq!(flags, 0);

            // Check that creating a value too large for the given precision
            // results in flags
            mpfr::clear_flags();
            let f = Float::with_val(precision, i64::max_value());
            // Confirm that precision isn't modified
            assert_eq!(f.prec(), precision);
            flags = mpfr::flags_save();
            assert!(flags > 0);
            assert!(mpfr::inexflag_p() > 0);
            let g = Float::with_val(precision, i64::max_value());
            // Clear the flags
            let h = f * g;
            assert_eq!(h.prec(), precision);
            flags = mpfr::flags_save();
            assert!(flags > 0);

            // Check overflow behavior
            mpfr::clear_flags();
            let max_u_precision = 3;
            let i = Float::with_val(max_u_precision, 16);
            assert_eq!(i.prec(), max_u_precision);
            let j = Float::with_val(max_u_precision, i + 2);
            assert_eq!(j.prec(), max_u_precision);
            assert!(j - 2 != 16);
            flags = mpfr::flags_save();
            assert!(flags > 0);
            assert!(mpfr::inexflag_p() > 0); // This sets the inexact flag rather than overflow

            // Check precision inheritance behavior
            // Addition results in a Float with precision of the first
            // element in the sum.
            mpfr::clear_flags();
            let k = Float::with_val(max_u_precision, 16);
            assert_eq!(k.prec(), max_u_precision);
            let l = Float::with_val(max_u_precision + 1, 20);
            assert_eq!(l.prec(), max_u_precision + 1);
            let m = k + l; // Switching the order of l and k will cause the test to fail.
            flags = mpfr::flags_save();
            assert!(flags > 0);
            assert_eq!(m.prec(), max_u_precision);
        }
    }
    
    
    #[test]
    fn test_high_precision_arithmetic_config_for_exponential() {
        let eta = &Eta::new(1, 2, 3).unwrap();
        let utility_min = 0;
        let utility_max = 2i64.pow(10);
        let max_outcomes = 2u32.pow(8);
        let arithmetic_config_result = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            false
        );
        assert!(arithmetic_config_result.is_ok());
        let arithmetic_config = arithmetic_config_result.unwrap();
        assert!(arithmetic_config.precision >= 6000);
        //let bx = 1;
        

        let emp_arithmetic_config = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            true
        ).unwrap();
        assert!(arithmetic_config.precision >= emp_arithmetic_config.precision);
        //let hyp_prec = (utility_min as u32 + utility_max as u32) * eta.z *(eta.y + bx) + max_outcomes;
        //assert!(arithmetic_config.precision<= hyp_prec);
    }



    #[test]
    fn test_arithmetic_config_for_exponential() {
        let eta = &Eta::new(1, 1, 1).unwrap();
        let utility_min = -100;
        let utility_max = 100;
        let max_outcomes = 10;
        let arithmetic_config_result = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            false
        );
        assert!(arithmetic_config_result.is_ok());
        let arithmetic_config = arithmetic_config_result.unwrap();
        assert!(arithmetic_config.precision >= 8);
    }

    #[test]
    fn test_exact_scope() {
        let eta = &Eta::new(1, 1, 1).unwrap();
        let utility_min = -100;
        let utility_max = 0;
        let max_outcomes = 10;
        let arithmetic_config_result = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            false,
        );
        assert!(arithmetic_config_result.is_ok());
        let mut arithmetic_config = arithmetic_config_result.unwrap();
        let working_precision = arithmetic_config.precision;

        // Test good behavior in exact scope
        // Enter exact scope
        let enter1 = arithmetic_config.enter_exact_scope();
        assert!(enter1.is_ok());

        // Do some arithmetic that should all be exact
        // Do some exact arithmetic
        let base_result = eta.get_base(working_precision);
        let base = &base_result.unwrap();
        let mut weight_sum = Float::with_val(working_precision, 0);
        for _i in 0..max_outcomes {
            let new_weight_sum =
                weight_sum + Float::with_val(working_precision, base.pow(utility_max));
            weight_sum = new_weight_sum;
        }
        //let x = Float::with_val(working_precision,base.pow(2));
        assert_eq!(10, weight_sum);
        // Exit exact scope
        let exit1 = arithmetic_config.exit_exact_scope();
        assert!(exit1.is_ok());

        // Test bad behavior in exact scope
        // Enter exact scope
        let enter2 = arithmetic_config.enter_exact_scope();
        assert!(enter2.is_ok());

        // Do some arithmetic that should raise flags
        let x = Float::with_val(working_precision, 1.0);
        let y = Float::with_val(working_precision, 3.0);
        let _z = x / y;

        // Exit exact scope
        let exit2 = arithmetic_config.exit_exact_scope();
        assert!(exit2.is_err());
        assert!(arithmetic_config.inexact_arithmetic);

        // Try to enter after bad behavior
        // Enter exact scope
        let enter2 = arithmetic_config.enter_exact_scope();
        assert!(enter2.is_err());
    }
    #[test]
    fn test_optimized_normalized_sample() {
        // Generate an arithmetic config
        let eta = &Eta::new(1, 1, 1).unwrap();
        let utility_min = -1;
        let utility_max = 10;
        let max_outcomes = 10;
        let rng = GeneratorOpenSSL {};
        let arithmetic_config_result = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            false,
        );
        assert!(arithmetic_config_result.is_ok());
        let mut arithmetic_config = arithmetic_config_result.unwrap();

        arithmetic_config.enter_exact_scope().unwrap();
        let n = 1000;
        // Construct a vector of equal weights and test we are getting
        // approximately equal probabilities
        let a = Float::with_val(arithmetic_config.precision, 1);
        let b = Float::with_val(arithmetic_config.precision, 1);
        let c = Float::with_val(arithmetic_config.precision, 1);
        let mut weights: Vec<Float> = Vec::new();
        weights.push(a);
        weights.push(b);
        weights.push(c);
        let mut counts = [0; 3];
        for _i in 0..n {
            let j = normalized_sample(&weights, &mut arithmetic_config, rng, true).unwrap();
            counts[j] += 1;
        }
        println!("{:?}", counts);

        arithmetic_config.exit_exact_scope().unwrap();

        let mut probs = [0.0; 3];
        for i in 0..counts.len() {
            probs[i] = (counts[i] as f64) / (n as f64);
            assert!(probs[i] - 0.333 < 0.05);
        }
    }


    #[test]
    fn test_normalized_sample() {
        // Generate an arithmetic config
        let eta = &Eta::new(1, 1, 1).unwrap();
        let utility_min = -1;
        let utility_max = 10;
        let max_outcomes = 10;
        let rng = GeneratorOpenSSL {};
        let arithmetic_config_result = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            false,
        );
        assert!(arithmetic_config_result.is_ok());
        let mut arithmetic_config = arithmetic_config_result.unwrap();

        arithmetic_config.enter_exact_scope().unwrap();
        let n = 1000;
        // Construct a vector of equal weights and test we are getting
        // approximately equal probabilities
        let a = Float::with_val(arithmetic_config.precision, 1);
        let b = Float::with_val(arithmetic_config.precision, 1);
        let c = Float::with_val(arithmetic_config.precision, 1);
        let mut weights: Vec<Float> = Vec::new();
        weights.push(a);
        weights.push(b);
        weights.push(c);
        let mut counts = [0; 3];
        for _i in 0..n {
            let j = normalized_sample(&weights, &mut arithmetic_config, rng, false).unwrap();
            counts[j] += 1;
        }

        let mut probs = [0.0; 3];
        for i in 0..counts.len() {
            probs[i] = (counts[i] as f64) / (n as f64);
            assert!(probs[i] - 0.333 < 0.05);
        }

        // Construct a vector with different weights, and confirm that
        // we still see low probability weights sometimes.
        weights.push(Float::with_val(arithmetic_config.precision, 0.0625));
        let mut new_counts = [0; 4];
        for _i in 0..n {
            let j = normalized_sample(&weights, &mut arithmetic_config, rng, false).unwrap();
            new_counts[j] += 1;
        }

        let mut new_probs = [0.0; 4];
        let new_expected_probs = [0.327, 0.327, 0.327, 0.02];
        for i in 0..counts.len() {
            new_probs[i] = (new_counts[i] as f64) / (n as f64);
            assert!(new_probs[i] - new_expected_probs[i] < 0.05);
        }

        arithmetic_config.exit_exact_scope().unwrap();
    }

    #[test]
    fn test_randomized_round() {
        // Generate an arithmetic config
        let eta = &Eta::new(1, 1, 1).unwrap();
        let utility_min = -100;
        let utility_max = 0; 
        let max_outcomes = 10;
        let rng = GeneratorOpenSSL {};
        let arithmetic_config_result = ArithmeticConfig::for_exponential(
            eta,
            utility_min,
            utility_max,
            max_outcomes,
            false,
        );
        assert!(arithmetic_config_result.is_ok());
        let mut arithmetic_config = arithmetic_config_result.unwrap();

        // Enter Exact scope
        arithmetic_config.enter_exact_scope().unwrap();
        let x = 1.25;
        let r = randomized_round(x, &mut arithmetic_config, rng);
        assert!((x - (r as f64)).abs() < 1.0);

        // Exit exact scope
        arithmetic_config.exit_exact_scope().unwrap();
    }
}
