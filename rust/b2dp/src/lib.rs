//! # Base-2 Differential Privacy Crate
//! Implements the exponential mechanism and other utilities for base-2 
//! Differential Privacy, based on [Ilvento '19](https://arxiv.org/abs/1912.04222).
//! 
//! **Status:** active development, reference implementation only. Not intended for uses other than research.
//! 
//! ## Background
//! Although the exponential mechanism does not directly reveal the result of inexact
//! floating point computations, it has been shown to be vulnerable to attacks based
//! on rounding and no-op addition behavior of floating point arithmetic. To prevent
//! these issues, base-2 differential privacy uses arithmetic with base 2, rather than 
//! base e, allowing for an exact implementation. This crate implements the base-2 exponential
//! mechanism as well as useful base-2 DP utilities for parameter conversion.
//! ## Details
//! ### Example Usage
//! **Converting a base-e parameter to base-2**
//! ```
//! use b2dp::Eta;
//! let epsilon = 1.25;
//! let eta = Eta::from_epsilon(epsilon).unwrap();
//! ```
//! **Running the exponential mechanism**
//! 
//! Run the exponential mechanism with utility function `utility_fn`.
//! ```
//! use b2dp::{exponential_mechanism, Eta, GeneratorOpenSSL};
//! 
//! fn util_fn (x: &u32) -> f64 {
//!     return ((*x as f64)-0.0).abs();
//! }
//! let eta = Eta::new(1,1,1).unwrap(); // Construct a privacy parameter
//! let utility_min = 0; // Set bounds on the utility and outcomes
//! let utility_max = 10;
//! let max_outcomes = 10;
//! let rng = GeneratorOpenSSL {};
//! let outcomes: Vec<u32> = (0..max_outcomes).collect();
//! let optimized_sample = false;
//! let empirical_precision = false;
//! let sample = exponential_mechanism(eta, &outcomes, util_fn, 
//!                                     utility_min, utility_max, 
//!                                     max_outcomes,
//!                                     rng, 
//!                                     optimized_sample, 
//!                                     empirical_precision).unwrap();
//! ```
//! **Scaling based on utility function sensitivity**
//! Given a utility function with sensitivity `alpha`, the `exponential_mechanism` 
//! implementation is `2*alpha*ln(2)*eta` base-e DP. To explicitly scale by `alpha`
//! the caller can either modify the `eta` used or the utility function.
//! ```
//! use b2dp::{exponential_mechanism, Eta, GeneratorOpenSSL};
//! 
//! // Scale the privacy parameter to account for the utility sensitivity
//! let epsilon = 1.25;
//! let eta = Eta::from_epsilon(epsilon).unwrap();
//! let alpha = 2.0;
//! let eta_scaled = Eta::from_epsilon(epsilon/alpha).unwrap();
//! ``` 
//! ```
//! 
//! // Or scale the utility function to reduce sensitivity
//! let alpha = 2.0;
//! 
//! fn util_fn (x: &u32) -> f64 {
//!     return (2.0*(*x as f64)-0.0).abs();
//! }
//! let scaled_utility_fn = |x: &f64| -> f64 { *x/alpha };
//! ```


/// Base-2 Differential Privacy Utilities
pub mod utilities;
/// Base-2 Differential Privacy Mechanisms
pub mod mechanisms;

pub use utilities::params::Eta as Eta;
pub use utilities::exactarithmetic::randomized_round as randomized_round;
pub use utilities::exactarithmetic::normalized_sample as normalized_sample;
pub use utilities::randomness::GeneratorOpenSSL;
pub use mechanisms::exponential::exponential_mechanism as exponential_mechanism;