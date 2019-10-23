# The Base-2 Exponential Mechanism Reference Implementation
# author: C. Ilvento, cilvento@gmail.com
# status: reference implementation only

import gmpy2
from gmpy2 import mpfr, mpz

class ExpMech:
  """ The ExpMech base class. """
  # Check that the appropriate precision and context settings are in place
  def check_context(self):
    """ Check that the context settings including precision and inexact arithmetic
        flags are set properly. """
    if self.precision_set != True:
      return False
    ctx = gmpy2.get_context()
    if ctx.precision != self.context.precision:
        return False
    else:
        return (ctx.trap_inexact and ctx.trap_overflow and ctx.trap_erange \
               and ctx.trap_divzero and ctx.trap_invalid and ctx.trap_expbound \
               and ctx.trap_underflow)

  # A method to test whether the current precision is sufficient for intended usage.
  def check_precision(self):
    """ Performs test computations at the current precision intended to capture the
        workload of the exponential mechanism and catch if the current precision is
        sufficient. """
    # 1. Compute base = 2^(-eta)
    self.base = mpfr(pow(self.eta_x,self.eta_z))
    self.base *= mpfr(gmpy2.exp2(-gmpy2.mul(self.eta_y,self.eta_z)))

    # 2. Compute (base)^u_min and (base)^u_max
    min_weight = pow(self.base, self.u_min)
    max_weight = pow(self.base, self.u_max)
    mm = max_weight + min_weight

    # 3. Compute maximum total utility and minimum total utility
    max_total = max_weight*self.max_outcomes
    min_total = min_weight*self.max_outcomes

    # 4. Add max and min total utilities
    max_min_total = max_total + min_weight
    min_max_total = min_total + max_weight

  # Initialize a new mechanism
  def __init__(self, rng, eta_x = 1, eta_y = 0, eta_z = 1, \
                     u_min = 10, u_max = 0, max_O=100, \
                     min_sampling_precision = 10):
    """ Initializes a new ExpMech object including computing the required precision
        and setting inexact arithmetic exceptions.

        **Args**:
          rng (function): a random bit generator;
          eta_x (int): privacy parameter;
          eta_y (int): privacy parameter;
          eta_z (int): privacy parameter;
          u_min (int): the minimum utility (maximum magnitude);
          u_max (int): the maximum utility (maximum magnitude);
          max_O (int): the maximum size of the outcome space;
          min_sampling_precision (int): the minimum precision at which to sample for randomized rounding
    """
    # initialize precision_set to False
    self.precision_set = False
    """ Status indicator for whether the precision necessary has been computed yet. """

    # Set the gmpy2 library context to trap on inexact arithmetic, overflows, underflows, etc.
    ctx = gmpy2.get_context()
    ctx.trap_inexact = True
    ctx.trap_overflow = True
    ctx.trap_divzero = True
    ctx.trap_invalid = True
    ctx.trap_underflow = True
    ctx.trap_expbound = True
    ctx.trap_erange = True

    # Set the rng, privacy parameters and utility bounds
    self.rng = rng
    """ Random bit generator used for generating all randomness in the procedures. """
    try: # try to cast the integer-valued arguments to mpz - raise an error if non-integer
      self.eta_x = mpz(eta_x)
      self.eta_y = mpz(eta_y)
      self.eta_z = mpz(eta_z)
      self.u_min = mpz(u_min)
      self.u_max = mpz(u_max)
      self.max_outcomes = mpz(max_O)
    except gmpy2.InexactResultError:
      raise RuntimeError('Non-integer parameter when integer was expected')

    # Compute the required precision for the desired parameters
    # start from a small-ish precision so we don't waste unnecessary bits
    ctx = gmpy2.get_context()
    ctx.precision = 16
    while self.precision_set != True:
      if gmpy2.get_context().precision >= gmpy2.get_max_precision():
        raise RuntimeError('Failed to set precision: maximum precision exceeded.')
      else:
        try:
          self.check_precision()
        except gmpy2.InexactResultError:
          ctx = gmpy2.get_context()
          ctx.precision = 2*ctx.precision
        else:
          self.precision_set = True

    if self.precision_set != True:
      raise RuntimeError('Failed to set precision.')
    else:
      ctx = gmpy2.get_context()
      ctx.clear_flags() # clear any flags
      self.context = ctx.copy() # store a copy to test future state
      """ The gmpy2 context required for computations on this mechanism. """
      assert self.check_context(), 'Context invalid.'

    # Set the minimum sampling precision, which cannot be greater than the context precision
    self.min_sampling_precision = min(min_sampling_precision, self.context.precision)
    """ The minimum precision at which to perform sampling for randomized rounding. """

  # Sample a random value with p bits of precision
  # from a given starting power of 2. Output is in [0,2^{start_pow+1})
  def get_random_value(self,start_pow, p=None):
    """ Sample a random value based on self.rng of p bits between [0,2^(start_pow+1)). """
    if p == None:
      p = self.context.precision
    s = mpfr(0)
    nextbit = gmpy2.exp2(start_pow)
    for i in range(1, p):
      s = gmpy2.add(s, gmpy2.mul(nextbit,mpfr(self.rng())))
      nextbit = gmpy2.exp2(start_pow - i)
    return s

  # Randomized rounding logic
  def randomized_round(self, x):
    """ Round the input to an integer value. Value is rounded up with probability (x - floor(x)).
        Rounding randomness is sampled at min_sampling_precision."""
    s = self.get_random_value(-1, self.min_sampling_precision)
    output = int(x)+1
    cutoff = x - int(x)
    if s > cutoff:
      output = int(x)
    output = min(max(self.u_max, output),self.u_min)
    return output

  # Set the utility function, wrapped in randomized rounding logic
  # INPUTS: util, a utility function taking a single element as argument
  def set_utility(self, util):
    """ Set the utility function, wrapped in randomized rounding logic.

        Args:
          u (function): a function taking a single argument from the outcome space returning a single real value.
     """
    self.u = lambda x: self.randomized_round(util(x))

  # Sample an index from W according to the normalized weight of each entry using
  # randomness from rng
  def normalized_sample(self, W):
    """ Normalized sampling without division.

        Args:
          W: a set of weights from which to sample.

        Returns: an integer in [0,len(W)] corresponding to the index sampled.
    """
    t = gmpy2.fsum(W)  # compute total weight
    C = [gmpy2.fsum(W[0:i+1]) for i in range(0, len(W))]  # compute cumulative weights

    # Determine the maximum power of two for sampling
    i_max = 0
    while gmpy2.exp2(i_max) <= t:
      i_max += 1
    # sample a random number
    s = gmpy2.exp2(i_max + 1)
    while s > t:
      s  = self.get_random_value(i_max,self.context.precision)
    # return the element that matches the sampled index
    for i in range(0, len(W)):
      if C[i] >= s:
       return i

  # Exact exponential mechanism
  def exact_exp_mech(self, O):
    """ Run the mechanism over the outcome space O. Returns a single element from O sampled from the exponential mechanism. """
    # check that O matches size requirements
    if len(O) > self.max_outcomes:
      raise RuntimeError('Outcome space size too large.')

    # Get utilities
    U = [self.u(o) for o in O]
    self.check_context()

    # Compute weights
    W = [pow(mpfr(self.base),mpfr(u)) for u in U]
    self.check_context()

    # Sample
    return O[self.normalized_sample(W)]


class LaplaceMech(ExpMech):
  """ The LaplaceMech child class. Implements the clamped Laplace mechanism utility function over a discrete set of outcomes.
      Samples from the outcome space [b_min,b_max] at granularity gamma. """
  def __init__(self, rng, x, sensitivity, eta_x = 1, eta_y = 0, eta_z = 1, \
                     b_min = -10, b_max = 10, gamma = 2**(-4), \
                     min_sampling_precision = 10):
    """ Initializes the LaplaceMech including computing the required precision.

        Args:
        rng (function): a random bit generator;
        x (float or int): the target value of the mechanism
        sensitivity: the sensitivity associated with the computation of x
        eta_x (int): privacy parameter;
        eta_y (int): privacy parameter;
        eta_z (int): privacy parameter;
        b_min (float or int): the lower bound of the output range;
        b_max (float or int): the upper bound of the output range;
        gamma (float): the discretization granularity
        min_sampling_precision (int): the minimum precision at which to sample for randomized rounding

      """
    # compute outcome space
    O = []
    b = b_min
    while b <= b_max:
      O.append(b)
      b += gamma
    self.Outcomes = O
    """ The outcome space used by the mechanism [b_min,b_max] at granularity gamma. """

    # clamp x to range
    x = min(max(b_min,x),b_max)

    # specify utility function
    if sensitivity <= 0:
      raise RuntimeError('Sensitivity must be greater than 0.')
    u = lambda y: abs(x - y)/sensitivity

    max_u = 0
    min_u = int((b_max - b_min)/sensitivity) + 1

    # initialize the base class
    max_O = len(self.Outcomes)+1
    ExpMech.__init__(self,rng,eta_x,eta_y,eta_z,min_u,max_u,max_O,min_sampling_precision)
    self.set_utility(u)

  def run_mechanism(self):
    """ Runs the mechanism and returns a single outcome from O. """
    O = self.Outcomes
    return self.exact_exp_mech(O)
