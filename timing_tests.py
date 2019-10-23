# Timing tests comparing naive implementation with base-2 implementation
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from naive import *
from expmech import *

# Run n discrete outcome space timing tests with max_O utility and k elements.
# Returns: (b2 mechanism setup time, b2 avg run time, naive avg run time) in seconds and precision
def utility_range_timing_tests(n=100, k = 10, max_O=100, eta_x=1, eta_y=1,min_sampling_precision=1):
  """ Tests with a fixed outcome set size with varying utility range.

      ** Args: **

      n : number of trials per parameter;
      k : outcome space size;
      eta_x (int): privacy parameter;
      eta_y (int): privacy parameter;
      min_sampling_precision: minimum sampling precision;

      **Returns: ** the base-2 mechanism setup time, b2 average run time, naive average run time in seconds, and the precision used
      """
  eta_z = 1

  # setup outcome space, utility function and other parameters
  O = [i for i in range(0, k)]
  t = np.random.randint(0,k) # select a random "target" value
  util = lambda x: abs(t - x)
  u_min = max_O
  u_max = 0
  rng = lambda : np.random.randint(0,2) # np.random is a placeholder for demonstration purposes only
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))

  # mechanism setup
  start_setup_time = timer()
  e = ExpMech(rng,eta_x,eta_y,eta_z,u_min,u_max,max_O,min_sampling_precision)
  e.set_utility(util)
  prec = e.context.precision
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time

  # base 2 test loop
  start_b2 = timer()
  for i in range(0, n):
    e.exact_exp_mech(O)
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/n

  # naive test loop
  start_naive = timer()
  for i in range(0,n):
    naive_exp_mech(eps,util,O)
  end_naive = timer()
  naive_avg = (end_naive - start_naive)/n

  return setup_time, b2_avg, naive_avg, prec

# Run n discrete outcome space timing tests with max_O elements.
# Returns: (b2 mechanism setup time, b2 avg run time, naive avg run time) in seconds and precision
def discrete_timing_tests(n=100, max_O=100, eta_x=1, eta_y=1,min_sampling_precision=1):
  """ Tests with a discrete outcome space of varying size and utility range.

          ** Args: **
          n : number of trials per parameter;
          max_O : outcome space size;
          eta_x (int): privacy parameter;
          eta_y (int): privacy parameter;
          min_sampling_precision: minimum sampling precision;

          **Returns: ** the base-2 mechanism setup time, b2 average run time, naive average run time in seconds, and the precision used
          """
  eta_z = 1

  # setup outcome space, utility function and other parameters
  O = [i for i in range(0, max_O)]
  t = np.random.randint(0,max_O) # select a random "target" value
  util = lambda x: abs(t - x)
  u_min = max_O
  u_max = 0
  rng = lambda : np.random.randint(0,2) # np.random is a placeholder for demonstration purposes only
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))

  # mechanism setup
  start_setup_time = timer()
  e = ExpMech(rng,eta_x,eta_y,eta_z,u_min,u_max,max_O,min_sampling_precision)
  e.set_utility(util)
  prec = e.context.precision
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time

  # base 2 test loop
  start_b2 = timer()
  for i in range(0, n):
    e.exact_exp_mech(O)
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/n

  # naive test loop
  start_naive = timer()
  for i in range(0,n):
    naive_exp_mech(eps,util,O)
  end_naive = timer()
  naive_avg = (end_naive - start_naive)/n

  return setup_time, b2_avg, naive_avg, prec


def laplace_timing_tests(n=100, b_min=-10,b_max=10, gamma=2**(-4), eta_x=1, eta_y=1,min_sampling_precision=10):
    """ Tests the LaplaceMech.

        ** Args: **
        n : number of trials per parameter;
        b_min and b_max : range parameters;
        gamma: granularity;
        eta_x (int): privacy parameter;
        eta_y (int): privacy parameter;
        min_sampling_precision: minimum sampling precision;

        **Returns: ** the base-2 mechanism setup time, b2 average run time, naive average run time in seconds
        """
  eta_z = 1

  # setup outcome space, utility function and other parameters
  t = np.random.randint(b_min,b_max) # select a random "target" value in the range
  util = lambda x: abs(t-x)
  s = 1 # set the sensitivity to 1
  rng = lambda : np.random.randint(0,2) # np.random is a placeholder for demonstration purposes only
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))

  # mechanism setup
  start_setup_time = timer()
  l = LaplaceMech(rng, t, s, eta_x, eta_y, eta_z, b_min,b_max,gamma, min_sampling_precision)
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time


  # base 2 test loop
  start_b2 = timer()
  for j in range(0, n):
    l.run_mechanism()
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/n

  # naive test loop
  O = l.Outcomes
  start_naive = timer()
  for j in range(0,n):
    naive_exp_mech(eps,util,O)
  end_naive = timer()
  naive_avg = (end_naive - start_naive)/n

  return setup_time, b2_avg, naive_avg

def run_timing_tests():
  """ Runs a series of timing tests for the base-2 ExpMech class versus the naive base-e implementation. """
  # Discrete tests
  n = 10
  min_sampling_precision = 1
  eta_x = 1
  eta_y = 1
  sizes = [100,200,300,400,500,600,700,800,900,1000,1250,1500,2000,2500]
  discrete_results = []
  for s in sizes:
    discrete_results.append(discrete_timing_tests(n,s,eta_x,eta_y,min_sampling_precision))
  plt.plot(sizes,[discrete_results[i][1] for i in range(0, len(discrete_results))],marker='o',label='b2 average time')
  plt.plot(sizes,[discrete_results[i][2] for i in range(0, len(discrete_results))],marker='o',label='naive average time')
  plt.legend()
  plt.title("Discrete Outcomes Tests")
  plt.xlabel("Outcome Space Size")
  plt.show()

  # Utility range tests
  extended_sizes = [100,1000,2000,3000,4000,5000]
  k = 100 # test on 100 elements
  for s in extended_sizes:
    extended_discrete_results.append(utility_range_timing_tests(n,s,k,eta_x,eta_y,min_sampling_precision))
  plt.plot(extended_sizes,[extended_discrete_results[i][1] for i in range(0, len(extended_discrete_results))],marker='o',label='b2 average time')
  #plt.plot(extended_sizes,[extended_discrete_results[i][2] for i in range(0, len(extended_discrete_results))],marker='o',label='naive average time')
  plt.legend()
  plt.title("Utility Range Tests")
  plt.xlabel("utility range size")
  plt.show()


  # Laplace tests
  min_sampling_precision = 10
  eta_x = 1
  eta_y = 1
  gammas = [2**(-2), 2**(-3), 2**(-4),2**(-5), 2**(-6), 2**(-7), 2**(-8)]
  laplace_results = []
  for g in gammas:
    laplace_results.append(laplace_timing_tests(n,gamma=g))

  plt.plot(gammas,[laplace_results[i][1] for i in range(0, len(laplace_results))],marker='o',label='b2 average time')
  plt.plot(gammas,[laplace_results[i][2] for i in range(0, len(laplace_results))],marker='o',label='naive average time')
  plt.xlabel(r"Granularity $\gamma$")
  plt.legend()
  plt.title("Laplace Tests")
  plt.show()


if __name__ == '__main__':
    run_timing_tests()
