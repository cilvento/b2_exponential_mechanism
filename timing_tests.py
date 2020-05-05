# Timing tests comparing naive implementation with base-2 implementation
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from naive import *
from expmech import *


def utility_range_benchmark(n, optimize):
  num_trials = 10
  start_setup_time = timer()    
  k = 1000 # outcome space size
  O = [i+n for i in range(1,k)]
  O.insert(0,0)
  eta_x = 1
  eta_y = 1
  eta_z = 1
  rng = lambda : np.random.randint(0,2) 
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))
  u_min = 0
  u_max = n + k
  max_O = k
  util = lambda x: x


  e = ExpMech(rng,eta_x,eta_y,eta_z,u_max,u_min,max_O,min_sampling_precision=8,empirical_precision=False)
  e.set_utility(util)
  prec = e.context.precision
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time

  # base 2 test loop
  start_b2 = timer()
  for i in range(0, num_trials):
    e.exact_exp_mech(O,optimized_sample=optimize)
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/num_trials

  # naive test loop
  start_naive = timer()
  for i in range(0,num_trials):
    naive_exp_mech(eps,util,O)
  end_naive = timer()
  naive_avg = (end_naive - start_naive)/num_trials
  return setup_time, b2_avg, b2_avg + setup_time, naive_avg


def outcomespace_benchmark(n, optimize):
  num_trials = 10
  start_setup_time = timer()    
  O = [i for i in range(0,n)]
  eta_x = 1
  eta_y = 1
  eta_z = 1
  rng = lambda : np.random.randint(0,2) 
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))
  u_min = 0
  u_max = n 
  max_O = n
  util = lambda x: x


  e = ExpMech(rng,eta_x,eta_y,eta_z,u_max,u_min,max_O,min_sampling_precision=8,empirical_precision=False)
  e.set_utility(util)
  prec = e.context.precision
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time

  # base 2 test loop
  start_b2 = timer()
  for i in range(0, num_trials):
    e.exact_exp_mech(O,optimized_sample=optimize)
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/num_trials

  # naive test loop
  start_naive = timer()
  for i in range(0,num_trials):
    naive_exp_mech(eps,util,O)
  end_naive = timer()
  naive_avg = (end_naive - start_naive)/num_trials
  return setup_time, b2_avg, b2_avg + setup_time, naive_avg

def precision_benchmark(n, precision):
  num_trials = 10
  optimize = True
  start_setup_time = timer()    
  O = [i for i in range(0,n)]
  eta_x = 1
  eta_y = 1
  eta_z = 1
  rng = lambda : np.random.randint(0,2) 
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))
  u_min = 0
  u_max = n 
  max_O = n
  util = lambda x: x


  e = ExpMech(rng,eta_x,eta_y,eta_z,u_max,u_min,max_O,min_sampling_precision=8,empirical_precision=precision)
  e.set_utility(util)
  prec = e.context.precision
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time

  # base 2 test loop
  start_b2 = timer()
  for i in range(0, num_trials):
    e.exact_exp_mech(O,optimized_sample=optimize)
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/num_trials


  return setup_time, b2_avg, b2_avg + setup_time


def laplace_benchmark(gamma, optimize):
  num_trials = 10
  start_setup_time = timer()    
  eta_x = 1
  eta_y = 1
  eta_z = 1
  rng = lambda : np.random.randint(0,2) 
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))

  l = LaplaceMech(rng, 0.0, 1.0, eta_x,eta_y,eta_z,-10.0,10.0,gamma, min_sampling_precision=8,empirical_precision=False)
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time

  # base 2 test loop
  start_b2 = timer()
  for i in range(0, num_trials):
    l.run_mechanism(optimize)
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/num_trials

  # naive test loop
  start_naive = timer()
  O = []
  util = lambda x: abs(x)
  while i*gamma - 10.0 <= 10.0:
      O.append(i*gamma - 10)
      i+= 1
  for i in range(0,num_trials):
    naive_exp_mech(eps,util,O)
  end_naive = timer()
  naive_avg = (end_naive - start_naive)/num_trials
  return setup_time, b2_avg, b2_avg + setup_time, naive_avg

def laplace_precision_benchmark(gamma, precision):
  num_trials = 10
  start_setup_time = timer()  
  optimize = False  
  eta_x = 1
  eta_y = 1
  eta_z = 1
  rng = lambda : np.random.randint(0,2) 
  eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))

  l = LaplaceMech(rng, 0.0, 1.0, eta_x,eta_y,eta_z,-10.0,10.0,gamma, min_sampling_precision=8,empirical_precision=precision)
  end_setup_time = timer()
  setup_time = end_setup_time - start_setup_time

  # base 2 test loop
  start_b2 = timer()
  for i in range(0, num_trials):
    l.run_mechanism(optimize)
  end_b2 = timer()
  b2_avg = (end_b2 - start_b2)/num_trials

  # naive test loop
  start_naive = timer()
  O = []
  util = lambda x: abs(x)
  while i*gamma - 10.0 <= 10.0:
      O.append(i*gamma - 10)
      i+= 1
  for i in range(0,num_trials):
    naive_exp_mech(eps,util,O)
  end_naive = timer()
  naive_avg = (end_naive - start_naive)/num_trials
  return setup_time, b2_avg, b2_avg + setup_time, naive_avg


def run_timing_tests():
  """ Runs a series of timing tests for the base-2 ExpMech class versus the naive base-e implementation. """
  
  
  # Laplace tests
  sizes = [1.0,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
  results = []
  opt_results = []
  for s in sizes:
    results.append(laplace_benchmark(s, False))
    opt_results.append(laplace_benchmark(s, True))
    
  plt.plot(sizes,[results[i][2] for i in range(0, len(results))],marker='o',linestyle=':',label='b2 average time')
  plt.plot(sizes,[opt_results[i][2] for i in range(0, len(opt_results))],marker='o',linestyle=':',label='b2-opt average time')
  plt.plot(sizes,[results[i][3] for i in range(0, len(results))],marker='o',linestyle=':',label='naive average time')
  plt.legend()
  plt.title("Laplace Tests")
  plt.xlabel("gamma")
  plt.savefig('laplace.png')
  plt.close()
  # Output results
  print("Laplace")
  print("Gamma,", ','.join(map(str,sizes)))
  print("Naive,", ','.join(map(str,[results[i][3] for i in range(0, len(results))])))
  print("Not Optimized,", ','.join(map(str,[results[i][2] for i in range(0, len(results))])))
  print("Optimized,", ','.join(map(str,[opt_results[i][2] for i in range(0, len(opt_results))])))


# Laplace Precision tests
  sizes = [1.0,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
  results = []
  opt_results = []
  for s in sizes:
    results.append(laplace_precision_benchmark(s, False))    # Theoretical
    opt_results.append(laplace_precision_benchmark(s, True)) # Empirical
    
  plt.plot(sizes,[results[i][2] for i in range(0, len(results))],marker='o',linestyle=':',label='theoretical')
  plt.plot(sizes,[opt_results[i][2] for i in range(0, len(opt_results))],marker='o',linestyle=':',label='empirical')
  plt.plot(sizes,[results[i][3] for i in range(0, len(results))],marker='o',linestyle=':',label='naive average time')
  plt.legend()
  plt.title("Laplace Tests")
  plt.xlabel("gamma")
  plt.savefig('laplace_precision.png')
  plt.close()
  # Output results
  print("Laplace Precision Type Tests")
  print("Gamma,", ','.join(map(str,sizes)))
  #print("Naive,", ','.join(map(str,[results[i][3] for i in range(0, len(results))])))
  print("Theoretical,", ','.join(map(str,[results[i][2] for i in range(0, len(results))])))
  print("Empirical,", ','.join(map(str,[opt_results[i][2] for i in range(0, len(opt_results))])))


  # Precision type tests
  sizes = [100, 1000, 2000, 3000, 4000, 5000]
  results = []
  opt_results = []
  for s in sizes:
    results.append(precision_benchmark(s, False))
    opt_results.append(precision_benchmark(s, True))
    
  plt.plot(sizes,[results[i][2] for i in range(0, len(results))],marker='o',linestyle=':',label='theoretical average time')
  plt.plot(sizes,[opt_results[i][2] for i in range(0, len(opt_results))],marker='o',linestyle=':',label='empirical average time')
  plt.legend()
  plt.title("Precision Type Tests")
  plt.xlabel("Size")
  plt.savefig('precision_type.png')
  plt.close()
  print("Precision Type")
  print("Sizes,", ','.join(map(str,sizes)))
  print("Theoretical,", ','.join(map(str,[results[i][2] for i in range(0, len(results))])))
  print("Empirical,", ','.join(map(str,[opt_results[i][2] for i in range(0, len(opt_results))])))


  # Outcomespace Size tests
  sizes = [100, 1000, 2000, 3000, 4000, 5000]
  results = []
  opt_results = []
  for s in sizes:
    results.append(outcomespace_benchmark(s, False))
    opt_results.append(outcomespace_benchmark(s, True))
    
  plt.plot(sizes,[results[i][2] for i in range(0, len(results))],marker='o',linestyle=':',label='b2 average time')
  plt.plot(sizes,[opt_results[i][2] for i in range(0, len(opt_results))],marker='o',linestyle=':',label='b2-opt average time')
  plt.plot(sizes,[results[i][3] for i in range(0, len(results))],marker='o',linestyle=':',label='naive average time')
  plt.legend()
  plt.title("Outcomesize Tests")
  plt.xlabel("Size")
  plt.savefig('outcome_size.png')
  plt.close()
  # Output results
  print("Outcomesizes")
  print("Sizes,", ','.join(map(str,sizes)))
  print("Naive,", ','.join(map(str,[results[i][3] for i in range(0, len(results))])))
  print("Not Optimized,", ','.join(map(str,[results[i][2] for i in range(0, len(results))])))
  print("Optimized,", ','.join(map(str,[opt_results[i][2] for i in range(0, len(opt_results))])))

  # Utility range tests
  sizes = [100,1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
  results = []
  opt_results = []
  for s in sizes:
    results.append(utility_range_benchmark(s, False))
    opt_results.append(utility_range_benchmark(s, True))
    
  plt.plot(sizes,[results[i][2] for i in range(0, len(results))],marker='o',linestyle=':',label='b2 average time')
  plt.plot(sizes,[opt_results[i][2] for i in range(0, len(opt_results))],marker='o',linestyle=':',label='b2-opt average time')
  plt.plot(sizes,[results[i][3] for i in range(0, len(results))],marker='o',linestyle=':',label='naive average time')
  plt.legend()
  plt.title("Utility Range Tests")
  plt.xlabel("Range")
  plt.savefig('utility_range.png')
  plt.close()
  # Output results
  print("Utility Range")
  print("Range,", ','.join(map(str,sizes)))
  print("Naive,", ','.join(map(str,[results[i][3] for i in range(0, len(results))])))
  print("Not Optimized,", ','.join(map(str,[results[i][2] for i in range(0, len(results))])))
  print("Optimized,", ','.join(map(str,[opt_results[i][2] for i in range(0, len(opt_results))])))


if __name__ == '__main__':
    run_timing_tests()