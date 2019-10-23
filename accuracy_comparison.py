# Accuracy bound comparisons
import math
import matplotlib.pyplot as plt
import numpy as np
from naive import *
from expmech import *

# Get the lower bound probabilities for each element
# given the set of utilities U and privacy parameter eps
def get_lower_bound_probs(U,eps):
  probs = []
  for i in range(0, len(U)):
    e_sum = 0
    for j in range(0,len(U)):
      if i != j:
        u = U[j]
        p = abs(math.ceil(u)-u)
        e_sum += p*np.exp(-eps*math.floor(u))
        e_sum += (1-p)*np.exp(-eps*np.ceil(u))
    u = U[i]
    p = abs(math.ceil(u)-u)
    e_prob = p*np.exp(-eps*math.floor(u))/(e_sum + np.exp(-eps*math.floor(u))) + (1-p)*np.exp(-eps*math.ceil(u))/(e_sum + np.exp(-eps*math.ceil(u)))
    probs.append(e_prob)
  return probs

# Return the probabilities assigned by the naive exponential mechanism given
# a set of utilities U and privacy parameter eps
def get_naive_probs(U,eps):
  W = [np.exp(-eps*u) for u in U]
  t = sum(W)
  probs = [w/t for w in W]
  return probs

# Compute the maximum point-wise error between the naive probability and the
# upper and lower bounds on the probability assigned by the randomized rounding
# mechanism.
def get_max_pointwise_errors(naive_probs, lower_probs, upper_probs):
  errs = []
  for i in range(0, len(naive_probs)):
    errbound = max(abs(lower_probs[i] - naive_probs[i]), abs(upper_probs[i] - naive_probs[i]))
    errs.append(errbound)
  return errs

# Compare the probabilities assigned to a set of outcomes by the un-rounded naive
# exponential mechanism versus the base-2 exponential mechanism with randomized
# rounding.
def compare_rounded_probabilities():
  min_sampling_precision = 10
  eta_x = 1
  eta_y = 1
  eta_z = 1
  t = 0
  b_min = -10
  b_max = 10
  gammas = [2**(-2), 2**(-3), 2**(-4),2**(-5), 2**(-6), 2**(-7), 2**(-8)]
  util = lambda x: abs(t-x)
  s = 1 # set the sensitivity to 1
  As = []
  rng = lambda : np.random.randint(0,2) # np.random is a placeholder for demonstration purposes only

  outcomesets = []
  errsets = []
  relerrsets = []

  for g in gammas:
    eps = 2*np.log(2)*(-np.log2((eta_x/(2**eta_y))))
    l = LaplaceMech(rng, t, s, eta_x, eta_y, eta_z, b_min,b_max,g, min_sampling_precision)

    O = l.Outcomes
    U = [util(o) for o in O]

    # Get the upper and lower bounds on the probabilities for the rounded mechanism
    lower_bound_probs = get_lower_bound_probs(U,eps)
    A = 1 - sum(lower_bound_probs)
    As.append(A)
    upper_bound_probs = [p + A for p in lower_bound_probs]

    # Get the probabilities of the unrounded mechanism
    unrounded_probs = get_naive_probs(U, eps)

    # Get the maximum error
    max_pointwise_errors = get_max_pointwise_errors(unrounded_probs, lower_bound_probs, upper_bound_probs)
    # Get the relative maximum error
    rel_max_pointwise_errors = [max_pointwise_errors[i]/unrounded_probs[i] for i in range(0, len(unrounded_probs))]

    # Store the outcome set and error set to display later
    outcomesets.append(O)
    errsets.append(max_pointwise_errors)
    relerrsets.append(rel_max_pointwise_errors)


    # Plot
    plt.plot(O, unrounded_probs,linestyle='--',marker='.',label='naive (unrounded)')
    plt.plot(O, lower_bound_probs,linestyle='--',marker='.',label='rr lower bound')
    plt.plot(O, upper_bound_probs,linestyle='--',marker='.',label='rr upper bound')
    plt.xlim(-5,5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Randomized rounding bounds vs unrounded probability")
    plt.show()



  plt.plot(gammas, As, linestyle='--',marker='o', label='bound width')
  plt.legend()
  plt.title("Bound Widths")
  plt.show()

  for i in range(0, len(gammas)):
    g = gammas[i]
    O = outcomesets[i]
    errs = errsets[i]
    plt.plot(O, errs,linestyle='--',marker='.',label='$\gamma=$'+str(g))
  plt.title("Point-wise error upper-bound")
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.show()


  for i in range(0, len(gammas)):
    g = gammas[i]
    O = outcomesets[i]
    errs = relerrsets[i]
    plt.plot(O, errs,linestyle='--',marker='.',label='$\gamma=$'+str(g))
  plt.title("Point-wise relative error upper-bound")
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.show()

if __name__ == '__main__':
    compare_rounded_probabilities()
