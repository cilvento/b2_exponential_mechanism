# The Naive Exponential Mechanism Reference Implementation
# author: C. Ilvento, cilvento@gmail.com
# status: reference implementation only - unsafe

import numpy as np
# The naive exponential mechanism using numpy
# Inputs:
#  eps: the privacy parameter
#  u: the utility function taking an element in O as an argument
#  O: the set of outputs
# Returns: the outcome of the exponential mechanism
def naive_exp_mech(eps, u, O):
    weights = [np.exp(-(eps/2.0)*u(o)) for o in O] # compute the weight of each element in the outcome space
    total_weight = sum(weights)
    c_weights = [sum(weights[0:i]/total_weight) for i in range(1, len(O)+1)] # compute the cumulative weight of each element
    index = np.random.rand() # sample a random value in [0,1].
    for i in range(0, len(O)):
        if c_weights[i] >= index:
            return O[i] # return the element corresponding to the random index

# A naive implementation of the Laplace mechanism
# Inputs:
#  f: the function f
#  eps: privacy parameter
#  delta_f: sensitivity of f
def naive_laplace(f, delta_f, eps):
    return f() + np.random.laplace(0,eps/delta_f)
