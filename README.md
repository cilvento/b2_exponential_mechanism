# Implementing the Exponential Mechanism with Base-2 DP
Author: Christina Ilvento

[*Paper*](https://arxiv.org/abs/1912.04222)
## *Code and supplemental materials.*

**Abstract**

Despite excellent theoretical support, Differential Privacy (DP) can still be a challenge to implement in practice. In part, this challenge is due to the very real concerns associated with converting arbitrary or infinite-precision theoretical mechanisms to the often messy realities of floating point or fixed-precision. Beginning with the troubling result of Mironov demonstrating the security issues of using floating point for implementing the Laplace mechanism, there have been many reasonable concerns raised on the vulnerabilities of real-world implementations of DP.

In this work, we examine the practicalities of implementing the exponential mechanism of McSherry and Talwar. We demonstrate that naive or malicious implementations can result in catastrophic privacy failures. To address these problems, we show that the mechanism can be implemented *exactly* for a rich set of values of the privacy parameter epsilon and utility functions with limited practical overhead in running time and minimal code complexity.

How do we achieve this result? We employ a simple trick of switching from base *e* to base 2, allowing us to perform precise base 2 arithmetic. A short, precise expression is always available for epsilon, and the only approximation error we incur is the conversion of the base-2 privacy parameter back to base *e* for reporting purposes. The core base *e* arithmetic of the mechanism can be simply and efficiently implemented using open-source high precision floating point libraries. Furthermore, the exact nature of the implementation lends itself to simple monitoring of correctness and proofs of privacy.


## Supplemental Materials
* Demo and project overview: Jupyter notebook [`python/demo.ipynb`](./python/demo.ipynb)  ([rendered as markdown](./demo/demo.md))
* Figures: `figures/`

## Code
### Python
* Naive implementation of the exponential mechanism: see `python/naive.py`
* Base-2 implementation of the exponential mechanism: see `python/expmech.py`
* Timing tests: `python/timing_tests.py`
* Accuracy bounds (randomized rounding): `python/accuracy_comparison.py`