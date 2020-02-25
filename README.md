# RobustFactorizations

[![Build Status](https://travis-ci.org/baggepinnen/RobustFactorizations.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/RobustFactorizations.jl)
[![Codecov](https://codecov.io/gh/baggepinnen/RobustFactorizations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/RobustFactorizations.jl)

This package provides some utilities for robust factorization of matrices, useful for, e.g., matrix completion and denoising.

## Examples

### Only sparse noise
```julia
L = lowrank(100,10,3)
E = 10sparserandn(100,10)
Ln = L + E
res = rpca(Ln, verbose=false)
@show opnorm(L - res.L)/opnorm(L)
```
### Dense and sparse noise
```julia
L = lowrank(100,10,3)
D = randn(100,10)
S = 10sparserandn(100,10)
Ln = L + D + S
λ = 1/sqrt(maximum(size(L)))
res1 = rpca(Ln, verbose=false)
res2 = rpca(Ln, verbose=false, proxD=SqrNormL2(λ/std(D))) # proxD parameter might need tuing
@show opnorm(L - res1.L)/opnorm(L), opnorm(L - res2.L)/opnorm(L)
```

## Functions
- `rpca` Works very well, uses "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices", Zhouchen Lin, Minming Chen, Leqin Wu, Yi Ma, https://people.eecs.berkeley.edu/~yima/psfile/Lin09-MP.pdf
- `rpca_fista` requires tuning.
- `rpca_admm` requires tuning.
