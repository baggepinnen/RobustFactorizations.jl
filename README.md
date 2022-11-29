# RobustFactorizations

[![CI](https://github.com/baggepinnen/RobustFactorizations.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/RobustFactorizations.jl/actions)
[![Codecov](https://codecov.io/gh/baggepinnen/RobustFactorizations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/RobustFactorizations.jl)

This package provides some utilities for robust factorization of matrices, useful for, e.g., matrix completion and denoising.

We try to find the low-rank matrix $L$ when given matrix $L_n$ corrupted by sparse noise $S$ and dense noise $D$ according to $L_n = L + D + S$. Typically, $S$ contains very few entries, but they may be very large, while the entries in $D$ are much smaller, and maybe normally distributed.

## Examples

### Only sparse noise
```julia
L = lowrank(100,10,3)
S = 10sparserandn(100,10)
Ln = L + S
res = rpca(Ln, verbose=false)
@show opnorm(L - res.L)/opnorm(L)
```
### Dense and sparse noise
```julia
L = lowrank(100,10,3)      # A low-rank matrix
D = randn(100,10)          # A dense noise matrix
S = 10sparserandn(100,10)  # A sparse noise matrix (large noise)
Ln = L + D + S             # Ln is the sum of them all
λ = 1/sqrt(maximum(size(L)))
res1 = rpca(Ln, verbose=false)
res2 = rpca(Ln, verbose=false, proxD=SqrNormL2(λ/std(D))) # proxD parameter might need tuning
@show opnorm(L - res1.L)/opnorm(L), opnorm(L - res2.L)/opnorm(L)
```

## Functions
- `rpca` Works very well, uses "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices", Zhouchen Lin, Minming Chen, Leqin Wu, Yi Ma, https://people.eecs.berkeley.edu/~yima/psfile/Lin09-MP.pdf
- `rpca_fista` requires tuning.
- `rpca_admm` requires tuning.


The `rpca` function is the recommended default choice:
```julia
rpca(Ln::Matrix; λ=1.0 / √(maximum(size(A))), iters=1000, tol=1.0e-7, ρ=1.5, verbose=false, nonnegL=false, nonnegS=false, nukeA=true)
```
It solves the following problem:
$$\operatorname{minimize}_{L,D,S} ||L||_* + \lambda ||S||_1 + \gamma ||D||^2_2 \quad \text{s.t. } L_n = L+D+S$$

Reference:
> "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices", Zhouchen Lin, Minming Chen, Leqin Wu, Yi Ma, https://people.eecs.berkeley.edu/~yima/psfile/Lin09-MP.pdf

**Arguments:**
- `Ln`: Input data matrix
- `λ`: Sparsity regularization
- `iters`: Maximum number of iterations
- `tol`: Tolerance
- `ρ`: Algorithm tuning param
- `verbose`: Print status
- `nonnegL`: Hard thresholding on A
- `nonnegS`: Hard thresholding on E
- `proxL`: Defaults to `NuclearNorm(1/2)`
- `proxD`: Defaults to `nothing`
- `proxS`: Defaults to `NormL1(λ))`

To speed up convergence you may either increase the tolerance or increase `ρ`. Increasing `tol` is often the best solution.