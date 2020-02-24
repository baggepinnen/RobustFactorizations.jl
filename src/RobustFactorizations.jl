module RobustFactorizations

using Statistics, LinearAlgebra
using ProximalOperators, TSVD

export rpca, lowrank, sparserandn


function lowrank(r,c,n)
    A = randn(r,c)
    U,S,V = svd(A)
    U[:,1:n]*Diagonal(S[1:n])*V[:,1:n]'
end

function sparserandn(r,c,γ=0.01)
    randn(r,c) .* (rand(r,c) .< γ)
end


@inline soft_th(x, ϵ) = max(x-ϵ,zero(x)) + min(x+ϵ,zero(x))
@inline function soft_th(x::Complex, ϵ)
    m,a = abs(x), angle(x)
    m = max(m-ϵ,zero(m)) + min(m+ϵ,zero(m))
    m*cis(a)
end

struct RPCA{T<:AbstractMatrix}
    L::T
    S::T
    D::T
end



"""
    RPCA = rpca(A::Matrix; λ=1.0 / √(maximum(size(A))), iters=1000, tol=1.0e-7, ρ=1.5, verbose=false, nonnegL=false, nonnegS=false, nukeA=true)

minimize_{L,D,S} ||L||_* + λ||S||₁ + γ||D||₂² s.t. A = L+D+S

Ref: "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices", Zhouchen Lin, Minming Chen, Leqin Wu, Yi Ma, https://people.eecs.berkeley.edu/~yima/psfile/Lin09-MP.pdf

#Arguments:
- `A`: Input matrix
- `λ`: Sparsity regularization
- `iters`: Maximum number of iterations
- `tol`: Tolerance
- `ρ`: Algorithm tuning param
- `verbose`: Print status
- `nonnegL`: Hard thresholding on A
- `nonnegS`: Hard thresholding on E
- `proxL`: NuclearNorm(1/2)
- `proxD`: nothing
- `proxS`: NormL1(λ))

To speed up convergence you may either increase the tolerance or increase `ρ`. Increasing `tol` is often the best solution.
"""
function rpca(A::AbstractMatrix{T};
                          λ              = real(T)(1.0/sqrt(maximum(size(A)))),
                          iters::Int     = 1000,
                          tol            = sqrt(eps(real(T))),
                          ρ              = real(T)(1.5),
                          verbose::Bool  = false,
                          nonnegL::Bool  = false,
                          nonnegS::Bool  = false,
                          proxL          = NuclearNorm(real(T)(nonnegL ? 1 : 1/2)),
                          proxD          = nothing,
                          proxS          = NormL1(λ)) where T
    RT        = real(T)
    M, N      = size(A)
    d         = min(M,N)
    L, S      = zeros(T, M, N), zeros(T, M, N)
    D         = zeros(T, M, N)
    Z         = similar(A)
    Y         = copy(A)
    norm²     = opnorm(Y)::RT # can be tuned
    norm∞     = norm(Y, Inf) / λ
    dual_norm = max(norm², norm∞)
    d_norm    = norm²
    Y       ./= dual_norm
    μ         = RT(1.25) / norm²
    μ̄         = μ  * RT(1.0e+7)
    sv        = 10
    for k = 1:iters
        if proxD !== nothing
            Z .= A .- L .- S .+ (1/μ) .* Y
            prox!(D, proxD, Z, 1/μ)
        end
        Z .= A .- L .- D .+ (1/μ) .* Y
        prox!(S, proxS, Z, 1/μ)
        if nonnegS
            S .= max.(S, 0)
        end
        Z .= A .- S .- D .+ (1/μ) .* Y
        prox!(L, proxL, Z, 1/μ)
        if nonnegL
            L .= max.(L, 0)
        end

        @. Z = A - L - S - D # Z are the reconstruction errors
        @. Y = Y + μ * Z # Y can not be moved below as it depends on μ which is changed below
        μ = min(μ*ρ, μ̄)

        cost = opnorm(Z) / d_norm
        verbose && println("$(k) cost: $(round(cost, sigdigits=4))")

        if cost < tol
            verbose && println("converged")
            break
        end
        k == iters && @warn "Maximum number of iterations reached, cost: $cost, tol: $tol"
    end

    RPCA(L, S, D)
end


function crossval(n, D, args...;kwargs...)

end

end # module
