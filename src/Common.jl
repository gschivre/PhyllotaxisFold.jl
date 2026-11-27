# Common
#
# Constants and small numerical tools shared by every block of the model: the two canonical divergence angles, log I₀ and
# the Bessel ratio (generic in the number type, so that they work with nested ForwardDiff duals), the half-t distribution,
# the -Inf-safe logaddexp and the twice-differentiable Optim objective with ForwardDiff derivatives.

"the golden divergence angle π (3 − √5) rad ≈ 137.5°, the canonical angle ψ of the main spiral"
const Golden_angle = π * (3 - sqrt(5))
"the Lucas divergence angle π (1 − 1/√5) rad ≈ 99.5°, the canonical angle ψ of the alternative spiral"
const Lucas_angle = π * (1 - inv(sqrt(5)))

"""
    fit_map(model; kwargs...)

MAP estimate of a model by Newton's method with trust region. Method: `fit_map(::PhyllotaxisModelAD)`, which returns a
`PhyllotaxisFit`.
"""
function fit_map end


"""
    logbesseli(n, x)

log Iₙ(x) (Plesner 2024): power series summed from its largest term for x ≤ 100, asymptotic expansion beyond. Generic in
the number type of x (works with ForwardDiff duals).
"""
function logbesseli(n, x::T; tol = eps(), Kmax = 20) where {T}
    isnan(x) && return x
    (x == 0) && return (n == 0 ? zero(T) : T(-Inf))
    (x == Inf) && return T(Inf)
    (x > 100) && return _logbesseli_asymptotic(n, x; tol, Kmax)
    K = max(1, floor(Int, 0.5 * (sqrt(_fullvalue(x)^2 + n^2) - n))) # maximum of the series at ~x / 2
    # log of the series terms: t_1 = −log Γ(n + 1), t_j = t_{j−1} + log(x² / 4) − log(j − 1) − log(n + j − 1). Two
    # passes without a buffer: up to the largest term t_K, then the terms below it summed downwards from it
    lstep = 2 * log(x) - log(4)
    tK = -loggamma(n + 1) + zero(T)
    for j in 2:K
        tK += lstep - log(j - 1) - log(n + j - 1)
    end
    res = one(T)
    t = tK
    for j in K:-1:2 # t_{j−1} from t_j
        t -= lstep - log(j - 1) - log(n + j - 1)
        res += exp(t - tK)
    end
    k = K
    loga = tK + lstep - log(k) - log(n + k)
    while (loga - tK) > log(tol)
        res += exp(loga - tK)
        k += 1
        loga += lstep - log(k) - log(n + k)
    end
    return log(res) + n * log(x / 2) + tK
end
function _logbesseli_asymptotic(n, x::T; tol = eps(), Kmax = 20) where {T}
    μ = 4 * (n^2)
    s = one(T)
    p = one(T)
    k = 0
    while (abs(p) > tol) && (k < Kmax)
        k += 1
        p *= -(μ - (2 * k - 1)^2) / (k * 8 * x)
        s += p
    end
    return x - 0.5 * (log(2 * π) + log(x)) + log(abs(s))
end

"""
    besseliratio(x)

A(x) = I₁(x) / I₀(x) = d log I₀(x) / dx. Duals are propagated with A'(x) = 1 − A² − A / x (besselix does not accept them).
"""
function besseliratio(x::Real)
    isnan(x) && return x
    (x == Inf) && return one(float(x))
    (x == 0) && return zero(float(x))
    (x > 1.0e4) && return 1 - 1 / (2x) - 1 / (8x^2) - 1 / (8x^3) # asymptotic series, error O(x⁻⁴)
    return besselix(1, x) / besselix(0, x)
end
function besseliratio(x::ForwardDiff.Dual{Tag}) where {Tag}
    v = ForwardDiff.value(x)
    A = besseliratio(v)
    # A' = 1 − A² − A / x cancels badly at large x: use the derivative of the asymptotic series there
    dA = (v == 0 ? 0.5 : v > 1.0e4 ? 1 / (2v^2) + 1 / (4v^3) + 3 / (8v^4) : 1 - A^2 - A / v)
    return ForwardDiff.Dual{Tag}(A, dA * ForwardDiff.partials(x))
end

# half-t distribution (location 0, scale τ, ν degrees of freedom) and its mean (ν > 1; Distributions has no mean for a
# truncated t)
_halft(ν, τ) = truncated(TDist(ν), 0.0, Inf) * τ
_halft_mean(ν, τ) = 2τ * sqrt(ν / π) * exp(loggamma((ν + 1) / 2) - loggamma(ν / 2)) / (ν - 1)

# obj (to minimise) as a twice-differentiable Optim objective at v0: gradient and Hessian by ForwardDiff (value, gradient
# and Hessian from one pass when Optim asks for all three); a non-finite value is +Inf, a rejected step
function _ad_objective(obj, v0::AbstractVector{Float64})
    f(v) = (l = obj(v); isfinite(l) ? l : oftype(l, Inf))
    return TwiceDifferentiable(f, v0; autodiff = AutoForwardDiff())
end

# the result of a minimisation with `_ad_objective`, or an error if it never reached a finite value: a non-finite start
# has a zero gradient, which Optim reports as converged
function _finite_result(res)
    isfinite(Optim.minimum(res)) || error("the objective is not finite at the starting point, $(Optim.minimizer(res))")
    return res
end

"""
    noble_angle(a, b, c, d)

The divergence angle (radians) of the noble number `(a + b τ⁻¹) / (c + d τ⁻¹)`, τ the golden mean and |ad − bc| = 1: 2π
times the number, folded into (0, π] (an angle ψ and 2π − ψ give the same spirals with the opposite orientation). The
golden angle is `noble_angle(1, 0, 2, 1)` and the Lucas angle `noble_angle(1, 0, 3, 1)`; other branches of the
phyllotaxis bifurcation diagram converge towards the other noble numbers. For the `angles` of `PhyllotaxisPrior`.
"""
function noble_angle(a::Integer, b::Integer, c::Integer, d::Integer)
    abs(a * d - b * c) == 1 || throw(ArgumentError("a noble number (a + b τ⁻¹) / (c + d τ⁻¹) needs |ad − bc| = 1"))
    τi = 2 / (1 + sqrt(5)) # τ⁻¹
    den = c + d * τi
    ψ = mod(2π * (a + b * τi) / den, 2π)
    ψ = (ψ > π ? 2π - ψ : ψ)
    ψ > 0 || throw(ArgumentError("the noble number is an integer: no divergence"))
    return ψ
end
