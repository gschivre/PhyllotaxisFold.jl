"""
    Common

Constants and small numerical tools shared by every block of the model: the two canonical divergence angles, log I₀ and
the Bessel ratio (generic in the number type, so that they work with nested ForwardDiff duals), the half-t density, the
-Inf-safe logaddexp and the scaled logistic map used for bounded parameters.
"""
module Common

using ForwardDiff, SpecialFunctions, Distributions, LogExpFunctions

export Golden_angle, Lucas_angle, logbesseli, besseliratio, logpdf_halft, dlogpdf_halft, fit_map

const Golden_angle = π * (3 - sqrt(5)) # rad2deg(Golden_angle) ≈ 137.5
const Lucas_angle = π * (1 - inv(sqrt(5))) # rad2deg(Lucas_angle) ≈ 99.5

"""
    fit_map(model; kwargs...)

MAP estimate by Newton's method with trust region, analytic gradient and a Hessian from one dual pass over it.
Methods: `fit_map(::PhyllotaxisModelAD)` and its `TransformedLogDensityProblem`; returns (; θ, u, H, result),
θ = `constrained_parameters(model, u)`.
"""
function fit_map end

# logaddexp that returns the other argument exactly when one is -Inf (keeps ForwardDiff partials free of NaN)
@inline _isneginf(x) = isinf(x) && x < 0 # (x == -Inf is false for a dual number with nonzero partials)
@inline _logaddexp(a, b) = (_isneginf(a) ? b : (_isneginf(b) ? a : logaddexp(a, b)))

# bounded parameter: u ∈ ℝ ↦ lo + (hi − lo) logistic(u)
_scaled_logistic(u, (lo, hi)) = lo + (hi - lo) * logistic(u)

# Float64 value of possibly nested dual numbers
_fullvalue(x) = (x isa ForwardDiff.Dual ? _fullvalue(ForwardDiff.value(x)) : x)

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
    K = max(1, floor(Int, 0.5 * (sqrt(_fullvalue(x) ^ 2 + n ^ 2) - n))) # maximum of the series at ~x / 2
    rec = zeros(T, K) # record the first K values
    rec[1] = -loggamma(n + 1)
    logx2 = 2 * log(x)
    log4 = log(4)
    for k in 2:K
        rec[k] = rec[k - 1] + logx2 - log4 - log(k - 1) - log(n + k - 1)
    end
    res = one(T)
    for i in (K - 1):-1:1
        res += exp(rec[i] - rec[end])
    end
    k = K
    loga = rec[end] + logx2 - log4 - log(k) - log(n + k)
    while (loga - rec[end]) > log(tol)
        res += exp(loga - rec[end])
        k += 1
        loga += logx2 - log4 - log(k) - log(n + k)
    end
    return log(res) + n * log(x / 2) + rec[end]
end
function _logbesseli_asymptotic(n, x::T; tol = eps(), Kmax = 20) where {T}
    μ = 4 * (n ^ 2)
    s = one(T)
    p = one(T)
    k = 0
    while (abs(p) > tol) && (k < Kmax)
        k += 1
        p *= -(μ - (2 * k - 1) ^ 2) / (k * 8 * x)
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
    (x == Inf) && return 1.0
    (x == 0) && return 0.0
    (x > 1e4) && return 1 - 1 / (2x) - 1 / (8x^2) - 1 / (8x^3) # asymptotic series, error O(x⁻⁴)
    return besselix(1, x) / besselix(0, x)
end
function besseliratio(x::ForwardDiff.Dual{Tag}) where {Tag}
    v = ForwardDiff.value(x)
    A = besseliratio(v)
    # A' = 1 − A² − A / x cancels badly at large x: use the derivative of the asymptotic series there
    dA = (v == 0 ? 0.5 : v > 1e4 ? 1 / (2v^2) + 1 / (4v^3) + 3 / (8v^4) : 1 - A^2 - A / v)
    return ForwardDiff.Dual{Tag}(A, dA * ForwardDiff.partials(x))
end

# half-t density (location 0, scale τ) and its derivative in x
logpdf_halft(x, ν, τ) = log(2) + logpdf(TDist(ν), x / τ) - log(τ)
dlogpdf_halft(x, ν, τ) = -(ν + 1) * x / (ν * τ ^ 2 + x ^ 2)

end # module Common
