"""
    Plastochron

The plastochron along the stem, from daily counts of open flowers: the input of the tempo option of `FoldModel`.

**Rate of one plant.** With t in days from the start of the first day of observation, plant i opens flowers at the rate
(flowers per day)

    λ(t; τᵢ) = b + a t                 for t ≤ τᵢ               (rise phase; a can be negative)
             = ρᵢ − d (t − τᵢ)          for τᵢ < t ≤ Aᵢ          (decline phase), ρᵢ = b + a τᵢ
             = 0                       after the arrest Aᵢ = τᵢ + ρᵢ / d

a piecewise-linear rate whose meristem arrests at a finite time. The change point τᵢ ("failure time": the plastochron
stops shortening and starts to lengthen) is specific to each plant, log τᵢ ~ N(μ_τ, σ_τ²), and integrated out. If the
rise phase reaches 0 before τᵢ (a < 0), the plant arrests there. Parameters p = (a, log b, log d, μ_τ, log σ_τ).

**Counts.** The new flowers of a day are Poisson with mean ∫λ over that day (over the whole gap when days were not
recorded), computed exactly from the cumulative F(t) = ∫₀ᵗ λ. All recorded days are used, including the days without new
flowers at the end: they date the arrest.

**Plastochron by rank.** Flower k opens when F(t_k) = k − ½, and its plastochron is T_P(k) = 1 / λ(t_k). Because λ is
linear in t on each phase, λ² is linear in F (λ² = b² + 2aF while rising, ρ² − 2d(F − F(τ)) while declining), so T_P(k)
is in closed form. Ranks beyond the expected total keep the last plastochron.

`fit_flowering` (maximum likelihood, Newton trust region with ForwardDiff gradient and Hessian), `plastochron_by_rank`
(a plant with a given τ, by default the median plant), `plant_plastochron` (one plant, from the posterior of its own τ
given its counts), and `relative_plastochron` (r = T_P / T̄_P for every organ of every stem).
"""
module Plastochron

using Statistics
using LinearAlgebra: diag
using LogExpFunctions, ForwardDiff, Optim, NLSolversBase

export flowering_rate, flowering_cumulative, fit_flowering, plastochron_by_rank, plant_plastochron, tau_posterior,
       relative_plastochron

# (a, b, d) from p and the end of the rise phase for a change point τ
_abd(p) = (p[1], exp(p[2]), exp(p[3]))
function _phases(p, τ)
    a, b, d = _abd(p)
    τe = (a < 0 ? min(τ, -b / a) : τ) # the rise phase cannot go below 0
    ρ = b + a * τe
    return a, b, d, τe, ρ, τe + ρ / d
end

"λ(t) of a plant with change point τ, p = (a, log b, log d, μ_τ, log σ_τ)"
function flowering_rate(t, p, τ)
    a, b, d, τe, ρ, A = _phases(p, τ)
    t ≤ τe && return b + a * t
    t ≤ A && return ρ - d * (t - τe)
    return zero(ρ)
end

"F(t) = ∫₀ᵗ λ of a plant with change point τ"
function flowering_cumulative(t, p, τ)
    a, b, d, τe, ρ, A = _phases(p, τ)
    t ≤ τe && return b * t + a * t^2 / 2
    Fτ = b * τe + a * τe^2 / 2
    s = min(t, A) - τe
    return Fτ + ρ * s - d * s^2 / 2
end

# quadrature on z = (log τ − μ_τ) / σ_τ: a dense grid, robust to the kinks of the piecewise rate
const ZGRID = collect(-6.0:0.05:6.0)
const LOGWZ = log.(exp.(-ZGRID .^ 2 ./ 2) ./ sqrt(2π) .* 0.05)

# log-likelihood of one plant's counts given τ (Poisson increments, constant terms dropped)
function _plant_loglik(p, c, τ)
    ℓ = zero(eltype(p))
    prev, prevF, prevk = 0.0, zero(eltype(p)), 0
    for (k, v) in enumerate(c)
        isnan(v) && continue
        F = flowering_cumulative(Float64(k), p, τ)
        μ = F - prevF
        ℓ += (v - prev) * log(μ + 1e-10) - μ
        prev, prevF, prevk = v, F, k
    end
    return ℓ
end

# per-plant terms log w_q + ℓ(τ_q) on the z grid
_plant_terms(p, c) = [LOGWZ[q] + _plant_loglik(p, c, exp(p[4] + exp(p[5]) * ZGRID[q])) for q in eachindex(ZGRID)]

_nll(p, counts) = -sum(logsumexp(_plant_terms(p, c)) for c in counts)

"""
    fit_flowering(counts; starts)

Maximum-likelihood rate of a group of plants. `counts`: one vector per plant of the cumulative number of open flowers at
the end of each day (day 1 = first day of observation; `NaN` for a day without record), up to the last recorded day.
Newton trust region with the ForwardDiff gradient and Hessian, best of the starting points `starts` (vectors
p = (a, log b, log d, μ_τ, log σ_τ)). Returns (; p, nll, se, converged), `se` the standard errors of p (inverse Hessian).
"""
function fit_flowering(counts::AbstractVector{<:AbstractVector{<:Real}};
        starts = [[a, log(3.0), log(0.5), log(t), log(0.2)] for t in (10.0, 15.0, 20.0) for a in (0.0, 0.1)])
    cs = [Float64.(c) for c in counts]
    f = p -> _nll(p, cs)
    fgh! = (F, G, H, p) -> begin
        (H !== nothing) && (H .= ForwardDiff.hessian(f, p))
        (G !== nothing) && (G .= ForwardDiff.gradient(f, p))
        return f(p)
    end
    best = nothing
    for p0 in starts
        res = optimize(NLSolversBase.only_fgh!(fgh!), collect(Float64, p0), NewtonTrustRegion())
        (best === nothing || Optim.minimum(res) < Optim.minimum(best)) && (best = res)
    end
    p = Optim.minimizer(best)
    H = ForwardDiff.hessian(f, p)
    se = sqrt.(max.(diag(inv(H)), 0.0))
    return (; p, nll = Optim.minimum(best), se, converged = Optim.converged(best))
end

"""
    plastochron_by_rank(fit_or_p, K; τ = nothing)

T_P(k) (days) for flower ranks k = 1..K of a plant with change point τ (default: the median plant, τ = exp(μ_τ)).
"""
plastochron_by_rank(fit::NamedTuple, K::Integer; kw...) = plastochron_by_rank(fit.p, K; kw...)
function plastochron_by_rank(p::AbstractVector, K::Integer; τ = nothing)
    τ = (τ === nothing ? exp(p[4]) : τ)
    a, b, d, τe, ρ, A = _phases(p, τ)
    Fτ = b * τe + a * τe^2 / 2
    TP = zeros(K)
    for k in 1:K
        y = k - 0.5
        λ² = (y ≤ Fτ ? b^2 + 2a * y : ρ^2 - 2d * (y - Fτ))
        TP[k] = (λ² > 1e-12 ? 1 / sqrt(λ²) : (k > 1 ? TP[k - 1] : 1 / b))
    end
    return TP
end

"posterior of a plant's change point given its counts: (; τ, w) on the quadrature grid"
function tau_posterior(fit, c::AbstractVector)
    p = fit isa NamedTuple ? fit.p : fit
    a = _plant_terms(p, Float64.(c))
    return (; τ = exp.(p[4] .+ exp(p[5]) .* ZGRID), w = exp.(a .- logsumexp(a)))
end

"""
    plant_plastochron(fit, c, K)

T_P(k) of one plant, k = 1..K, from the posterior of its own change point given its counts `c` (geometric posterior
mean over τ).
"""
function plant_plastochron(fit, c::AbstractVector, K::Integer)
    p = fit isa NamedTuple ? fit.p : fit
    post = tau_posterior(p, c)
    keep = findall(>(1e-8), post.w)
    lT = sum(post.w[q] .* log.(plastochron_by_rank(p, K; τ = post.τ[q])) for q in keep) ./ sum(post.w[keep])
    return exp.(lT)
end

"""
    relative_plastochron(TP, ranks; Tref = nothing)

r = T_P(rank) / T̄_P for every organ of every stem. `TP`: one plastochron-by-rank vector for all stems
(`plastochron_by_rank`) or one per stem (`plant_plastochron`); `ranks`: one vector of flower ranks per stem (rank of
the organ at stem position p; ranks beyond `length(TP)` use the last value). T̄_P is the mean over all the organs given
(or `Tref`), so that r averages to 1 and γ of `FoldModel(...; tempo = r)` is the precision at the mean tempo.
Returns (; r, T̄).
"""
function relative_plastochron(TP::AbstractVector, ranks::AbstractVector{<:AbstractVector{<:Integer}}; Tref = nothing)
    TPs = (eltype(TP) <: Real ? fill(TP, length(ranks)) : TP)
    T = [[TPs[s][clamp(k, 1, length(TPs[s]))] for k in rk] for (s, rk) in enumerate(ranks)]
    T̄ = (Tref === nothing ? mean(reduce(vcat, T)) : Tref)
    return (; r = [t ./ T̄ for t in T], T̄)
end

end # module Plastochron
