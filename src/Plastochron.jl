# Plastochron
#
# The plastochron along the stem, from daily counts of open flowers (to express commitment delays in hours).
#
# **Rate of one plant.** With t in days from the start of the first day of observation, plant i opens flowers at the rate
# (flowers per day)
#
#     λ(t; τᵢ) = b + a t                 for t ≤ τᵢ               (rise phase; a can be negative)
#              = ρᵢ − d (t − τᵢ)          for τᵢ < t ≤ Aᵢ          (decline phase), ρᵢ = b + a τᵢ
#              = 0                       after the arrest Aᵢ = τᵢ + ρᵢ / d
#
# a piecewise-linear rate whose meristem arrests at a finite time. The change point τᵢ ("failure time": the plastochron
# stops shortening and starts to lengthen) is specific to each plant, `log τᵢ ~ N(μ_τ, σ_τ²)`, and integrated out. If the
# rise phase reaches 0 before τᵢ (a < 0), the plant arrests there. Parameters `p = (a, log b, log d, μ_τ, log σ_τ)`.
#
# **Counts.** The new flowers of a day are Poisson with mean ∫λ over that day (over the whole gap when days were not
# recorded), computed exactly from the cumulative F(t) = ∫₀ᵗ λ. All recorded days are used, including the days without new
# flowers at the end: they date the arrest.
#
# **Plastochron by rank.** Flower k opens when `F(t_k) = k − ½`, and its plastochron is `T_P(k) = 1 / λ(t_k)`. Because λ is
# linear in t on each phase, λ² is linear in F (λ² = b² + 2aF while rising, ρ² − 2d(F − F(τ)) while declining), so `T_P(k)`
# is in closed form. Ranks beyond the expected total keep the last plastochron.
#
# `fit_flowering` (maximum likelihood, Newton trust region with ForwardDiff gradient and Hessian), `plastochron_by_rank`
# (a plant with a given τ, by default the median plant), `plant_plastochron` (one plant, from the posterior of its own τ
# given its counts), and `relative_plastochron` (`r = T_P / T̄_P` for every organ of every stem).
#
# **Delays in hours.** `mean_plastochron` is the mean `T_P` of the measured organs, in hours, with its standard error (delta
# method through the covariance of the fit), and `in_hours` converts a delay in plastochrons (e.g. the core sd of a
# delay law) with its standard error.

# (a, b, d) from p and the end of the rise phase for a change point τ
_abd(p) = (p[1], exp(p[2]), exp(p[3]))
function _phases(p, τ)
    a, b, d = _abd(p)
    τe = (a < 0 ? min(τ, -b / a) : τ) # the rise phase cannot go below 0
    ρ = b + a * τe
    return a, b, d, τe, ρ, τe + ρ / d
end

"λ(t) of a plant with change point τ, `p = (a, log b, log d, μ_τ, log σ_τ)`"
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
const ZGRID = -6.0:0.05:6.0
const LOGWZ = -ZGRID .^ 2 ./ 2 .- log(2π) / 2 .+ log(step(ZGRID))

# log-likelihood of one plant's counts given τ (Poisson increments, constant terms dropped)
function _plant_loglik(p, c, τ)
    ℓ = zero(eltype(p))
    prev, prevF = 0.0, zero(eltype(p))
    for (k, v) in enumerate(c)
        isnan(v) && continue
        F = flowering_cumulative(Float64(k), p, τ)
        μ = F - prevF
        ℓ += (v - prev) * log(μ + 1.0e-10) - μ
        prev, prevF = v, F
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
`p = (a, log b, log d, μ_τ, log σ_τ)`). Returns (; p, nll, se, V, converged): `V` the covariance of p (inverse Hessian),
`se` its standard errors.
"""
function fit_flowering(
        counts::AbstractVector{<:AbstractVector{<:Real}};
        starts = [[a, log(3.0), log(0.5), log(t), log(0.2)] for t in (10.0, 15.0, 20.0) for a in (0.0, 0.1)]
    )
    cs = [Float64.(c) for c in counts]
    f = p -> _nll(p, cs)
    best = nothing
    for p0 in starts
        v0 = collect(Float64, p0)
        res = optimize(_ad_objective(f, v0), v0, NewtonTrustRegion())
        (best === nothing || Optim.minimum(res) < Optim.minimum(best)) && (best = res)
    end
    p = Optim.minimizer(best)
    H = ForwardDiff.hessian(f, p)
    V = inv(H)
    se = sqrt.(max.(diag(V), 0.0))
    return (; p, nll = Optim.minimum(best), se, V, converged = Optim.converged(best))
end

"""
    plastochron_by_rank(fit_or_p, K; τ = nothing)

`T_P(k)` (days) for flower ranks k = 1..K of a plant with change point τ (default: the median plant, `τ = exp(μ_τ)`).
"""
plastochron_by_rank(fit::NamedTuple, K::Integer; kw...) = plastochron_by_rank(fit.p, K; kw...)
function plastochron_by_rank(p::AbstractVector, K::Integer; τ = nothing)
    τ = (τ === nothing ? exp(p[4]) : τ)
    a, b, d, τe, ρ, A = _phases(p, τ)
    Fτ = b * τe + a * τe^2 / 2
    TP = zeros(typeof(ρ), K) # generic: ForwardDiff passes through it (mean_plastochron)
    for k in 1:K
        y = k - 0.5
        λ² = (y ≤ Fτ ? b^2 + 2a * y : ρ^2 - 2d * (y - Fτ))
        TP[k] = (λ² > 1.0e-12 ? 1 / sqrt(λ²) : (k > 1 ? TP[k - 1] : 1 / b))
    end
    return TP
end

"posterior of a plant's change point given its counts: (; τ, w) on the quadrature grid"
tau_posterior(fit::NamedTuple, c::AbstractVector) = tau_posterior(fit.p, c)
function tau_posterior(p::AbstractVector, c::AbstractVector)
    a = _plant_terms(p, Float64.(c))
    return (; τ = exp.(p[4] .+ exp(p[5]) .* ZGRID), w = exp.(a .- logsumexp(a)))
end

"""
    plant_plastochron(fit, c, K)

`T_P(k)` of one plant, k = 1..K, from the posterior of its own change point given its counts `c` (geometric posterior
mean over τ).
"""
plant_plastochron(fit::NamedTuple, c::AbstractVector, K::Integer) = plant_plastochron(fit.p, c, K)
function plant_plastochron(p::AbstractVector, c::AbstractVector, K::Integer)
    post = tau_posterior(p, c)
    keep = findall(>(1.0e-8), post.w)
    lT = sum(post.w[q] .* log.(plastochron_by_rank(p, K; τ = post.τ[q])) for q in keep) ./ sum(post.w[keep])
    return exp.(lT)
end

"""
    relative_plastochron(TP, ranks; Tref = nothing)

`r = T_P(rank) / T̄_P` for every organ of every stem. `TP`: one plastochron-by-rank vector for all stems
(`plastochron_by_rank`) or one per stem (`plant_plastochron`); `ranks`: one vector of flower ranks per stem (rank of
the organ at stem position p; ranks beyond `length(TP)` use the last value). `T̄_P` is the mean over all the organs given
(or `Tref`), so that r averages to 1. Returns (; r, T̄).
"""
function relative_plastochron(TP::AbstractVector, ranks::AbstractVector{<:AbstractVector{<:Integer}}; Tref = nothing)
    TPs = (eltype(TP) <: Real ? fill(TP, length(ranks)) : TP)
    T = [[TPs[s][clamp(k, 1, length(TPs[s]))] for k in rk] for (s, rk) in enumerate(ranks)]
    T̄ = (Tref === nothing ? mean(reduce(vcat, T)) : Tref)
    return (; r = [t ./ T̄ for t in T], T̄)
end

"""
    mean_plastochron(fit, ranks)

Mean plastochron of the measured organs, in hours: `T_P(k)` of the median plant (`plastochron_by_rank`), averaged over
the flower rank k of every organ of every stem (`ranks`: one vector of ranks per stem; ranks beyond the expected total
keep the last plastochron). Its standard error is the delta method through the covariance `V` of `fit_flowering`.
Returns (; T, se).

It sets the scale of the delays when the plastochron varies little over the one or two plastochrons that a delay
spans; `relative_plastochron` shows how much it varies along the stems.
"""
function mean_plastochron(fit::NamedTuple, ranks::AbstractVector{<:AbstractVector{<:Integer}})
    rk = reduce(vcat, collect.(ranks))
    K = maximum(rk)
    g(p) = 24 * mean(plastochron_by_rank(p, K)[rk])
    J = ForwardDiff.gradient(g, fit.p)
    return (; T = g(fit.p), se = sqrt(max(J' * fit.V * J, 0.0)))
end

"""
    in_hours(x, se, T)

A delay `x ± se` in plastochrons (e.g. the core sd `σc` of `delay_law_inference`), in hours, given the mean plastochron
`T = mean_plastochron(fit, ranks)`: the estimate x T̄, and the standard error `√((T̄ se)² + (x se_T̄)²)`. The two
uncertainties are taken as independent, since they come from different data (the angles and the flower counts).
Returns (; estimate, se).
"""
in_hours(x::Real, se::Real, T::NamedTuple) = (estimate = x * T.T, se = sqrt((T.T * se)^2 + (x * T.se)^2))
