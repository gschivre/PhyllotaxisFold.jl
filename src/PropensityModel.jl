# PropensityModel
#
# The propensity model: the angle model with free log-propensities η (one κ per stem from a log-normal population,
# softplus prior on η), optionally outlier angles and a stem twist. Parameters (constrained space, `coef`)
#
#     [η_1..η_k, log μκ, σκ, α_1..α_M (, π_o) (, m_ω, s_ω)]   (M + 1 canonical angles; M = 1 by default)
#
# with symmetric costs `η_{−d} = η_d` (`π_o`: probability of an outlier angle, with `outlier = true`; `m_ω`, `s_ω`: the
# population law of the per-stem twist rates, with a twist prior `torsion`). With free early and late costs, the overall
# balance between them would be set by the prior alone: a tilt `η_d → η_d + a d` leaves every order probability
# unchanged. Early and late organs are told apart by the timing law (`CommitmentTiming`) instead.
#
# The per-stem likelihood is `AngleLikelihood.stem_loglik!` with η free: the gradient in η is analytic
# (`∂ℓ/∂η_d = E_post[N_d] − E_η[N_d]`), as are those in log μκ, σκ, α and `π_o`. The unconstrained parameters u map to θ by
# a Stacked bijector (softplus gaps for η, identity, log, logit), and `(m_ω, log s_ω)` for the twist.
#
# `fit_map(model)` (in `Fitting.jl`) is the one fitting function: it dispatches on the model (outliers or not, twist or
# not) and returns a `PhyllotaxisFit`. The LogDensityProblems interface gives the same log posterior: `logdensity(model, θ)` in the
# constrained space and `TransformedLogDensityProblem(model)` in u (with its analytic gradient).

# default gap prior: (μΔ, σΔ) of gaps with means (0.5, 1.25, 2, 3.75) and variances (0.1, 0.75, 2, 4)
const _DEFAULT_GAPS = η_prior((0.5, 1.25, 2.0, 3.75), (0.1, 0.75, 2.0, 4.0))

"""
    PhyllotaxisPrior(; μΔ, σΔ, mκ = 1.5, sκ = 0.5, νκ = 4, τκ = 0.25, aα = (95,), bα = (5,),
        angles = (Golden_angle, Lucas_angle), aπ = 1, bπ = 49)

Prior of the propensity model; its band k is the length N of the gap tuples (k = 4 by default):

- `η_1..η_k`: `LogNormalSoftPlus(μΔ, σΔ)`, gaps with means (0.5, 1.25, 2, 3.75) and variances (0.1, 0.75, 2, 4) by
  default (`η_prior`);
- log μκ ~ N(mκ, sκ²) and σκ ~ half-t(νκ, τκ): the log-normal population of the per-stem κ;
- the canonical angle of each stem, one of the M + 1 `angles` (radians), with probabilities set by M stick-breaking
  probabilities `α_j ~ Beta(aα[j], bα[j])`: α_1 is the probability of the first angle, α_2 that of the second among
  the stems not at the first, and so on; the last angle takes what is left. The default, one Beta(95, 5) and the angles
  (golden, Lucas), is the golden spiral with probability α ≈ 0.95, else the Lucas one. `aα = bα = ()` with a single
  angle fixes the canonical angle (no α). List the angles from the most to the least expected: a rare angle is given
  a Beta that holds its α near 1 at the previous step (e.g. `aα = (95, 999)`, `bα = (5, 1)` for (golden, Lucas, π)).
  `noble_angle` builds angles from noble numbers;
- `π_o ~ Beta(aπ, bπ)`: the outlier probability (the default Beta(1, 49) has mean 2 %, median 1.4 %), unused with
  `outlier = false`.

Numbers are converted to `Float64`; a single number for `aα`, `bα` or `angles` is a 1-tuple.
`PhyllotaxisPrior{N}(; ...)` also checks the band. The twist prior is given to the model separately (`TorsionPrior`, keyword `torsion`).
"""
struct PhyllotaxisPrior{N, M}
    # LogNormal prior on the gaps Δ_d = η_{d−1} − η_d (mean m, variance v of each gap)
    μΔ::NTuple{N, Float64}
    σΔ::NTuple{N, Float64}

    # log(μκ) mean and standard deviation
    mκ::Float64
    sκ::Float64

    # Half-t prior on σκ
    νκ::Float64
    τκ::Float64

    # the canonical angles (radians) and the Beta priors of their stick-breaking probabilities
    aα::NTuple{M, Float64}
    bα::NTuple{M, Float64}
    angles::Tuple{Float64, Vararg{Float64, M}}

    # Beta prior on the outlier probability π_o (unused with `outlier = false`): mean 2 %
    aπ::Float64
    bπ::Float64
end

_ftuple(x::Real) = (Float64(x),)
_ftuple(x) = Tuple(Float64(v) for v in x)

function PhyllotaxisPrior(;
        μΔ = _DEFAULT_GAPS[1], σΔ = _DEFAULT_GAPS[2], mκ::Real = 1.5, sκ::Real = 0.5, νκ::Real = 4.0, τκ::Real = 0.25,
        aα = (95.0,), bα = (5.0,), angles = (Golden_angle, Lucas_angle), aπ::Real = 1.0, bπ::Real = 49.0
    )
    μΔ, σΔ, aα, bα, ψ = _ftuple(μΔ), _ftuple(σΔ), _ftuple(aα), _ftuple(bα), _ftuple(angles)
    length(μΔ) == length(σΔ) || throw(ArgumentError("μΔ and σΔ: one entry per cost η_1..η_k"))
    length(aα) == length(bα) || throw(ArgumentError("aα and bα: one Beta per stick-breaking probability, as many of each"))
    length(ψ) == length(aα) + 1 ||
        throw(ArgumentError("angles: M + 1 canonical angles for M = $(length(aα)) Betas in aα, bα (got $(length(ψ)) angles)"))
    all(>(0), aα) && all(>(0), bα) || throw(ArgumentError("aα, bα: the Beta parameters must be positive"))
    for i in eachindex(ψ), j in 1:(i - 1)
        # ψ and 2π − ψ give the same sites with the opposite orientation: the model could not tell them apart
        (_samesite(ψ[i], ψ[j]) || _samesite(ψ[i], -ψ[j])) &&
            throw(ArgumentError("angles $(j) and $(i) give the same spirals (equal, or opposite, modulo 2π)"))
    end
    return PhyllotaxisPrior{length(μΔ), length(aα)}(μΔ, σΔ, mκ, sκ, νκ, τκ, aα, bα, ψ, aπ, bπ)
end
function PhyllotaxisPrior{N}(; kw...) where {N}
    p = PhyllotaxisPrior(; kw...)
    length(p.μΔ) == N || throw(ArgumentError("PhyllotaxisPrior{$(N)}: the gap tuples μΔ, σΔ have $(length(p.μΔ)) entries"))
    return p
end
_samesite(a, b) = abs(sin((a - b) / 2)) < 1.0e-9

# log-probabilities of the M + 1 angles from the M stick-breaking α: log p_j = Σ_{i<j} log(1 − α_i) + log α_j
function _angle_logprobs(α::Tuple{T, Vararg{T, M}}) where {T, M}
    return ntuple(Val(M + 2)) do j
        s = zero(T)
        for i in 1:min(j - 1, M + 1)
            s += log1p(-α[i])
        end
        j ≤ M + 1 ? s + log(α[j]) : s
    end
end
_angle_logprobs(::Tuple{}) = (0.0,)
_angle_logprobs(α::Real) = _angle_logprobs((α,)) # (a single α: the golden/Lucas pair)
# ∂ log p_j / ∂α_i: 1/α_i if i = j, −1/(1 − α_i) if i < j, 0 if i > j
_dlogp_dalpha(α, i::Int, j::Int) = (i == j ? inv(α[i]) : (i < j ? -inv(1 - α[i]) : zero(α[i])))

# Type parameters: T the angles, N the band k, O whether angles can be outliers (a Bool), P the twist prior
# (`TorsionPrior`, or `Nothing` without twist): `fit_map` and the likelihood dispatch on them.
struct PhyllotaxisModelAD{T, N, O, P <: Union{Nothing, TorsionPrior}, B, Gr <: NamedTuple}
    data::Vector{Vector{T}} # absolute azimuths (radians), one vector per plant
    k::Int # permutation bandwidth = N
    ngrid_ϕ::Int
    ngrid_κ::Int
    grid::Gr # (; k, ngrid_ϕ, ngrid_κ, ϕ_grid, κ_nodes, κ_logweights, angles), as `stem_loglik!` reads it
    logspace::Bool
    n_max::Int
    prior::PhyllotaxisPrior{N} # (abstract in M: the likelihood reads the angles from `grid`)
    outlier::Bool # = O
    torsion::P # twist hyperprior, or nothing (no twist)
    torsion_x::Vector{Vector{Float64}} # twist coordinate of each organ of each stem
    binv::B # unconstrained ↦ constrained model parameters (inverse of the Stacked bijector)
    graph::PermutationGraph # the banded-permanent graph of band k, shared by every workspace
    pool::WorkspacePool{Float64} # Float64 workspaces, reused across evaluations
    dual_pools::Dict{DataType, Any} # the same for each dual type of the Hessian passes (guarded by pools_lock)
    pools_lock::ReentrantLock
    nodes::Base.RefValue{Union{Nothing, TorsionNodes}} # twist quadrature nodes, placed by fit_map
end

"""
    PhyllotaxisModelAD(data, prior = PhyllotaxisPrior(); outlier = true, torsion = nothing, torsion_x = nothing,
        ngrid_ϕ = 32, ngrid_κ = 16, logspace = false)

The propensity model of the stems `data` (one vector of absolute azimuths per stem, radians, bottom to top); its
band k is the length of the prior tuples. Every organ needs a finite angle: `NaN`, `Inf` or `missing` is an error.

- `outlier` (on by default): each angle is an outlier (uniform on the circle) with probability `π_o`, fitted (prior
  Beta(aπ, bπ)); `outlier = false` drops `π_o`.
- `torsion`: a `TorsionPrior` to fit a twist of each stem, `ω_s ~ N(m_ω, s_ω²)` (degrees per unit of the twist
  coordinate), each rate integrated out; `nothing` (the default): no twist. `torsion_x`: the twist coordinate of each
  stem (`torsion_coordinate(ℓ)` from its internodes, in mm, or `torsion_coordinate(n)`, per position, the default).
  The prior is in degrees per unit of that coordinate.
- `ngrid_ϕ`, `ngrid_κ`: the nuisance grid (offset, precision). Fit coarse grids first, then refine from that solution
  (`fit_map(fine; u0 = coarse_fit.u)`). `logspace`: always run the permanents in log-space (slower; the scaled pass
  falls back to it by itself when needed).
"""
function PhyllotaxisModelAD(
        data::Vector{Vector{T}}, prior::PhyllotaxisPrior{N} = PhyllotaxisPrior(); ngrid_ϕ::Int = 32, ngrid_κ::Int = 16,
        logspace::Bool = false, outlier::Bool = true, torsion::Union{Nothing, TorsionPrior} = nothing, torsion_x = nothing
    ) where {T, N}
    _all_finite(x) = all(v -> v isa Real && isfinite(v), x)
    bad = findall(!_all_finite, data)
    isempty(bad) || throw(ArgumentError("data: every angle must be a finite number (no NaN, Inf or missing); see stems $(bad)"))
    n_min, n_max = extrema(length.(data))
    N ≤ n_min || throw(ArgumentError("the band k = $(N) (length of the prior tuples) exceeds the shortest stem ($(n_min) organs)"))
    tx = (torsion_x === nothing ? [torsion_coordinate(length(y)) for y in data] : [collect(Float64, x) for x in torsion_x])
    (length(tx) == length(data) && all(length.(tx) .== length.(data))) ||
        throw(ArgumentError("torsion_x: one coordinate per organ of each stem"))
    all(_all_finite, tx) || throw(ArgumentError("torsion_x: every coordinate must be a finite number"))
    κ_nodes, κ_weights = gausshermite(ngrid_κ; normalize = true)
    grid = (;
        k = N, ngrid_ϕ, ngrid_κ, ϕ_grid = collect(range(0, 2π; length = ngrid_ϕ + 1)[1:ngrid_ϕ]), κ_nodes,
        κ_logweights = log.(κ_weights), angles = prior.angles,
    )
    binv = Bijectors.inverse(_bijector(N, outlier, length(prior.aα)))
    return PhyllotaxisModelAD{T, N, outlier, typeof(torsion), typeof(binv), typeof(grid)}(
        data, N, ngrid_ϕ, ngrid_κ, grid, logspace, n_max, prior, outlier, torsion, tx, binv, PermutationGraph(N),
        WorkspacePool{Float64}(), Dict{DataType, Any}(), ReentrantLock(), Ref{Union{Nothing, TorsionNodes}}(nothing)
    )
end

function Base.show(io::IO, m::PhyllotaxisModelAD)
    S, n = length(m.data), sum(length, m.data)
    print(io, "PhyllotaxisModelAD: band k = $(m.k), $(S) stems ($(n) organs), ")
    print(io, m.outlier ? "outlier angles" : "no outliers", ", ", _has_twist(m) ? "twist integrated out" : "no twist")
    return print(io, ", grid $(m.ngrid_ϕ) ϕ × $(m.ngrid_κ) κ")
end

_has_outlier(::PhyllotaxisModelAD{T, N, O}) where {T, N, O} = O
_has_twist(::PhyllotaxisModelAD{T, N, O, P}) where {T, N, O, P} = P !== Nothing
# M, the number of stick-breaking α, from the grid's type
_nalpha(::PhyllotaxisModelAD{T, N, O, P, B, Gr}) where {T, N, O, P, B, Gr} = fieldcount(fieldtype(Gr, :angles)) - 1
nmodel(m::PhyllotaxisModelAD) = m.k + 2 + _nalpha(m) + _has_outlier(m)

# Stacked bijector of the model parameters, a tuple of bijectors (type-stable): softplus gaps for η, identity, log,
# logit (each α_j, and π_o with outliers)
function _bijector(k::Int, outlier::Bool, M::Int)
    d = (Normal(0.0, 1.0), _halft(1.0, 1.0), ntuple(_ -> Beta(1.0, 1.0), M)...)
    dists = (LogNormalSoftPlus(zeros(k), ones(k)), (outlier ? (d..., Beta(1.0, 1.0)) : d)...)
    ranges = (1:k, ntuple(i -> (k + i):(k + i), length(dists) - 1)...)
    return Bijectors.Stacked(map(Bijectors.bijector, dists), ranges)
end

# the model part of u and θ (the twist hyperparameters follow)
_umodel(m, u::AbstractVector) = (G = nmodel(m); length(u) == G ? u : u[1:G])

# flat constrained θ from u, and back: [model θ; m_ω; s_ω] with the twist (s_ω = exp(u[end]))
function _constrained(m::PhyllotaxisModelAD, u::AbstractVector)
    θ = m.binv(_umodel(m, u))
    _has_twist(m) || return θ
    G = nmodel(m)
    return vcat(θ, u[G + 1], exp(u[G + 2]))
end
function _unconstrained(m::PhyllotaxisModelAD, θ::AbstractVector)
    G = nmodel(m)
    length(θ) == LogDensityProblems.dimension(m) || throw(DimensionMismatch("θ: $(LogDensityProblems.dimension(m)) parameters"))
    u = Bijectors.inverse(m.binv)(θ[1:G])
    return _has_twist(m) ? vcat(u, θ[G + 1], log(θ[G + 2])) : u
end

function stem_params(m::PhyllotaxisModelAD, u::AbstractVector)
    k = m.k
    θ = m.binv(_umodel(m, u))
    M = _nalpha(m)
    α = ntuple(i -> θ[k + 2 + i], Val(M))
    return (; η = full_eta(m, θ), logμκ = θ[k + 1], σκ = θ[k + 2], α, outlier = (_has_outlier(m) ? θ[k + 3 + M] : nothing))
end
# (the chunk size of ForwardDiff is chosen at run time: its result type is asserted)
params_jacobian(m::PhyllotaxisModelAD, u::AbstractVector{T}) where {T} = ForwardDiff.jacobian(m.binv, _umodel(m, u))::Matrix{T}
function accumulate_gradθ!(gθ::AbstractVector, m::PhyllotaxisModelAD, g::AbstractVector, wgt)
    k, K = m.k, 2 * m.k + 1
    @inbounds for d in 1:k
        gθ[d] += wgt * (g[k + 1 + d] + g[k + 1 - d])
    end
    for i in 1:(2 + _nalpha(m) + _has_outlier(m))
        gθ[k + i] += wgt * g[K + i]
    end
    return gθ
end
function model_logprior(m::PhyllotaxisModelAD, u::AbstractVector)
    θ, lj = Bijectors.with_logabsdet_jacobian(m.binv, _umodel(m, u))
    return _logprior(m, θ) + lj
end

"""
    full_eta(model, θ)

The 2k + 1 log-propensities `η[d + k + 1] = η_d` (`η_0 = 0`) from a parameter vector: `η_{±d} = θ[d]`.
"""
full_eta(model::PhyllotaxisModelAD, θ::NamedTuple) = θ.η
function full_eta(model::PhyllotaxisModelAD, θ::AbstractVector{T}) where {T}
    k = model.k
    η = zeros(T, 2k + 1)
    for d in 1:k
        η[k + 1 + d] = θ[d]
        η[k + 1 - d] = θ[d]
    end
    return η
end

# prior of (η_1..η_k)
_eta_dist(model) = LogNormalSoftPlus(collect(model.prior.μΔ), collect(model.prior.σΔ))

# priors of the other parameters, in the order of θ: log μκ, σκ, α_1..α_M (π_o with outliers; m_ω, s_ω with the twist)
function _nuisance_dists(model::PhyllotaxisModelAD)
    (; mκ, sκ, νκ, τκ, aα, bα, aπ, bπ) = model.prior
    # (M from the grid's concrete type: the model's prior field is abstract in M)
    d = (Normal(mκ, sκ), _halft(νκ, τκ), ntuple(i -> Beta(aα[i], bα[i]), Val(_nalpha(model)))...)
    d = (_has_outlier(model) ? (d..., Beta(aπ, bπ)) : d)
    _has_twist(model) || return d
    (; mt, st, νt, τt) = model.torsion
    return (d..., Normal(mt, st), _halft(νt, τt))
end

"prior mean in the constrained space"
function prior_mean(model::PhyllotaxisModelAD)
    _mean(d) = (d isa Distributions.LocationScale ? _halft_mean(d.ρ.untruncated.ν, d.σ) : mean(d)) # half-t
    return vcat(mean(_eta_dist(model)), [_mean(d) for d in _nuisance_dists(model)])
end

"one parameter vector drawn from the prior"
rand_priorparameters(model::PhyllotaxisModelAD; rng::AbstractRNG = Random.default_rng()) =
    vcat(rand(rng, _eta_dist(model)), [rand(rng, d) for d in _nuisance_dists(model)])

"prior (α/2, 1 − α/2) intervals of each parameter, dim × 2"
function prior_CIs(model::PhyllotaxisModelAD; α = 0.05)
    dists = _nuisance_dists(model)
    ci(q) = vcat(marginal_quantiles(_eta_dist(model), q), [quantile(d, q) for d in dists])
    return hcat(ci(0.5 * α), ci(1 - 0.5 * α))
end

# prior sd of each parameter (η_d: the sum of d independent log-normal gaps); Inf for a half-t without variance
function _prior_sds(model::PhyllotaxisModelAD)
    vΔ = [(exp(s^2) - 1) * exp(2μ + s^2) for (μ, s) in zip(model.prior.μΔ, model.prior.σΔ)]
    sd(d) = (d isa Distributions.LocationScale ? _halft_sd(d.ρ.untruncated.ν, d.σ) : std(d))
    return vcat(sqrt.(cumsum(vΔ)), [sd(d) for d in _nuisance_dists(model)])
end
_halft_sd(ν, τ) = (ν > 2 ? sqrt(max(τ^2 * ν / (ν - 2) - _halft_mean(ν, τ)^2, 0.0)) : Inf)

"log prior density of the constrained model parameters (the twist hyperparameters excluded)"
function _logprior(model::PhyllotaxisModelAD, θ::AbstractVector)
    k = model.k
    d = _nuisance_dists(model)
    G = Val(2 + _nalpha(model) + _has_outlier(model))
    return logpdf(_eta_dist(model), view(θ, 1:k)) + sum(ntuple(i -> logpdf(d[i], θ[k + i]), G))
end
# and of the twist hyperparameters (m_ω, s_ω)
function _logprior_twist(model::PhyllotaxisModelAD, mω, sω)
    d = _nuisance_dists(model)
    return logpdf(d[end - 1], mω) + logpdf(d[end], sω)
end

#################################################################################################################
# LogDensityProblems
#################################################################################################################

LogDensityProblems.dimension(model::PhyllotaxisModelAD) = nmodel(model) + 2 * _has_twist(model)
LogDensityProblems.capabilities(::Type{<:PhyllotaxisModelAD}) = LogDensityProblems.LogDensityOrder{0}()

# total log-likelihood at u, generic in its number type: with the twist, each stem's rate integrated over the model's
# quadrature nodes
function _loglik(m::PhyllotaxisModelAD, u::AbstractVector{T}) where {T}
    _has_twist(m) || return _data_value_grad(m, u, nothing, false)[1]
    nodes = _nodes(m)
    p = stem_params(m, u)
    return sum(_per_stem((s, fws) -> logsumexp(_node_terms(m, p, u, s, nodes, fws)), m, T, T))
end

"""
    logdensity(model, θ)

Log posterior in the constrained space θ (`coef`; dimension `k + 2 + M` for M + 1 canonical angles, + 1 with outliers,
+ 2 with the twist), the objective
of `fit_map` up to the Jacobian of the parameter map. Per stem: `log ∫ N(log κ) ½ Σ_o Σ_ψ p(ψ) (1/2π) ∫ dϕ perm(A) / (perm(W)
(2π I₀(κ))ⁿ)`, `A_{i, i+d} = exp(η_d + κ cos(y_i − ϕ − o (i + d − 1) ψ))`; rectangle rule in ϕ, Gauss–Hermite in log κ.
With the twist, each stem's rate is integrated over the quadrature nodes placed by `fit_map` (fit the model first).
"""
function LogDensityProblems.logdensity(model::PhyllotaxisModelAD, θ::AbstractVector)
    G = nmodel(model)
    lp = _logprior(model, view(θ, 1:G))
    _has_twist(model) && (lp += _logprior_twist(model, θ[G + 1], θ[G + 2]))
    isfinite(lp) || return lp
    return lp + _loglik(model, _unconstrained(model, θ))
end

"""
    TransformedLogDensityProblem(model)

The log posterior of `model` in the unconstrained space u (the space of `fit_map`, Jacobian included), for
LogDensityProblems users (e.g. a sampler), with its analytic gradient (`logdensity_and_gradient`).
"""
struct TransformedLogDensityProblem{M <: PhyllotaxisModelAD}
    model::M
end
_logpost_u(m, u) = _loglik(m, u) + model_logprior(m, u) + (_has_twist(m) ? torsion_hyperprior(m.torsion, u[end - 1], u[end]) : 0)
LogDensityProblems.logdensity(prob::TransformedLogDensityProblem, u::AbstractVector) = _logpost_u(prob.model, u)
LogDensityProblems.dimension(prob::TransformedLogDensityProblem) = LogDensityProblems.dimension(prob.model)
LogDensityProblems.capabilities(::Type{<:TransformedLogDensityProblem}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(prob::TransformedLogDensityProblem, u::AbstractVector)
    m = prob.model
    l, g, _ = (_has_twist(m) ? twist_vgh(m, _nodes(m), u, 1) : angle_vgh(m, u, nothing, 1))
    return l, g
end

#################################################################################################################
# parameters
#################################################################################################################

"""
    constrained_parameters(model, u)

(; η, logμκ, σκ, α, outlier, `angle_probs`) from the unconstrained u (+ `m_ω`, `s_ω` with the twist): η the full 2k + 1
log-propensities, α the M stick-breaking probabilities (a tuple, `()` for a single canonical angle), outlier = `π_o`, or
0 without outliers, and `angle_probs` the probability of each canonical angle.
"""
function constrained_parameters(model::PhyllotaxisModelAD, u::AbstractVector)
    p = stem_params(model, u)
    nt = (; p.η, p.logμκ, p.σκ, p.α, outlier = (p.outlier === nothing ? zero(p.logμκ) : p.outlier),
        angle_probs = exp.(_angle_logprobs(p.α)))
    _has_twist(model) || return nt
    G = nmodel(model)
    return (; nt..., m_ω = u[G + 1], s_ω = exp(u[G + 2]))
end

"""
    unconstrained_parameters(model; η, logμκ, σκ, α, outlier, m_ω, s_ω)
    unconstrained_parameters(model, θ::AbstractVector)

u from named values (η: `η_1..η_k`, or the full 2k + 1 vector, symmetric, `η_0 = 0`; α: the M stick-breaking
probabilities, a tuple or vector, or a number when M = 1; α, outlier, `m_ω`, `s_ω` default to their prior means and
are used only by a model that has them), or from the flat constrained vector θ (`coef`).
"""
function unconstrained_parameters(
        model::PhyllotaxisModelAD; η, logμκ, σκ, α = map((a, b) -> a / (a + b), model.prior.aα, model.prior.bα),
        outlier = mean(Beta(model.prior.aπ, model.prior.bπ)), m_ω = nothing, s_ω = nothing
    )
    k = model.k
    if length(η) == 2k + 1
        (iszero(η[k + 1]) && η[1:k] ≈ reverse(η[(k + 2):end])) ||
            throw(ArgumentError("η: the full vector must be symmetric, η_{−d} = η_d, with η_0 = 0"))
        η = η[(k + 2):end]
    end
    length(η) == k || throw(ArgumentError("η: k = $(k) costs η_1..η_k, or the full vector of 2k + 1"))
    αv = (α isa Real ? [α] : collect(α))
    length(αv) == _nalpha(model) || throw(ArgumentError("α: $(_nalpha(model)) stick-breaking probabilities"))
    θ = vcat(η, [logμκ, σκ], αv, (_has_outlier(model) ? [outlier] : Float64[]))
    if _has_twist(model)
        pm = prior_mean(model)
        θ = vcat(θ, something(m_ω, pm[end - 1]), something(s_ω, pm[end]))
    end
    return _unconstrained(model, θ)
end
unconstrained_parameters(model::PhyllotaxisModelAD, θ::AbstractVector) = _unconstrained(model, θ)
prior_mean_parameters(model::PhyllotaxisModelAD) = _unconstrained(model, prior_mean(model))

#################################################################################################################
# post-processing
#################################################################################################################

"the η block (`η_1..η_k`) of a parameter vector (or of each column of a matrix of draws)"
log_propensity(model::PhyllotaxisModelAD, θ::AbstractVecOrMat) = view(θ, 1:(model.k), :)

"""
    displacement_probability(model, θ)

P(|d|), d = 0..k, from the local softmax of (`η_{−k}`, …, `η_k`), for a parameter vector or each column of a matrix of
draws: the instantaneous propensities, not the realised displacement frequencies (`estimate_displacement_frequencies`,
same rows).
"""
function displacement_probability(model::PhyllotaxisModelAD, θ::AbstractVecOrMat)
    η = log_propensity(model, θ)
    prop = vcat(zeros(1, size(η, 2)), η .+ log(2)) # d and −d together
    softmax!(prop; dims = 1)
    return prop
end

"""
    estimate_displacement_frequencies(model, q = nothing; N, n, α, rng)

Realised P(|d|), d = 0..k, in the bulk of a stem of n organs, averaged over θ ~ q (any object with `rand`, e.g. the Laplace
MvNormal mapped back) or over the prior: exact edge marginals by forward–backward, averaged over positions. Returns
(mean, intervals (k+1) × 2).
"""
function estimate_displacement_frequencies(model::PhyllotaxisModelAD, q = nothing; N::Int = 10_000, n::Int = 60, α = 0.05, rng::AbstractRNG = Random.default_rng())
    (; k, logspace) = model
    ws = PermanentWorkspace(Float64, k, n; logspace = logspace)
    p = zeros(k + 1, N)
    diagmean(rho, d) = mean(rho[i, d + k + 1] for i in max(1, 1 - d):min(n, n - d))
    for s in 1:N
        θ = (q === nothing ? rand_priorparameters(model; rng) : rand(rng, q))
        toeplitz_weights!(ws, full_eta(model, θ), n)
        forward!(ws, n)
        rho = backward!(ws, n)
        for d in 0:k
            p[d + 1, s] = (d == 0 ? diagmean(rho, 0) : diagmean(rho, d) + diagmean(rho, -d)) # P(d) + P(−d)
        end
    end
    return vec(mean(p; dims = 2)), mapslices(x -> quantile(x, [α / 2, 1 - α / 2]), p; dims = 2)
end

"""
    estimate_permutation_frequencies(model, q = nothing; N, n, α, rng)

Fraction of the organs of a stem of n organs that lie in cycles of length 1 (fixed points), 2 (swaps), 3, … of the
sampled permutations, θ ~ q or the prior. Returns (mean, intervals n × 2), row ℓ for cycles of length ℓ.
"""
function estimate_permutation_frequencies(model::PhyllotaxisModelAD, q = nothing; N::Int = 10_000, n::Int = 60, α = 0.05, rng::AbstractRNG = Random.default_rng())
    (; k, logspace) = model
    ws = PermanentWorkspace(Float64, k, n; logspace = logspace)
    p = zeros(n, N)
    for s in 1:N
        θ = (q === nothing ? rand_priorparameters(model; rng) : rand(rng, q))
        toeplitz_weights!(ws, full_eta(model, θ), n)
        σ = sample_permutation!(rng, ws, n)
        for (len, c) in countmap(cycle_lengths(σ))
            p[len, s] = len * c / n # organs in cycles of this length
        end
    end
    return vec(mean(p; dims = 2)), mapslices(x -> quantile(x, [α / 2, 1 - α / 2]), p; dims = 2)
end

"""
    sample_prior(model; n, Nsample, rng)

Prior predictive: η, κ, ψ, ϕ, o from the prior, σ from the permanent model, angles from the von Mises emission; with
outliers, each angle replaced by a uniform one with probability `π_o ~ Beta(aπ, bπ)`; with a twist, a rate
`ω ~ N(m_ω, s_ω²)` with (`m_ω`, `s_ω`) from the twist prior, on the coordinate `torsion_coordinate(n)` (per position).
Returns one (; data, permutation, `offset_angle`, orientation, `canonical_angle`, dispersion, outliers, twist) per sample.
"""
function sample_prior(model::PhyllotaxisModelAD{T}; n::Int = round(Int, mean(map(length, model.data))), Nsample::Int = 1, rng::AbstractRNG = Random.default_rng()) where {T}
    (; k, logspace) = model
    dists = _nuisance_dists(model)
    M = _nalpha(model)
    dlogμκ, dσκ, dα = dists[1], dists[2], dists[3:(2 + M)]
    x = torsion_coordinate(n)
    ws = PermanentWorkspace(Float64, k, n; logspace = logspace)
    return map(1:Nsample) do _
        toeplitz_weights!(ws, full_eta(model, rand(rng, _eta_dist(model))), n)
        σ = sample_permutation!(rng, ws, n)
        κ = clamp(rand(rng, LogNormal(rand(rng, dlogμκ), rand(rng, dσκ))), 1.0e-3, 1.0e3) # for numerical stability
        ψ = _draw_angle(rng, Tuple(rand(rng, d) for d in dα), model.prior.angles)
        ϕ = rand(rng, Uniform(0.0, 2π))
        o = (rand(rng, Bernoulli(0.5)) ? 1 : -1)
        πo = (_has_outlier(model) ? rand(rng, dists[3 + M]) : 0.0)
        ω = (_has_twist(model) ? rand(rng, Normal(rand(rng, dists[end - 1]), rand(rng, dists[end]))) : 0.0)
        bad = [rand(rng) < πo for _ in 1:n]
        y = T[bad[i] ? 2π * rand(rng) : mod2pi(rand(rng, VonMises(mod2pi(ϕ + o * (σ[i] - 1) * ψ + deg2rad(ω) * x[i]), κ))) for i in 1:n]
        (; data = y, permutation = σ, offset_angle = ϕ, orientation = o, canonical_angle = ψ, dispersion = κ, outliers = findall(bad), twist = ω)
    end
end
