"""
    PropensityModel

The propensity model (A): the angle model with free log-propensities η (one κ per stem from a log-normal population,
softplus prior on η). Parameters (constrained space)

    [η_1..η_k, log μκ, σκ, α (, π_o)]

with symmetric costs η_{−d} = η_d (π_o: probability of an outlier angle, fitted unless `outlier = false`). With free early and
late costs, the overall balance between them would be set by the prior alone: a tilt η_d → η_d + a d leaves every order
probability unchanged. Early and late organs are told apart by the timing law (`CommitmentTiming`) instead.

The per-stem likelihood is `AngleLikelihood.stem_loglik!` with η free: the gradient in η is
analytic (∂ℓ/∂η_d = E_post[N_d] − E_η[N_d]), as are those in log μκ, σκ and α. `TransformedLogDensityProblem` maps to
ℝ^dim with the Stacked bijector (softplus gaps for η, identity, log, logit); `fit_map` is Newton trust region with the
analytic gradient (chain rule through the bijector Jacobian) and the Hessian from one dual pass over it. The log density
stays generic in the number type (ForwardDiff works on it).

The model itself is passed to `fit_map`, `fit_map_torsion!`, `update_torsion!`,
`torsion_nodes`, `stem_loglik`, `angle_loglikelihood`, `decode_orders` and `order_posterior`, always with the
unconstrained parameters u (`constrained_parameters`, `unconstrained_parameters`, `prior_mean_parameters` convert);
`TransformedLogDensityProblem(model)` remains for LogDensityProblems users and is accepted wherever the model is.
`simulate_stems(η; ...)` draws stems from the model, `sample_prior(model)` from its prior predictive.
"""
module PropensityModel

using Random, Statistics
using Distributions, LogExpFunctions, StatsBase, FastGaussQuadrature
using ForwardDiff, Optim
using LogDensityProblems: LogDensityProblems
using Bijectors: Bijectors
using ..Common: Golden_angle, Lucas_angle, _halft, _halft_mean
import ..Common: fit_map
using ..Permanents: PermanentWorkspace, toeplitz_weights!, forward!, backward!, sample_permutation!, cycle_lengths
using ..PropensityPriors: LogNormalSoftPlus, η_prior, marginal_quantiles
using ..Torsion: StemTorsion, TorsionPrior, MAP_OPTIONS, torsion_coordinate, torsion_corrected
import ..Torsion: corrected_angles, update_torsion!, torsion_nodes, fit_map_torsion!
using ..AngleLikelihood: StemWorkspace, stem_loglik!, angle_fit_map, angle_update_torsion!, angle_torsion_nodes,
    angle_fit_map_torsion!, _chunks, _workspace, _simulate_angles
import ..AngleLikelihood: stem_value_grad, global_logprior, angle_model, stem_etas, angle_nuisance, stem_grid, simulate_stems,
    constrained_parameters, unconstrained_parameters, prior_mean_parameters

export PhyllotaxisPrior, PhyllotaxisModelAD, TransformedLogDensityProblem, full_eta, prior_mean, rand_priorparameters,
    prior_CIs, log_propensity, displacement_probability, estimate_displacement_frequencies,
    estimate_permutation_frequencies, sample_prior

# default gap prior: (μΔ, σΔ) of gaps with means (0.5, 1.25, 2, 3.75) and variances (0.1, 0.75, 2, 4)
const _DEFAULT_GAPS = η_prior((0.5, 1.25, 2.0, 3.75), (0.1, 0.75, 2.0, 4.0))

"""
    PhyllotaxisPrior(; μΔ, σΔ, mκ = 1.5, sκ = 0.5, νκ = 4, τκ = 0.25, aα = 95, bα = 5, aπ = 1, bπ = 49)

Prior of the propensity model; its band k is the length N of the gap tuples (k = 4 by default):

- η_1..η_k: `LogNormalSoftPlus(μΔ, σΔ)`, gaps with means (0.5, 1.25, 2, 3.75) and variances (0.1, 0.75, 2, 4) by
  default (`η_prior`);
- log μκ ~ N(mκ, sκ²) and σκ ~ half-t(νκ, τκ): the log-normal population of the per-stem κ;
- α ~ Beta(aα, bα): the probability of the golden spiral (else the Lucas one);
- π_o ~ Beta(aπ, bπ): the outlier probability (mean 2 %, median 1.4 %), unused with `outlier = false`.
"""
Base.@kwdef struct PhyllotaxisPrior{N}
    # LogNormal prior on the gaps Δ_d = η_{d−1} − η_d (mean m, variance v of each gap)
    μΔ::NTuple{N, Float64} = _DEFAULT_GAPS[1]
    σΔ::NTuple{N, Float64} = _DEFAULT_GAPS[2]

    # log(μκ) mean and standard deviation
    mκ::Float64 = 1.5
    sκ::Float64 = 0.5

    # Half-t prior on σκ
    νκ::Float64 = 4.0
    τκ::Float64 = 0.25

    # Beta prior on the Golden vs Lucas sequence probability
    aα::Float64 = 95.0
    bα::Float64 = 5.0

    # Beta prior on the outlier probability π_o (unused with `outlier = false`): mean 2 %
    aπ::Float64 = 1.0
    bπ::Float64 = 49.0
end

struct PhyllotaxisModelAD{T, N, B}
    data::Vector{Vector{T}} # absolute azimuths (radians), one vector per plant
    k::Int # permutation bandwidth = N
    ngrid_ϕ::Int
    ngrid_κ::Int
    κ_nodes::Vector{Float64} # standard normal Gauss–Hermite nodes
    κ_logweights::Vector{Float64}
    logspace::Bool
    n_max::Int
    prior::PhyllotaxisPrior{N}
    torsion::StemTorsion # per-stem torsion coordinates and rates (rates 0 until fitted)
    outlier::Bool # angles are a mixture with uniform outliers (one more parameter, π_o)
    binv::B # unconstrained u ↦ constrained θ (inverse of the Stacked bijector)
end

"""
    PhyllotaxisModelAD(data, prior = PhyllotaxisPrior(); ngrid_ϕ = 32, ngrid_κ = 16, logspace = false, torsion_x = nothing,
        outlier = true)

k is the length of the prior tuples. Fit coarse grids first, then refine from that solution. `torsion_x`: the torsion
coordinate of each stem (`torsion_coordinate(ℓ)` from its internodes, in mm, or `torsion_coordinate(n)`, per position,
the default). The rates start at 0; `fit_map_torsion!(model)` fits them. `outlier` (on by default): each angle is an
outlier (uniform on the circle) with probability π_o, fitted (prior Beta(aπ, bπ)); `outlier = false` drops π_o.
"""
function PhyllotaxisModelAD(
        data::Vector{Vector{T}}, prior::PhyllotaxisPrior{N} = PhyllotaxisPrior(); ngrid_ϕ::Int = 32, ngrid_κ::Int = 16,
        logspace::Bool = false, torsion_x = nothing, outlier::Bool = true
    ) where {T, N}
    n_min, n_max = extrema(length.(data))
    N ≤ n_min || throw(ArgumentError("the band k = $(N) (length of the prior tuples) exceeds the shortest stem ($(n_min) organs)"))
    tx = (torsion_x === nothing ? [torsion_coordinate(length(y)) for y in data] : torsion_x)
    (length(tx) == length(data) && all(length.(tx) .== length.(data))) ||
        throw(ArgumentError("torsion_x: one coordinate per organ of each stem"))
    κ_nodes, κ_weights = gausshermite(ngrid_κ; normalize = true)
    binv = Bijectors.inverse(_bijector(N, outlier))
    return PhyllotaxisModelAD{T, N, typeof(binv)}(
        data, N, ngrid_ϕ, ngrid_κ, κ_nodes, log.(κ_weights), logspace, n_max,
        prior, StemTorsion(tx), outlier, binv
    )
end

corrected_angles(model::PhyllotaxisModelAD, s::Int; τ::Real = model.torsion.τ[s]) = torsion_corrected(model.data[s], model.torsion.x[s], τ)

LogDensityProblems.dimension(model::PhyllotaxisModelAD) = model.k + 3 + model.outlier
LogDensityProblems.capabilities(::Type{<:PhyllotaxisModelAD}) = LogDensityProblems.LogDensityOrder{0}()

# (log μκ, σκ, α, π_o or nothing) from a parameter vector
function _nuisance(model::PhyllotaxisModelAD, θ)
    j = model.k
    return (θ[j + 1], θ[j + 2], θ[j + 3], (model.outlier ? θ[j + 4] : nothing))
end
_nuisance(model::PhyllotaxisModelAD, θ::NamedTuple) = (θ.logμκ, θ.σκ, θ.α, (model.outlier ? θ.outlier : nothing))

"""
    full_eta(model, θ)

The 2k + 1 log-propensities η[d + k + 1] = η_d (η_0 = 0) from a parameter vector: η_{±d} = θ[d].
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

# priors of the nuisance parameters, in the order of θ: log μκ, σκ, α (and π_o with outliers)
function _nuisance_dists(model::PhyllotaxisModelAD)
    (; mκ, sκ, νκ, τκ, aα, bα, aπ, bπ) = model.prior
    d = (Normal(mκ, sκ), _halft(νκ, τκ), Beta(aα, bα))
    return (model.outlier ? (d..., Beta(aπ, bπ)) : d)
end

"prior mean in the constrained space"
function prior_mean(model::PhyllotaxisModelAD)
    (; νκ, τκ) = model.prior
    dists = _nuisance_dists(model)
    return vcat(mean(_eta_dist(model)), mean(dists[1]), _halft_mean(νκ, τκ), [mean(d) for d in dists[3:end]])
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

"log prior density of the constrained parameters"
function _logprior(model::PhyllotaxisModelAD, θ::AbstractVector)
    k = model.k
    d = _nuisance_dists(model)
    lp = logpdf(_eta_dist(model), view(θ, 1:k)) + logpdf(d[1], θ[k + 1]) + logpdf(d[2], θ[k + 2]) + logpdf(d[3], θ[k + 3])
    return (model.outlier ? lp + logpdf(d[4], θ[k + 4]) : lp)
end

"""
    logdensity(model, θ)

Log posterior in the constrained space. Per plant: log ∫ N(log κ) ½ Σ_o Σ_ψ p(ψ) (1/2π) ∫ dϕ perm(A) / (perm(W) (2π I₀(κ))ⁿ),
A_{i, i+d} = exp(η_d + κ cos(y_i − ϕ − o (i + d − 1) ψ)); rectangle rule in ϕ, Gauss–Hermite in log κ. The weights are
written as η_d + κ (cos − 1) and n κ is added back (no overflow in linear mode).
"""
function LogDensityProblems.logdensity(model::PhyllotaxisModelAD, θ::AbstractVector{T}) where {T}
    data = model.data
    logμκ, σκ, α, πo = _nuisance(model, θ)
    logdens = _logprior(model, θ)
    isfinite(logdens) || return logdens

    η = full_eta(model, θ)
    grid = stem_grid(model)
    # plants are split in chunks, each processed by one task with its own workspace
    tasks = map(_chunks(length(data))) do idxs
        Threads.@spawn begin
            local fws = _workspace(model, T, maximum(length, view(data, idxs)))
            sum(s -> stem_loglik!(fws, grid, corrected_angles(model, s), η, logμκ, σκ, α, false; outlier = πo), idxs)
        end
    end
    for t in tasks
        logdens += fetch(t)::T
    end
    return logdens
end

"the nuisance grid of the model, in the form `stem_loglik!` reads"
stem_grid(model::PhyllotaxisModelAD) = (;
    model.k, model.ngrid_ϕ, model.ngrid_κ, ϕ_grid = collect(range(0, 2π; length = model.ngrid_ϕ + 1)[1:model.ngrid_ϕ]),
    model.κ_nodes, model.κ_logweights,
)

# Stacked bijector of the constrained parameters: softplus gaps for η, identity, log, logit (α, and π_o with outliers)
function _bijector(k::Int, outlier::Bool)
    nd = vcat([Normal(0.0, 1.0), _halft(1.0, 1.0), Beta(1.0, 1.0)], (outlier ? [Beta(1.0, 1.0)] : []))
    nr = [(k + i):(k + i) for i in eachindex(nd)]
    return Bijectors.Stacked(Bijectors.bijector.(vcat([LogNormalSoftPlus(zeros(k), ones(k))], nd)), vcat([1:k], nr))
end
Bijectors.bijector(model::PhyllotaxisModelAD) = _bijector(model.k, model.outlier)

"the log density in the unconstrained space: logdensity(binv(u)) + log|J|"
struct TransformedLogDensityProblem{Prob, BInv}
    prob::Prob
    binv::BInv
end
function TransformedLogDensityProblem(prob)
    b = Bijectors.bijector(prob)
    binv = Bijectors.inverse(b)
    return TransformedLogDensityProblem{typeof(prob), typeof(binv)}(prob, binv)
end
TransformedLogDensityProblem(model::PhyllotaxisModelAD) = TransformedLogDensityProblem{typeof(model), typeof(model.binv)}(model, model.binv)
function LogDensityProblems.logdensity(prob_trans::TransformedLogDensityProblem, θ_trans)
    (; prob, binv) = prob_trans
    θ, logabsdetjac = Bijectors.with_logabsdet_jacobian(binv, θ_trans)
    return LogDensityProblems.logdensity(prob, θ) + logabsdetjac
end
function LogDensityProblems.dimension(prob_trans::TransformedLogDensityProblem)
    (; prob, binv) = prob_trans
    b = Bijectors.inverse(binv)
    d = LogDensityProblems.dimension(prob)
    return prod(Bijectors.output_size(b, (d,)))
end
LogDensityProblems.capabilities(::Type{TransformedLogDensityProblem{Prob, BInv}}) where {Prob, BInv} = LogDensityProblems.capabilities(Prob)

"""
    fit_map(model::PhyllotaxisModelAD; u0, options)

MAP by Newton trust region in the unconstrained space, with the analytic gradient and the Hessian from one dual pass over
it (`AngleLikelihood.angle_fit_map`). Returns (; θ, u, H, result): θ = `constrained_parameters(model, u)`, H the Hessian of
the log posterior in u (−inv(H) is the Laplace covariance of u). Also accepts a `TransformedLogDensityProblem`.
"""
function fit_map(model::PhyllotaxisModelAD; u0 = prior_mean_parameters(model), options = MAP_OPTIONS)
    r = angle_fit_map(model; u0, options)
    return (; θ = constrained_parameters(model, r.u), r.u, r.H, r.result)
end
fit_map(prob::TransformedLogDensityProblem; kw...) = fit_map(prob.prob; kw...)

#################################################################################################################
# parameters
#################################################################################################################

"(; η, logμκ, σκ, α, outlier) from the unconstrained u: η the full 2k + 1 log-propensities, outlier = π_o or 0"
function constrained_parameters(model::PhyllotaxisModelAD, u::AbstractVector)
    θ = model.binv(u)
    logμκ, σκ, α, πo = _nuisance(model, θ)
    return (; η = full_eta(model, θ), logμκ, σκ, α, outlier = (πo === nothing ? zero(eltype(θ)) : πo))
end

"""
    unconstrained_parameters(model::PhyllotaxisModelAD; η, logμκ, σκ, α, outlier)

u from named values; η: η_1..η_k, or the full 2k + 1 vector (symmetric, η_0 = 0).
"""
function unconstrained_parameters(model::PhyllotaxisModelAD; η, logμκ, σκ, α, outlier = mean(Beta(model.prior.aπ, model.prior.bπ)))
    k = model.k
    if length(η) == 2k + 1
        (iszero(η[k + 1]) && η[1:k] ≈ reverse(η[(k + 2):end])) ||
            throw(ArgumentError("η: the full vector must be symmetric, η_{−d} = η_d, with η_0 = 0"))
        η = η[(k + 2):end]
    end
    length(η) == k || throw(ArgumentError("η: k = $(k) costs η_1..η_k, or the full vector of 2k + 1"))
    θ = vcat(η, [logμκ, σκ, α], (model.outlier ? [outlier] : Float64[]))
    return Bijectors.inverse(model.binv)(θ)
end
prior_mean_parameters(model::PhyllotaxisModelAD) = Bijectors.inverse(model.binv)(prior_mean(model))

#################################################################################################################
# AngleLikelihood interface (the model and its TransformedLogDensityProblem)
#################################################################################################################

angle_model(model::PhyllotaxisModelAD) = model
function global_logprior(model::PhyllotaxisModelAD, u::AbstractVector)
    θ, lj = Bijectors.with_logabsdet_jacobian(model.binv, u)
    return _logprior(model, θ) + lj
end
stem_etas(model::PhyllotaxisModelAD, u::AbstractVector, s::Int) = ([full_eta(model, model.binv(u))], [0.0])
function angle_nuisance(model::PhyllotaxisModelAD, u::AbstractVector)
    logμκ, σκ, α, πo = _nuisance(model, model.binv(u))
    return (; logμκ, σκ, α, outlier = πo)
end
function stem_value_grad(
        model::PhyllotaxisModelAD, u::AbstractVector, s::Int; τ::Real = model.torsion.τ[s], with_grad::Bool = false,
        fws::StemWorkspace = StemWorkspace(eltype(u), model.k, length(model.data[s]), model.ngrid_ϕ, model.ngrid_κ; logspace = model.logspace)
    )
    k, K = model.k, 2 * model.k + 1
    θ = model.binv(u)
    logμκ, σκ, α, πo = _nuisance(model, θ)
    fill!(fws.grad, 0)
    ℓ = stem_loglik!(fws, stem_grid(model), corrected_angles(model, s; τ), full_eta(model, θ), logμκ, σκ, α, with_grad; outlier = πo)
    with_grad || return ℓ, nothing
    g = fws.grad
    gη = [g[k + 1 + d] + g[k + 1 - d] for d in 1:k]
    gθ = vcat(gη, g[K + 1], g[K + 2], g[K + 3], (model.outlier ? [g[K + 4]] : eltype(g)[]))
    J = ForwardDiff.jacobian(model.binv, u) # θ = binv(u)
    return ℓ, J' * gθ
end

for f in (:angle_model, :stem_grid)
    @eval $f(prob::TransformedLogDensityProblem) = $f(prob.prob)
end
for f in (:global_logprior, :angle_nuisance, :constrained_parameters)
    @eval $f(prob::TransformedLogDensityProblem, u::AbstractVector) = $f(prob.prob, u)
end
stem_etas(prob::TransformedLogDensityProblem, u::AbstractVector, s::Int) = stem_etas(prob.prob, u, s)
stem_value_grad(prob::TransformedLogDensityProblem, u::AbstractVector, s::Int; kw...) = stem_value_grad(prob.prob, u, s; kw...)
prior_mean_parameters(prob::TransformedLogDensityProblem) = prior_mean_parameters(prob.prob)

#################################################################################################################
# stem torsion: per-stem rates
#################################################################################################################

"each stem's rate set to the maximum of its own log-likelihood at the unconstrained u"
update_torsion!(model::PhyllotaxisModelAD, u::AbstractVector; scan = -25.0:0.5:25.0) = angle_update_torsion!(model, u; scan)
"twist quadrature nodes of every stem at the unconstrained u"
torsion_nodes(model::PhyllotaxisModelAD, u::AbstractVector; kw...) = angle_torsion_nodes(model, u; kw...)

"""
    fit_map_torsion!(model::PhyllotaxisModelAD; prior, u0, recenter, nodes_kw, options)

MAP with the twist rates integrated out (ω_s ~ N(m_ω, s_ω²), hyperprior `prior`). Returns (; fit, m_ω, s_ω, τ, τ_mean,
τ_sd, loglik, nodes); fit = (; θ, u, H, result), fit.u = [model parameters; m_ω; log s_ω].
"""
function fit_map_torsion!(
        model::PhyllotaxisModelAD; prior::TorsionPrior = TorsionPrior(), u0 = prior_mean_parameters(model), recenter::Int = 1,
        nodes_kw = (;), options = MAP_OPTIONS
    )
    r = angle_fit_map_torsion!(model; u0, prior, recenter, nodes_kw, options)
    G = length(u0)
    fit = (; θ = constrained_parameters(model, r.u[1:G]), r.u, r.H, r.result)
    return (; fit, r.m_ω, r.s_ω, r.τ, r.τ_mean, r.τ_sd, r.loglik, r.nodes)
end
fit_map_torsion!(prob::TransformedLogDensityProblem; kw...) = fit_map_torsion!(prob.prob; kw...)

#################################################################################################################
# post-processing
#################################################################################################################

"the η block (η_1..η_k) of a parameter vector (or of each column of a matrix of draws)"
log_propensity(model::PhyllotaxisModelAD, θ::AbstractVecOrMat) = view(θ, 1:(model.k), :)

"""
    displacement_probability(model, θ)

P(|d|), d = 0..k, from the local softmax of (η_{−k}, …, η_k), for a parameter vector or each column of a matrix of
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

Prior predictive: η, κ, ψ, ϕ, o from the prior, σ from the permanent model, angles from the von Mises emission.
"""
function sample_prior(model::PhyllotaxisModelAD{T}; n::Int = round(Int, mean(map(length, model.data))), Nsample::Int = 1, rng::AbstractRNG = Random.default_rng()) where {T}
    (; k, logspace) = model
    dlogμκ, dσκ, dα = _nuisance_dists(model)
    ws = PermanentWorkspace(Float64, k, n; logspace = logspace)
    return map(1:Nsample) do _
        toeplitz_weights!(ws, full_eta(model, rand(rng, _eta_dist(model))), n)
        σ = sample_permutation!(rng, ws, n)
        κ = clamp(rand(rng, LogNormal(rand(rng, dlogμκ), rand(rng, dσκ))), 1.0e-3, 1.0e3) # for numerical stability
        ψ = (rand(rng, Bernoulli(rand(rng, dα))) ? Golden_angle : Lucas_angle)
        ϕ = rand(rng, Uniform(0.0, 2π))
        o = (rand(rng, Bernoulli(0.5)) ? 1 : -1)
        y = T[rand(rng, VonMises(mod2pi(ϕ + o * (σ[i] - 1) * ψ), κ)) for i in 1:n]
        (; data = y, permutation = σ, offset_angle = ϕ, orientation = o, canonical_angle = ψ, dispersion = κ)
    end
end

"""
    simulate_stems(η; plants = 100, organs = 50, x = nothing, logμκ = 1.5, σκ = 0.0, α = 0.95, outlier = 0.0,
        torsion_mean = 0.0, torsion_sd = 0.0, rng)

Simulate from the propensity model: the order of each stem is an exact draw from the banded permanent with
log-propensities η (η_1..η_k, symmetric, or the full 2k + 1 vector, recognised by its middle entry η_0 = 0),
κ_s ~ LogNormal(logμκ, σκ) per stem, and
the angles are y_p ~ vM(ϕ + o (σ[p] − 1) ψ + τ_s x_p, κ_s), replaced with probability `outlier` by a uniform angle.
`x`: the twist coordinate of each stem (default: the positions 0, 1, …); twist τ_s ~ Normal(torsion_mean, torsion_sd)
degrees per unit of x. Returns (; angles, orders, torsion, κ, outliers, ψ, o) (outliers: the positions whose angle is
uniform; ψ, o: the spiral of each stem).
"""
function simulate_stems(
        η::AbstractVector{<:Real}; plants::Int = 100, organs::Int = 50, x = nothing, logμκ::Real = 1.5, σκ::Real = 0.0,
        α::Real = 0.95, outlier::Real = 0.0, torsion_mean::Real = 0.0, torsion_sd::Real = 0.0, rng::AbstractRNG = Random.default_rng()
    )
    # a cost η_d (d ≠ 0) is negative, never 0: an η_1..η_k of odd length is not taken for the full vector
    ηf = (isodd(length(η)) && iszero(η[(end + 1) ÷ 2]) ? collect(Float64, η) : vcat(reverse(η), 0.0, η))
    k = length(ηf) ÷ 2
    xs = (x === nothing ? fill(torsion_coordinate(organs), plants) : x)
    (length(xs) == plants && all(length.(xs) .== organs)) || throw(ArgumentError("x: one coordinate per organ of each stem"))
    ws = PermanentWorkspace(Float64, k, organs; logspace = true)
    toeplitz_weights!(ws, ηf, organs)
    out = map(1:plants) do s
        σ = sample_permutation!(rng, ws, organs)
        (; σ, _simulate_angles(rng, σ, xs[s]; logμκ, σκ, α, outlier, torsion_mean, torsion_sd)...)
    end
    return (;
        angles = [z.y for z in out], orders = [z.σ for z in out], torsion = [z.τ for z in out], κ = [z.κ for z in out],
        outliers = [z.outliers for z in out], ψ = [z.ψ for z in out], o = [z.o for z in out],
    )
end

end # module PropensityModel
