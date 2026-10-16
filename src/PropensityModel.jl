"""
    PropensityModel

Model A: angle model with free log-propensities η (one κ per stem from a log-normal population, softplus prior on η).
Parameters (constrained space)

    symmetric:   [η_1..η_k, log μκ, σκ, α]                       (η_{−d} = η_d)
    asymmetric:  [η_{−k}..η_{−1}, η_1..η_k, log μκ, σκ, α]

The per-stem likelihood is the one of model B (`AngleLikelihood.stem_loglik!`) with η free: the gradient in η is
analytic (∂ℓ/∂η_d = E_post[N_d] − E_η[N_d]), as are those in log μκ, σκ and α. `TransformedLogDensityProblem` maps to
ℝ^dim with the Stacked bijector (softplus gaps for η, identity, log, logit); `fit_map` is Newton trust region with the
analytic gradient (chain rule through the bijector Jacobian) and the Hessian from one dual pass over it. The log density
stays generic in the number type (ForwardDiff works on it).
"""
module PropensityModel

using LinearAlgebra, Random, Statistics
using Distributions, LogExpFunctions, StatsBase, SpecialFunctions, FastGaussQuadrature, AdditionalDistributions, Roots
using ForwardDiff, Optim, NLSolversBase
using LogDensityProblems: LogDensityProblems
using Bijectors: Bijectors
using ..Common: Golden_angle, Lucas_angle, logbesseli, besseliratio, _fullvalue
import ..Common: fit_map
using ..Permanents: PermutationGraph, PermanentWorkspace, toeplitz_weights!, forward!, backward!, sample_permutation!,
                    map_permutation, cycle_lengths
using ..PropensityPriors: LogNormalSoftPlus, ReverseLogNormalSoftPlus, η_prior, marginal_quantiles
using ..Torsion: StemTorsion, TorsionPrior, torsion_coordinate, torsion_corrected, torsion_step
import ..Torsion: corrected_angles, update_torsion!, torsion_nodes, fit_map_torsion
using ..AngleLikelihood: StemWorkspace, stem_loglik!, angle_fit_map, angle_update_torsion!, angle_torsion_nodes, angle_fit_map_torsion
import ..AngleLikelihood: stem_value_grad, global_logprior, angle_model

const DiffResults = ForwardDiff.DiffResults

export PhyllotaxisPrior, PhyllotaxisModelAD, TransformedLogDensityProblem, full_eta, prior_mean, rand_priorparameters,
       prior_CIs, log_propensity, displacement_probability, estimate_displacement_frequencies,
       estimate_permutation_frequencies, sample_prior!, decode_sequence!, propensity_plant_loglik

Base.@kwdef struct PhyllotaxisPrior{N}
    # LogNormal prior on the gaps Δ_d = η_{d−1} − η_d (mean m, variance v of each gap)
    μΔ::NTuple{N, Float64} = η_prior((0.5, 1.25, 2.0, 3.75), (0.1, 0.75, 2.0, 4.0))[1]
    σΔ::NTuple{N, Float64} = η_prior((0.5, 1.25, 2.0, 3.75), (0.1, 0.75, 2.0, 4.0))[2]

    # log(μκ) mean and standard deviation
    mκ::Float64 = 1.5
    sκ::Float64 = 0.5

    # Half-t prior on σκ
    νκ::Float64 = 4.0
    τκ::Float64 = 0.25

    # Beta prior on the Golden vs Lucas sequence probability
    aα::Float64 = 95.0
    bα::Float64 = 5.0
end

struct PhyllotaxisModelAD{T, N}
    data::Vector{Vector{T}} # absolute azimuths (radians), one vector per plant
    k::Int # permutation bandwidth = N
    ngrid_ϕ::Int
    ngrid_κ::Int
    κ_nodes::Vector{Float64} # standard normal Gauss–Hermite nodes
    κ_logweights::Vector{Float64}
    logspace::Bool
    symmetric::Bool
    n_max::Int
    graph::PermutationGraph
    prior::PhyllotaxisPrior{N}
    torsion::StemTorsion # per-stem torsion coordinates and rates (rates 0 until fitted)
end

"""
    PhyllotaxisModelAD(data, prior; ngrid_ϕ = 32, ngrid_κ = 16, logspace = false, symmetric = true, torsion_x = nothing)

k is the length of the prior tuples. Fit coarse grids first, then refine from that solution. `torsion_x`: the torsion
coordinate of each plant (`torsion_coordinate(ℓ)` from its internodes, in mm, or `torsion_coordinate(n)`, per position,
the default). The rates start at 0; `fit_map_torsion(TransformedLogDensityProblem(model))` fits them.
"""
function PhyllotaxisModelAD(data::Vector{Vector{T}}, prior::PhyllotaxisPrior{N}; ngrid_ϕ::Int = 32, ngrid_κ::Int = 16, logspace::Bool = false, symmetric::Bool = true, torsion_x = nothing) where {T, N}
    n_min, n_max = extrema(length.(data))
    @assert N ≤ n_min
    tx = (torsion_x === nothing ? [torsion_coordinate(length(y)) for y in data] : torsion_x)
    @assert length(tx) == length(data) && all(length.(tx) .== length.(data)) "torsion_x: one coordinate per organ of each plant"
    κ_nodes, κ_weights = gausshermite(ngrid_κ; normalize = true)
    return PhyllotaxisModelAD{T, N}(data, N, ngrid_ϕ, ngrid_κ, κ_nodes, log.(κ_weights), logspace, symmetric, n_max, PermutationGraph(N), prior,
        StemTorsion(tx))
end

corrected_angles(model::PhyllotaxisModelAD, s::Int; τ::Real = model.torsion.τ[s]) = torsion_corrected(model.data[s], model.torsion.x[s], τ)

LogDensityProblems.dimension(model::PhyllotaxisModelAD) = (model.symmetric ? model.k + 3 : 2 * model.k + 3)
LogDensityProblems.capabilities(::Type{<:PhyllotaxisModelAD}) = LogDensityProblems.LogDensityOrder{0}()

# (log μκ, σκ, α) from a parameter vector
function _nuisance(model::PhyllotaxisModelAD, θ)
    j = (model.symmetric ? model.k : 2 * model.k)
    return (θ[j + 1], θ[j + 2], θ[j + 3])
end

"""
    full_eta(model, θ)

The 2k + 1 log-propensities η[d + k + 1] = η_d (η_0 = 0) from a parameter vector: η_{±d} = θ[d] (symmetric) or
θ[1:k] = η_{−k..−1}, θ[k+1:2k] = η_{1..k} (asymmetric).
"""
function full_eta(model::PhyllotaxisModelAD, θ::AbstractVector{T}) where {T}
    k = model.k
    η = zeros(T, 2k + 1)
    for d in 1:k
        if model.symmetric
            η[k + 1 + d] = θ[d]
            η[k + 1 - d] = θ[d]
        else
            η[k + 1 + d] = θ[k + d]
            η[k + 1 - d] = θ[k + 1 - d]
        end
    end
    return η
end

mean_halft(ν, τ) = (@assert ν > 1; τ * sqrt(ν / π) * gamma(0.5 * (ν - 1)) / gamma(0.5 * ν))

# priors of (η_1..η_k) and (η_{−k}..η_{−1}). ReverseLogNormalSoftPlus takes its gaps from the outside in (its last gap is
# −η_{−1}), so the gap tuples are reversed: gap j from the centre has the same prior on both sides.
_eta_dists(model) = (LogNormalSoftPlus([model.prior.μΔ...], [model.prior.σΔ...]),
    ReverseLogNormalSoftPlus(reverse([model.prior.μΔ...]), reverse([model.prior.σΔ...])))

"prior mean in the constrained space"
function prior_mean(model::PhyllotaxisModelAD)
    (; k, symmetric, prior) = model
    (; mκ, νκ, τκ, aα, bα) = prior
    dpos, dneg = _eta_dists(model)
    η = (symmetric ? mean(dpos) : vcat(mean(dneg), mean(dpos)))
    return vcat(η, [mκ, mean(HalfTDist(νκ, τκ)), mean(Beta(aα, bα))])
end

"one parameter vector drawn from the prior"
function rand_priorparameters(model::PhyllotaxisModelAD; rng::AbstractRNG = Random.default_rng())
    (; symmetric, prior) = model
    (; mκ, sκ, νκ, τκ, aα, bα) = prior
    dpos, dneg = _eta_dists(model)
    η = (symmetric ? rand(rng, dpos) : vcat(rand(rng, dneg), rand(rng, dpos)))
    return vcat(η, [rand(rng, Normal(mκ, sκ)), rand(rng, HalfTDist(νκ, τκ)), rand(rng, Beta(aα, bα))])
end

"prior (α/2, 1 − α/2) intervals of each parameter, dim × 2"
function prior_CIs(model::PhyllotaxisModelAD; α = 0.05)
    (; symmetric, prior) = model
    (; mκ, sκ, νκ, τκ, aα, bα) = prior
    q1, q2 = 0.5 * α, 1 - 0.5 * α
    dpos, dneg = _eta_dists(model)
    ηq(q) = (symmetric ? marginal_quantiles(dpos, q) : vcat(marginal_quantiles(dneg, q), marginal_quantiles(dpos, q)))
    nq(q) = [quantile(Normal(mκ, sκ), q), quantile(HalfTDist(νκ, τκ), q), quantile(Beta(aα, bα), q)]
    return hcat(vcat(ηq(q1), nq(q1)), vcat(ηq(q2), nq(q2)))
end

"log prior density of the constrained parameters"
function _logprior(model::PhyllotaxisModelAD, θ::AbstractVector)
    (; k, symmetric, prior) = model
    (; mκ, sκ, νκ, τκ, aα, bα) = prior
    logμκ, σκ, α = _nuisance(model, θ)
    dpos, dneg = _eta_dists(model)
    lp = (symmetric ? logpdf(dpos, view(θ, 1:k)) : logpdf(dneg, view(θ, 1:k)) + logpdf(dpos, view(θ, (k + 1):(2k))))
    return lp + logpdf(Normal(mκ, sκ), logμκ) + logpdf(HalfTDist(νκ, τκ), σκ) + logpdf(Beta(aα, bα), α)
end

"""
    logdensity(model, θ)

Log posterior in the constrained space. Per plant: log ∫ N(log κ) ½ Σ_o Σ_ψ p(ψ) (1/2π) ∫ dϕ perm(A) / (perm(W) (2π I₀(κ))ⁿ),
A_{i, i+d} = exp(η_d + κ cos(y_i − ϕ − o (i + d − 1) ψ)); rectangle rule in ϕ, Gauss–Hermite in log κ. The weights are
written as η_d + κ (cos − 1) and n κ is added back (no overflow in linear mode).
"""
function LogDensityProblems.logdensity(model::PhyllotaxisModelAD, θ::AbstractVector{T}) where {T}
    (; data, k, ngrid_ϕ, ngrid_κ, logspace, n_max) = model
    logμκ, σκ, α = _nuisance(model, θ)
    logdens = _logprior(model, θ)
    isfinite(logdens) || return logdens

    η = full_eta(model, θ)
    grid = stem_grid(model)
    # plants are split in chunks, each processed by one task with its own workspace
    chunks = collect(Iterators.partition(eachindex(data), max(1, cld(length(data), 2 * Threads.nthreads()))))
    tasks = map(chunks) do idxs
        Threads.@spawn begin
            local fws = StemWorkspace(T, k, n_max, ngrid_ϕ, ngrid_κ; logspace)
            local acc = zero(T)
            for s in idxs
                acc += stem_loglik!(fws, grid, corrected_angles(model, s), η, logμκ, σκ, α, false)
            end
            acc
        end
    end
    for t in tasks
        logdens += fetch(t)::T
    end
    return logdens
end

"the nuisance grid of the model, in the form `stem_loglik!` reads"
stem_grid(model::PhyllotaxisModelAD) = (; model.k, model.ngrid_ϕ, model.ngrid_κ, ϕ_grid = collect(range(0, 2π; length = model.ngrid_ϕ + 1)[1:model.ngrid_ϕ]),
    model.κ_nodes, model.κ_logweights)

function Bijectors.bijector(model::PhyllotaxisModelAD)
    k = model.k
    if model.symmetric
        return Bijectors.Stacked(
            Bijectors.bijector.([LogNormalSoftPlus(zeros(k), ones(k)), Normal(0.0, 1.0), HalfTDist(1.0, 1.0), Beta(1.0, 1.0)]),
            [1:k, (k + 1):(k + 1), (k + 2):(k + 2), (k + 3):(k + 3)],
        )
    else
        return Bijectors.Stacked(
            Bijectors.bijector.([ReverseLogNormalSoftPlus(zeros(k), ones(k)), LogNormalSoftPlus(zeros(k), ones(k)), Normal(0.0, 1.0), HalfTDist(1.0, 1.0), Beta(1.0, 1.0)]),
            [1:k, (k + 1):(2 * k), (2 * k + 1):(2 * k + 1), (2 * k + 2):(2 * k + 2), (2 * k + 3):(2 * k + 3)],
        )
    end
end

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
    fit_map(prob::TransformedLogDensityProblem; u0, options)

MAP by Newton trust region in the unconstrained space, with the analytic gradient and the Hessian from one dual pass over
it (`AngleLikelihood.angle_fit_map`). Returns (; θ, u, H, result): θ constrained, H the Hessian of the log posterior in u
(−inv(H) is the Laplace covariance of u).
"""
function fit_map(prob::TransformedLogDensityProblem; u0 = Bijectors.bijector(prob.prob)(prior_mean(prob.prob)), options = Optim.Options(g_tol = 1e-5, iterations = 100))
    r = angle_fit_map(prob; u0, options)
    return (; θ = prob.binv(r.u), r.u, r.H, r.result)
end

# AngleLikelihood interface
angle_model(prob::TransformedLogDensityProblem) = prob.prob
function global_logprior(prob::TransformedLogDensityProblem, u::AbstractVector)
    θ, lj = Bijectors.with_logabsdet_jacobian(prob.binv, u)
    return _logprior(prob.prob, θ) + lj
end
function stem_value_grad(prob::TransformedLogDensityProblem, u::AbstractVector, s::Int; τ::Real = prob.prob.torsion.τ[s], with_grad::Bool = false,
        fws::StemWorkspace = StemWorkspace(eltype(u), prob.prob.k, length(prob.prob.data[s]), prob.prob.ngrid_ϕ, prob.prob.ngrid_κ; logspace = prob.prob.logspace))
    m = prob.prob
    k, K = m.k, 2 * m.k + 1
    θ = prob.binv(u)
    logμκ, σκ, α = _nuisance(m, θ)
    fill!(fws.grad, 0)
    ℓ = stem_loglik!(fws, stem_grid(m), corrected_angles(m, s; τ), full_eta(m, θ), logμκ, σκ, α, with_grad)
    with_grad || return ℓ, nothing
    g = fws.grad
    gη = (m.symmetric ? [g[k + 1 + d] + g[k + 1 - d] for d in 1:k] : vcat(g[1:k], g[(k + 2):K]))
    gθ = vcat(gη, g[K + 1], g[K + 2], g[K + 3])
    J = ForwardDiff.jacobian(prob.binv, u) # θ = binv(u)
    return ℓ, J' * gθ
end

#################################################################################################################
# stem torsion: per-stem rates
#################################################################################################################

"""
    propensity_plant_loglik(model, θ, s; τ)

Log-likelihood of plant s alone (prior excluded) at the constrained parameters θ, with twist rate τ (degrees per unit of
x). Generic in the number types of θ and τ (ForwardDiff, nested duals).
"""
function propensity_plant_loglik(model::PhyllotaxisModelAD, θ::AbstractVector, s::Int; τ::Real = model.torsion.τ[s])
    (; k, ngrid_ϕ, ngrid_κ, logspace) = model
    T = promote_type(eltype(θ), typeof(τ))
    logμκ, σκ, α = _nuisance(model, θ)
    y = corrected_angles(model, s; τ)
    fws = StemWorkspace(T, k, length(y), ngrid_ϕ, ngrid_κ; logspace)
    return stem_loglik!(fws, stem_grid(model), y, full_eta(model, θ), logμκ, σκ, α, false)
end

update_torsion!(model::PhyllotaxisModelAD, θ::AbstractVector; scan = -25.0:0.5:25.0) =
    angle_update_torsion!(TransformedLogDensityProblem(model), Bijectors.bijector(model)(map(_fullvalue, θ)); scan)
torsion_nodes(model::PhyllotaxisModelAD, θ::AbstractVector; kw...) =
    angle_torsion_nodes(TransformedLogDensityProblem(model), Bijectors.bijector(model)(map(_fullvalue, θ)); kw...)

function fit_map_torsion(prob::TransformedLogDensityProblem; prior::TorsionPrior = TorsionPrior(),
        u0 = Bijectors.bijector(prob.prob)(prior_mean(prob.prob)), recenter::Int = 1, nodes_kw = (;),
        options = Optim.Options(g_tol = 1e-5, iterations = 100))
    r = angle_fit_map_torsion(prob; u0, prior, recenter, nodes_kw, options)
    G = length(u0)
    fit = (; θ = prob.binv(r.u[1:G]), r.u, r.H, r.result)
    return (; fit, r.m_ω, r.s_ω, r.τ, r.τ_mean, r.τ_sd, r.loglik, r.nodes)
end

#################################################################################################################
# post-processing
#################################################################################################################

"the η block of a parameter vector (or of each column of a matrix of draws)"
log_propensity(model::PhyllotaxisModelAD, θ::AbstractVecOrMat) = view(θ, 1:(model.symmetric ? model.k : 2 * model.k), :)

"local softmax of (0, η): the instantaneous propensities, not the realised displacement frequencies"
function displacement_probability(model::PhyllotaxisModelAD, θ::AbstractVecOrMat)
    η = log_propensity(model, θ)
    prop = vcat(zeros(1, size(η, 2)), η)
    softmax!(prop; dims = 1)
    sort!(prop; dims = 2)
    return prop
end

"""
    estimate_displacement_frequencies(model, q = nothing; N, n, α, rng)

Realised P(|d|), d = 0..k, in the bulk of a stem of n organs, averaged over θ ~ q (any object with `rand`, e.g. the Laplace
MvNormal mapped back) or over the prior: exact edge marginals by forward–backward, averaged over positions. Returns
(mean, intervals (k+1) × 2).
"""
function estimate_displacement_frequencies(model::PhyllotaxisModelAD, q = nothing; N::Int = 10_000, n::Int = 60, α = 0.05, rng::AbstractRNG = Random.default_rng())
    (; k, logspace, symmetric) = model
    ws = PermanentWorkspace(Float64, k, n; logspace = logspace)
    p = zeros(k + 1, N)
    diagmean(rho, d) = mean(rho[i, d + k + 1] for i in max(1, 1 - d):min(n, n - d))
    for s in 1:N
        θ = (q === nothing ? rand_priorparameters(model; rng) : rand(rng, q))
        toeplitz_weights!(ws, full_eta(model, θ), n)
        forward!(ws, n)
        rho = backward!(ws, n)
        for d in 0:k
            p[d + 1, s] = (symmetric ? diagmean(rho, d) : 0.5 * diagmean(rho, d) + 0.5 * diagmean(rho, -d))
        end
    end
    return vec(mean(p; dims = 2)), mapslices(x -> quantile(x, [α / 2, 1 - α / 2]), p; dims = 2)
end

"""
    estimate_permutation_frequencies(model, q = nothing; N, n, α, rng)

Fraction of organs in cycles of length 1 (fixed points), 2 (swaps), 3, … of sampled permutations, θ ~ q or the prior.
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
            p[len, s] = c
        end
        p[:, s] ./= sum(view(p, :, s))
    end
    return vec(mean(p; dims = 2)), mapslices(x -> quantile(x, [α / 2, 1 - α / 2]), p; dims = 2)
end

"""
    sample_prior!(model; n, Nsample, rng)

Prior predictive: η, κ, ψ, ϕ, o from the prior, σ from the permanent model, angles from the von Mises emission.
"""
function sample_prior!(model::PhyllotaxisModelAD{T}; n::Int = round(Int, mean(map(length, model.data))), Nsample::Int = 1, rng::AbstractRNG = Random.default_rng()) where {T}
    (; k, logspace, symmetric, prior) = model
    (; mκ, sκ, νκ, τκ, aα, bα) = prior
    ws = PermanentWorkspace(Float64, k, n; logspace = logspace)
    dpos, dneg = _eta_dists(model)
    return map(1:Nsample) do _
        ηs = (symmetric ? rand(rng, dpos) : vcat(rand(rng, dneg), rand(rng, dpos)))
        toeplitz_weights!(ws, full_eta(model, ηs), n)
        σ = sample_permutation!(rng, ws, n)
        κ = clamp(rand(rng, LogNormal(rand(rng, Normal(mκ, sκ)), rand(rng, HalfTDist(νκ, τκ)))), 1e-3, 1e3) # for numerical stability
        ψ = (rand(rng, Bernoulli(rand(rng, Beta(aα, bα)))) ? Golden_angle : Lucas_angle)
        ϕ = rand(rng, Uniform(0.0, 2π))
        o = (rand(rng, Bernoulli(0.5)) ? 1 : -1)
        y = T[rand(rng, VonMises(mod2pi(ϕ + o * (σ[i] - 1) * ψ), κ)) for i in 1:n]
        (; data = y, permutation = σ, offset_angle = ϕ, orientation = o, canonical_angle = ψ, dispersion = κ)
    end
end

"""
    decode_sequence!(model, θ, y; niter = 1000)

Per-plant decoding at θ: for each (o, ψ), coordinate ascent on σ (Viterbi), ϕ (circular mean) and log κ (Newton on the
MAP equation with the log-normal prior), from the identity, the circular mean and the genotype mean κ; every sweep
updates all three before convergence (σ unchanged) is tested. The (o, ψ) with the best joint score (prior of κ included)
is returned as (; σ, o, ψ, ϕ, κ).
"""
function decode_sequence!(model::PhyllotaxisModelAD, θ, y; niter::Int = 1_000)
    (; k, graph) = model
    n = length(y)
    η = full_eta(model, θ)
    logμκ, σκ, α = _nuisance(model, θ)

    best_logp = -Inf
    best = (; σ = collect(1:n), o = 1, ψ = Golden_angle, ϕ = 0.0, κ = 1.0)
    L = fill(-Inf, n, 2 * k + 1) # log-weight matrix
    σ = zeros(Int, n)
    for o in (1, -1), ψ in (Golden_angle, Lucas_angle)
        σ .= 1:n
        κ = mean(LogNormal(logμκ, σκ))
        ϕ = angle(sum(cis(y[i] - o * mod2pi((i - 1) * ψ)) for i in 1:n))
        for it in 1:niter
            # σ given (ϕ, κ); κ is subtracted inside the cosine term for stability (does not change the argmax)
            for d in -k:k, i in max(1, 1 - d):min(n, n - d)
                L[i, d + k + 1] = η[d + k + 1] + κ * (cos(y[i] - ϕ - o * (i + d - 1) * ψ) - 1)
            end
            σ_new = map_permutation(graph, L, n)
            converged = (it > 1 && σ_new == σ)
            σ .= σ_new
            # ϕ given σ: circular mean
            ϕ = angle(sum(cis(y[i] - o * (σ[i] - 1) * ψ) for i in 1:n))
            # log κ given (σ, ϕ): Newton on n κ (C − A(κ)) − (log κ − log μκ) / σκ² = 0
            C = mean(cos(y[i] - ϕ - o * (σ[i] - 1) * ψ) for i in 1:n)
            f(lκ) = (κ_ = exp(lκ); n * κ_ * (C - besseliratio(κ_)) - (lκ - logμκ) / (σκ ^ 2))
            function df(lκ)
                κ_ = exp(lκ)
                A = besseliratio(κ_)
                return n * κ_ * (C - A) - n * (κ_ ^ 2) * (1 - A ^ 2 - A / κ_) - 1 / (σκ ^ 2)
            end
            κ = exp(find_zero((f, df), logμκ, Roots.Newton()))
            converged && break
        end
        logp = (ψ == Golden_angle ? log(α) : log1p(-α)) + logpdf(Normal(logμκ, σκ), log(κ))
        for i in 1:n
            logp += η[σ[i] - i + k + 1] + κ * cos(y[i] - ϕ - o * (σ[i] - 1) * ψ) - logbesseli(0, κ)
        end
        if logp > best_logp
            best_logp = logp
            best = (; σ = copy(σ), o, ψ, ϕ, κ)
        end
    end
    return best
end

end # module PropensityModel
