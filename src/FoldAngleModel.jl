"""
    FoldAngleModel

Angle model whose log-propensities come from the passage law of a commitment switch near its fold (model B):
η = η(shape, log γ), read from a `PropensityTable`. Parameters u (unconstrained) ↦ θ = (shape, log γ, log μκ, σκ, α).

**Tempo option.** When the plastochron changes along the stem, `FoldModel(...; tempo = r)` takes the relative
plastochron r_sp = T_P(p) / T̄_P of every organ and lets the readiness spacing follow it,

    log γ_sp = log γ + β log r_sp,      η_sp = η(shape, log γ_sp)    (one row of log-propensities per organ),

which adds one parameter β (u[6]): β = 1, a timing noise independent of the tempo; β = 0, a timing noise proportional
to the plastochron (γ conserved: the model without `tempo`). γ is then the precision at the mean tempo.

When the flower rank of the first measured organ is uncertain (organs missed at the base of a reconstruction),
`FoldModel(...; plastochron = TP, offsets = 0:2)` takes the plastochron by flower rank instead, and sums each stem's
likelihood over its candidate offsets (a discrete random effect, like the twist rate).

The likelihood of a stem sums the banded permanent over the nuisance grid (ϕ, o, ψ) × κ nodes (`AngleLikelihood`). The
gradient is analytic (expected displacement counts per row and expected cosines from forward–backward, chain rule
through the bicubic η table); the Hessian is one ForwardDiff pass over that gradient; the MAP is Newton trust region
(`fit_map`). Stem torsion: one rate per stem, ω_s ~ N(m_ω, s_ω²), integrated out by adaptive quadrature
(`fit_map_torsion`).
"""
module FoldAngleModel

using Random, Statistics
using Distributions, LogExpFunctions, FastGaussQuadrature, AdditionalDistributions
using ForwardDiff, Optim
using LogDensityProblems: LogDensityProblems
using ..Common: Golden_angle, Lucas_angle, logbesseli, logpdf_halft, _scaled_logistic
import ..Common: fit_map
using ..Permanents: PermanentWorkspace, forward!, map_permutation
using ..PassageTime: PassageTable, passage_sampler
using ..OrderStatistics: PropensityTable, propensity!
using ..Torsion: StemTorsion, TorsionPrior, torsion_coordinate, torsion_corrected
using ..AngleLikelihood: StemWorkspace, stem_loglik!, angle_vgh, angle_fit_map, angle_update_torsion!, angle_torsion_nodes,
    angle_fit_map_torsion
import ..AngleLikelihood: stem_value_grad, global_logprior, angle_model
import ..Torsion: corrected_angles, update_torsion!, torsion_nodes, fit_map_torsion

export FoldPrior, FoldModel, has_tempo, constrained_parameters, unconstrained_parameters, prior_mean_parameters,
       fold_logprior, fold_loglikelihood, logdensity_and_gradient!, fold_eta, offset_posterior, decode_orders, simulate_stems,
       with_logspace,
       fold_plant_loglik

Base.@kwdef struct FoldPrior
    # shape: plateau margin μ (noise units) or ramp log ν; Normal prior truncated to the table range
    mμ::Float64 = 2.0
    sμ::Float64 = 2.0

    # readiness spacing in passage sd units (at the mean tempo): log γ ~ Normal
    mγ::Float64 = log(2.0)
    sγ::Float64 = 0.5

    # log(μκ) mean and standard deviation
    mκ::Float64 = 1.5
    sκ::Float64 = 0.5

    # Half-t prior on σκ
    νκ::Float64 = 4.0
    τκ::Float64 = 0.25

    # Beta prior on the golden vs Lucas spiral probability
    aα::Float64 = 95.0
    bα::Float64 = 5.0

    # tempo exponent β ~ Normal (used only with a tempo profile): neutral between β = 0 and β = 1
    mβ::Float64 = 0.5
    sβ::Float64 = 1.0
end

struct FoldModel{T}
    data::Vector{Vector{T}}
    k::Int
    ngrid_ϕ::Int
    ngrid_κ::Int
    ϕ_grid::Vector{Float64}
    κ_nodes::Vector{Float64} # standard normal nodes
    κ_logweights::Vector{Float64}
    ptab::PropensityTable
    prior::FoldPrior
    μ_bounds::Tuple{Float64, Float64}
    logγ_bounds::Tuple{Float64, Float64}
    logspace::Bool
    torsion::StemTorsion # per-stem torsion coordinates and rates (rates 0 until fitted)
    logr::Union{Nothing, Vector{Vector{Vector{Float64}}}} # per stem, per candidate offset: log r of each organ (or nothing)
    logw::Union{Nothing, Vector{Vector{Float64}}} # per stem: log prior weight of each candidate offset
    offsets::Union{Nothing, Vector{Vector{Int}}} # per stem: the candidate offsets (nothing with `tempo` or no tempo)
end

"""
    FoldModel(data, ptab, prior = FoldPrior(); ngrid_ϕ = 32, ngrid_κ = 16, logspace = false, torsion_x = nothing, tempo = nothing)

`data`: absolute azimuths (radians), one vector per stem, bottom to top. k = ptab.k. `torsion_x`: the torsion coordinate
of each stem (`torsion_coordinate(ℓ)` from its internodes, in mm, or `torsion_coordinate(n)`, per position, the
default); the rates start at 0 and `fit_map_torsion` fits them. `tempo`: the relative plastochron of every organ of
every stem (positive, one vector per stem, e.g. from `relative_plastochron`); adds the exponent β.

`plastochron` (instead of `tempo`): the plastochron by flower rank, one vector for all stems (`plastochron_by_rank`)
or one per stem (`plant_plastochron`), with `offsets`, the number
of organs below the first measured one that were counted as flowers: one integer for all stems, one set of candidates
for all stems (e.g. `0:2`), or one set per stem. With several candidates the stem likelihood is summed over them, with
prior weights `offset_prior` (uniform by default); `offset_posterior` gives each stem's posterior.
"""
function FoldModel(data::Vector{Vector{T}}, ptab::PropensityTable, prior::FoldPrior = FoldPrior(); ngrid_ϕ::Int = 32, ngrid_κ::Int = 16,
        logspace::Bool = false, torsion_x = nothing, tempo = nothing, plastochron = nothing, offsets = 0, offset_prior = nothing) where {T}
    k = ptab.k
    tx = (torsion_x === nothing ? [torsion_coordinate(length(y)) for y in data] : torsion_x)
    @assert length(tx) == length(data) && all(length.(tx) .== length.(data)) "torsion_x: one coordinate per organ of each stem"
    if tempo !== nothing
        @assert length(tempo) == length(data) && all(length.(tempo) .== length.(data)) "tempo: one value per organ of each stem"
        @assert all(r -> all(>(0), r), tempo) "tempo: relative plastochrons must be positive"
    end
    n_min = minimum(length, data)
    @assert k < n_min
    κ_nodes, κ_weights = gausshermite(ngrid_κ; normalize = true)
    ϕ_grid = collect(range(0, 2π; length = ngrid_ϕ + 1)[1:ngrid_ϕ])
    @assert tempo === nothing || plastochron === nothing "give either `tempo` (known ranks) or `plastochron` (with `offsets`)"
    logr, logw, offs = nothing, nothing, nothing
    if tempo !== nothing
        logr = [[log.(Float64.(r))] for r in tempo]
        logw = [[0.0] for _ in tempo]
    elseif plastochron !== nothing
        logr, logw, offs = _offset_profiles(plastochron, length.(data), offsets, offset_prior)
    end
    return FoldModel{T}(data, k, ngrid_ϕ, ngrid_κ, ϕ_grid, κ_nodes, log.(κ_weights), ptab, prior, extrema(ptab.μ_grid),
        extrema(ptab.logγ_grid), logspace, StemTorsion(tx), logr, logw, offs)
end

# candidate tempo profiles: organ i of stem s with offset o has flower rank i + o and r = T_P(i + o) / T̄, T̄ the mean over
# the organs (prior-weighted over the candidates), so that r averages to 1 and γ is the precision at the mean tempo
function _offset_profiles(TP::AbstractVector, ns::AbstractVector{Int}, offsets, offset_prior)
    S = length(ns)
    TPs = (eltype(TP) <: Real ? fill(TP, S) : TP) # one curve for all stems, or one per stem
    @assert length(TPs) == S "plastochron: one vector by flower rank, or one per stem"
    offs = (offsets isa Integer ? [[Int(offsets)] for _ in 1:S] :
        eltype(offsets) <: Integer ? [collect(Int, offsets) for _ in 1:S] : [collect(Int, o) for o in offsets])
    @assert length(offs) == S "offsets: one integer, one candidate set for all stems, or one candidate set per stem"
    @assert all(o -> all(≥(0), o), offs) "offsets are numbers of organs missed below the first measured one (≥ 0)"
    w = map(offs) do o
        v = (offset_prior === nothing ? ones(length(o)) : Float64.(collect(offset_prior)))
        @assert length(v) == length(o) && all(≥(0), v) "offset_prior: one non-negative weight per candidate offset"
        v ./ sum(v)
    end
    T(s, o) = [Float64(TPs[s][clamp(i + o, 1, length(TPs[s]))]) for i in 1:ns[s]]
    T̄ = sum(sum(w[s][c] * sum(T(s, o)) for (c, o) in enumerate(offs[s])) for s in 1:S) / sum(ns)
    logr = [[log.(T(s, o) ./ T̄) for o in offs[s]] for s in 1:S]
    return logr, [log.(v) for v in w], offs
end

"whether the model has a tempo profile (and the exponent β)"
has_tempo(model::FoldModel) = model.logr !== nothing

corrected_angles(model::FoldModel, s::Int; τ::Real = model.torsion.τ[s]) = torsion_corrected(model.data[s], model.torsion.x[s], τ)

LogDensityProblems.dimension(model::FoldModel) = 5 + has_tempo(model)
LogDensityProblems.capabilities(::Type{<:FoldModel}) = LogDensityProblems.LogDensityOrder{2}()

"the same model evaluated in log-space"
with_logspace(m::FoldModel{T}) where {T} = FoldModel{T}(m.data, m.k, m.ngrid_ϕ, m.ngrid_κ, m.ϕ_grid, m.κ_nodes, m.κ_logweights, m.ptab,
    m.prior, m.μ_bounds, m.logγ_bounds, true, m.torsion, m.logr, m.logw, m.offsets)

#################################################################################################################
# parameters and prior
#################################################################################################################

"""
    constrained_parameters(model, u)

(; μ, logγ, logμκ, σκ, α, β) from the unconstrained u: μ (the shape) and log γ through a scaled logistic onto the table
ranges, σκ = exp, α = logistic, β = u[6] (0 without tempo).
"""
function constrained_parameters(model::FoldModel, u::AbstractVector)
    return (; μ = _scaled_logistic(u[1], model.μ_bounds), logγ = _scaled_logistic(u[2], model.logγ_bounds),
        logμκ = u[3], σκ = exp(u[4]), α = logistic(u[5]), β = (has_tempo(model) ? u[6] : zero(eltype(u))))
end
function unconstrained_parameters(model::FoldModel; μ, logγ, logμκ, σκ, α, β = model.prior.mβ)
    _inv(x, (lo, hi)) = logit((x - lo) / (hi - lo))
    u = [_inv(μ, model.μ_bounds), _inv(logγ, model.logγ_bounds), logμκ, log(σκ), logit(α)]
    return (has_tempo(model) ? vcat(u, β) : u)
end
function prior_mean_parameters(model::FoldModel)
    (; mμ, mγ, mκ, νκ, τκ, aα, bα, mβ) = model.prior
    μ = clamp(mμ, model.μ_bounds[1] + 0.1, model.μ_bounds[2] - 0.1)
    logγ = clamp(mγ, model.logγ_bounds[1] + 0.1, model.logγ_bounds[2] - 0.1)
    return unconstrained_parameters(model; μ, logγ, logμκ = mκ, σκ = mean(HalfTDist(νκ, τκ)), α = mean(Beta(aα, bα)), β = mβ)
end

"log prior on the constrained parameters plus the log-Jacobian of the transform u ↦ θ (generic in the number type)"
function fold_logprior(model::FoldModel, u::AbstractVector)
    (; mμ, sμ, mγ, sγ, mκ, sκ, νκ, τκ, aα, bα, mβ, sβ) = model.prior
    (; μ, logγ, logμκ, σκ, α, β) = constrained_parameters(model, u)
    sμ_u, sγ_u = logistic(u[1]), logistic(u[2])
    lp = logpdf(Normal(mμ, sμ), μ) + logpdf(Normal(mγ, sγ), logγ) + logpdf(Normal(mκ, sκ), logμκ) + logpdf_halft(σκ, νκ, τκ) + logpdf(Beta(aα, bα), α)
    lp += log((model.μ_bounds[2] - model.μ_bounds[1]) * sμ_u * (1 - sμ_u)) + log((model.logγ_bounds[2] - model.logγ_bounds[1]) * sγ_u * (1 - sγ_u))
    lp += u[4] + log(α) + log1p(-α)
    has_tempo(model) && (lp += -0.5 * ((β - mβ) / sβ)^2 - log(sβ * sqrt(2π)))
    return lp
end

#################################################################################################################
# log-propensities and the per-stem likelihood
#################################################################################################################

"""
    fold_eta(model, θ, s; with_grad = false, c = 1)

Log-propensities of stem s (candidate offset c) at the constrained parameters θ, with their derivatives in
(shape, log γ) if `with_grad`:
without tempo, the vector η (2k + 1) and ∂η ((2k + 1) × 2); with tempo, the n × (2k + 1) matrix H (row p at
log γ_p = log γ + β log r_p, clamped to the table, where its derivative is 0) and the matrices ∂H/∂shape, ∂H/∂log γ_p.
"""
function fold_eta(model::FoldModel, θ, s::Int; with_grad::Bool = false, c::Int = 1)
    (; ptab, logγ_bounds) = model
    (; μ, logγ, β) = θ
    T = promote_type(typeof(μ), typeof(logγ), typeof(β))
    K = 2 * ptab.k + 1
    η = zeros(T, K)
    ∂η = (with_grad ? zeros(T, K, 2) : nothing)
    if !has_tempo(model)
        propensity!(η, ∂η, ptab, μ, logγ)
        return η, ∂η
    end
    lr = model.logr[s][c]
    n = length(lr)
    H = zeros(T, n, K)
    dμ = (with_grad ? zeros(T, n, K) : nothing)
    dγ = (with_grad ? zeros(T, n, K) : nothing)
    lo, hi = logγ_bounds
    for i in 1:n
        lg = logγ + β * lr[i]
        propensity!(η, ∂η, ptab, μ, clamp(lg, lo, hi))
        H[i, :] .= η
        if with_grad
            dμ[i, :] .= view(∂η, :, 1)
            (lo < lg < hi) && (dγ[i, :] .= view(∂η, :, 2))
        end
    end
    return H, (dμ, dγ)
end

# ∂ℓ/∂(shape, log γ, β) from the gradient of the stem likelihood in its log-propensities
function _eta_chain(model::FoldModel, fws, ∂, s, c, ::Type{T}) where {T}
    K = 2 * model.k + 1
    if !has_tempo(model)
        g = fws.grad
        return (sum(g[d] * ∂[d, 1] for d in 1:K), sum(g[d] * ∂[d, 2] for d in 1:K), zero(T))
    end
    dμ, dγ = ∂
    G = fws.gradH
    lr = model.logr[s][c]
    gμ = gγ = gβ = zero(T)
    for i in eachindex(lr)
        gi = zero(T)
        for d in 1:K
            gμ += G[i, d] * dμ[i, d]
            gi += G[i, d] * dγ[i, d]
        end
        gγ += gi
        gβ += gi * lr[i]
    end
    return (gμ, gγ, gβ)
end

_workspace(model::FoldModel, ::Type{T}, n::Int) where {T} = StemWorkspace(T, model.k, n, model.ngrid_ϕ, model.ngrid_κ; logspace = model.logspace)

# AngleLikelihood interface
angle_model(model::FoldModel) = model
global_logprior(model::FoldModel, u::AbstractVector) = fold_logprior(model, u)
# log-likelihood of stem s for candidate offset c, and its gradient in u
function _stem_candidate(model::FoldModel, u::AbstractVector, s::Int, c::Int, y, with_grad::Bool, fws::StemWorkspace)
    θ = constrained_parameters(model, u)
    (; σκ, α) = θ
    η, ∂ = fold_eta(model, θ, s; with_grad, c)
    fill!(fws.grad, 0)
    ℓ = stem_loglik!(fws, model, y, η, θ.logμκ, σκ, α, with_grad)
    with_grad || return ℓ, nothing
    K = 2 * model.k + 1
    g = fws.grad
    gμ, gγ, gβ = _eta_chain(model, fws, ∂, s, c, eltype(u))
    sμ_u, sγ_u = logistic(u[1]), logistic(u[2])
    grad = [gμ * (model.μ_bounds[2] - model.μ_bounds[1]) * sμ_u * (1 - sμ_u), gγ * (model.logγ_bounds[2] - model.logγ_bounds[1]) * sγ_u * (1 - sγ_u),
        g[K + 1], g[K + 2] * σκ, g[K + 3] * α * (1 - α)]
    return ℓ, (has_tempo(model) ? push!(grad, gβ) : grad)
end

_ncandidates(model::FoldModel, s::Int) = (has_tempo(model) ? length(model.logr[s]) : 1)

# stem likelihood: with several candidate offsets, log Σ_c w_c L_c and the posterior-weighted mean of the gradients
function stem_value_grad(model::FoldModel, u::AbstractVector, s::Int; τ::Real = model.torsion.τ[s], with_grad::Bool = false,
        fws::StemWorkspace = _workspace(model, eltype(u), length(model.data[s])))
    y = corrected_angles(model, s; τ)
    C = _ncandidates(model, s)
    C == 1 && return _stem_candidate(model, u, s, 1, y, with_grad, fws)
    vals = [_stem_candidate(model, u, s, c, y, with_grad, fws) for c in 1:C]
    a = [model.logw[s][c] + first(vals[c]) for c in 1:C]
    L = logsumexp(a)
    with_grad || return L, nothing
    π = exp.(a .- L)
    return L, sum(π[c] .* vals[c][2] for c in 1:C)
end

"""
    offset_posterior(model, u)

Posterior probability of each candidate offset of each stem at the parameters u (and the current twist rates): one
vector per stem, in the order of `model.offsets[s]`.
"""
function offset_posterior(model::FoldModel, u::AbstractVector)
    uf = map(Float64, u[1:LogDensityProblems.dimension(model)])
    out = Vector{Vector{Float64}}(undef, length(model.data))
    Threads.@threads for s in eachindex(model.data)
        fws = _workspace(model, Float64, length(model.data[s]))
        y = corrected_angles(model, s)
        C = _ncandidates(model, s)
        a = [(C == 1 ? 0.0 : model.logw[s][c]) + first(_stem_candidate(model, uf, s, c, y, false, fws)) for c in 1:C]
        out[s] = exp.(a .- logsumexp(a))
    end
    return out
end

"""
    logdensity_and_gradient!(∇, model, u)

Log posterior density (unconstrained parametrisation, Jacobian included) and its analytic gradient, written into ∇ if
not `nothing`.
"""
function logdensity_and_gradient!(∇::Union{AbstractVector, Nothing}, model::FoldModel, u::AbstractVector)
    l, g, _ = angle_vgh(model, u, ∇ === nothing ? 0 : 1)
    (∇ !== nothing) && (∇ .= g)
    return l
end

"log-likelihood alone (no prior), e.g. to compare models at their MAP"
fold_loglikelihood(model::FoldModel, u::AbstractVector) = logdensity_and_gradient!(nothing, model, u) - fold_logprior(model, u)

LogDensityProblems.logdensity(model::FoldModel, u) = logdensity_and_gradient!(nothing, model, u)
function LogDensityProblems.logdensity_and_gradient(model::FoldModel, u)
    ∇ = similar(u, float(eltype(u)))
    return (logdensity_and_gradient!(∇, model, u), ∇)
end

"""
    logdensity_gradient_and_hessian(model, u)

Value, analytic gradient and Hessian (one dual pass over the gradient). The scaled linear pass can underflow with dual
numbers at extreme κ: such a point is redone in log-space.
"""
function LogDensityProblems.logdensity_gradient_and_hessian(model::FoldModel, u::AbstractVector)
    l, g, H = angle_vgh(model, u, 2)
    if !model.logspace && !(isfinite(l) && all(isfinite, g) && all(isfinite, H))
        return angle_vgh(with_logspace(model), u, 2)
    end
    return (l, g, H)
end

"""
    fit_map(model::FoldModel; u0, options)

MAP by Newton's method with trust region (one `only_fgh!` callback: value + analytic gradient, and the dual-pass
Hessian only when asked). Returns (; θ, u, H, result), H the Hessian of the log posterior at the last Newton step
(−inv(H) is the Laplace covariance of u). g_tol = 1e-5 is the noise floor of the interpolated tables.
"""
function fit_map(model::FoldModel; u0 = prior_mean_parameters(model), options = Optim.Options(g_tol = 1e-5, iterations = 100))
    r = angle_fit_map(model; u0, options)
    return (; θ = constrained_parameters(model, r.u), r.u, r.H, r.result)
end

#################################################################################################################
# stem torsion: per-stem rates
#################################################################################################################

"log-likelihood of stem s alone (prior excluded) at the parameters u, with twist rate τ (degrees per unit of x)"
fold_plant_loglik(model::FoldModel, u::AbstractVector, s::Int; τ::Real = model.torsion.τ[s]) =
    first(stem_value_grad(model, map(Float64, u), s; τ))

update_torsion!(model::FoldModel, u::AbstractVector; scan = -25.0:0.5:25.0) = angle_update_torsion!(model, u; scan)
torsion_nodes(model::FoldModel, u::AbstractVector; kw...) = angle_torsion_nodes(model, u; kw...)

"""
    fit_map_torsion(model::FoldModel; prior, u0, recenter, nodes_kw, options)

MAP with the twist rates integrated out (ω_s ~ N(m_ω, s_ω²), hyperprior `prior`). Returns (; fit, m_ω, s_ω, τ, τ_mean,
τ_sd, loglik, nodes); fit.u = [model parameters; m_ω; log s_ω].
"""
function fit_map_torsion(model::FoldModel; prior::TorsionPrior = TorsionPrior(), u0 = prior_mean_parameters(model), recenter::Int = 1,
        nodes_kw = (;), options = Optim.Options(g_tol = 1e-5, iterations = 100))
    r = angle_fit_map_torsion(model; u0, prior, recenter, nodes_kw, options)
    G = LogDensityProblems.dimension(model)
    fit = (; θ = constrained_parameters(model, r.u[1:G]), r.u, r.H, r.result)
    return (; fit, r.m_ω, r.s_ω, r.τ, r.τ_mean, r.τ_sd, r.loglik, r.nodes)
end

#################################################################################################################
# decoding and simulation
#################################################################################################################

"""
    decode_orders(model, u)

Most likely organ order of each stem at the parameters u: the nuisance combination (ϕ, o, ψ, κ node) with the largest
posterior weight (and, with candidate offsets, the most probable offset) is selected, then the MAP permutation is found
by Viterbi on the log edge weights
η_pd + κ (cos(y_p − ϕ − o (p + d − 1) ψ) − 1). Used for posterior predictive checks of order statistics.
"""
function decode_orders(model::FoldModel, u::AbstractVector)
    (; k, ngrid_κ, ϕ_grid, κ_nodes, κ_logweights, data) = model
    θ = constrained_parameters(model, map(Float64, u))
    (; logμκ, σκ, α) = θ
    cbest = (has_tempo(model) && any(s -> _ncandidates(model, s) > 1, eachindex(data)) ? argmax.(offset_posterior(model, u)) : ones(Int, length(data)))
    combos = [(o, ψ, lp, ϕ) for (o, (ψ, lp)) in Iterators.product((-1, 1), ((Golden_angle, log(α)), (Lucas_angle, log1p(-α)))) for ϕ in ϕ_grid]
    out = Vector{Vector{Int}}(undef, length(data))
    Threads.@threads for s in eachindex(data)
        y = corrected_angles(model, s)
        n = length(y)
        η = first(fold_eta(model, θ, s; c = cbest[s]))
        ηrow(i, j) = (η isa AbstractMatrix ? η[i, j] : η[j])
        ws = PermanentWorkspace(k, n; logspace = true)
        L = fill(-Inf, n, 2k + 1)
        fill_L! = (o, ψ, ϕ, κ) -> begin
            fill!(L, -Inf)
            for d in -k:k, i in max(1, 1 - d):min(n, n - d)
                L[i, d + k + 1] = ηrow(i, d + k + 1) + κ * (cos(y[i] - ϕ - o * (i + d - 1) * ψ) - 1)
            end
        end
        best, arg = -Inf, (1, Golden_angle, 0.0, 1.0)
        for q in 1:ngrid_κ, (o, ψ, lp, ϕ) in combos
            κ = exp(logμκ + σκ * κ_nodes[q])
            fill_L!(o, ψ, ϕ, κ)
            ws.w .= L
            v = forward!(ws, n) + n * κ - n * logbesseli(0, κ) + lp + κ_logweights[q]
            (v > best) && (best = v; arg = (o, ψ, ϕ, κ))
        end
        fill_L!(arg...)
        out[s] = map_permutation(ws.g, L, n)
    end
    return out
end

"""
    simulate_stems(tab, μ, γ; plants, organs, κ, α, torsion_mean, torsion_sd, tempo, β, rng)

Simulate from the generative model: site i is ready at iγ and commits at τ_i = iγ + z_i (z_i i.i.d. from the passage
law of shape μ, standardised); the organ order is σ = sortperm(τ), and the angles are von Mises,
y_p ~ vM(ϕ + o (σ[p] − 1) ψ + τ_s (p − 1), κ), with a twist τ_s ~ Normal(torsion_mean, torsion_sd) degrees per stem
position. Returns (; angles, orders, torsion).

`tempo`: a plastochron that changes along the stem, either a vector r (relative plastochron of organ i, the same for
every stem) or a function s ↦ r. Organ i is then ready at γ Σ_{j ≤ i} r_j and its delay has sd r_i^(1 − β): its local
readiness spacing is γ r_i^β (the model `FoldModel(...; tempo)`).
"""
function simulate_stems(tab::PassageTable, μ::Real, γ::Real; plants::Int = 100, organs::Int = 50, κ::Real = exp(1.5), α::Real = 0.95,
        torsion_mean::Real = 0.0, torsion_sd::Real = 0.0, tempo = nothing, β::Real = 0.0, rng::AbstractRNG = Random.default_rng())
    smp = passage_sampler(tab, μ)
    orders = Vector{Vector{Int}}(undef, plants)
    angles = Vector{Vector{Float64}}(undef, plants)
    torsion = zeros(plants)
    twist = (torsion_sd > 0 || torsion_mean != 0)
    for s in 1:plants
        if tempo === nothing
            σ = sortperm([i * γ + smp(rng) for i in 1:organs])
        else
            r = (tempo isa AbstractVector ? tempo : tempo(s))
            R = cumsum(r[1:organs])
            σ = sortperm([γ * R[i] + r[i] ^ (1 - β) * smp(rng) for i in 1:organs])
        end
        ψ = (rand(rng) < α ? Golden_angle : Lucas_angle)
        ϕ = 2π * rand(rng)
        o = (rand(rng) < 0.5 ? 1 : -1)
        τ = (twist ? torsion_mean + torsion_sd * randn(rng) : 0.0)
        orders[s] = σ
        torsion[s] = τ
        angles[s] = [rand(rng, VonMises(mod2pi(ϕ + o * (σ[p] - 1) * ψ + deg2rad(τ) * (p - 1)), κ)) for p in 1:organs]
    end
    return (; angles, orders, torsion)
end

end # module FoldAngleModel
