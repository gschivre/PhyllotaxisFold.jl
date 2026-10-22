"""
    AngleLikelihood

The angle likelihood of an order model: per stem, the banded permanent summed over the nuisance grid, with its analytic
gradient in (η, log μκ, σκ, α) from forward–backward (expected displacement counts and expected cosines). Optionally the
emission is a mixture with outlier angles (probability π_o, uniform on the circle). The propensity model
(`PhyllotaxisModelAD`) is the model that plugs in here.

A model plugs in through these methods (all on the unconstrained parameters u):

- `stem_value_grad(m, u, s; τ, with_grad, fws)`: the log-likelihood of stem s at u (twist rate τ) and its gradient in u
  (the model's map u ↦ η, κ, α, π_o and its chain rule);
- `global_logprior(m, u)`: the log prior of u, Jacobian included;
- `stem_etas(m, u, s)`: the candidate log-propensity vectors η of stem s and their log prior weights (one candidate,
  or one per uncertain rank offset);
- `angle_nuisance(m, u)`: (; logμκ, σκ, α, outlier), outlier = π_o or `nothing`;
- `stem_grid(m)`: the nuisance grid (k, ngrid_ϕ, ngrid_κ, ϕ_grid, κ_nodes, κ_logweights).

Everything else is generic: the log density and gradient (threaded over stems), the Hessian by one dual pass over the
analytic gradient (`angle_vgh`), the MAP (`angle_fit_map`), stem torsion as an integrated random effect (`torsion_vgh`,
`angle_fit_map_torsion!`), the per-stem and total log-likelihoods (`stem_loglik`, `angle_loglikelihood`), and the
decoding of the organ orders (`decode_orders`: joint MAP; `order_posterior`: exact posterior marginals and samples, the
twist rate integrated out).
"""
module AngleLikelihood

using Random, Statistics
using LogExpFunctions, ForwardDiff, Optim
using Distributions: VonMises
using ..Common: Golden_angle, Lucas_angle, logbesseli, besseliratio, _isneginf
using ..Permanents: PermanentWorkspace, forward!, backward!, toeplitz_weights!, map_permutation, sample_permutation!
using ..Torsion: TorsionPrior, TorsionNodes, MAP_OPTIONS, torsion_step, torsion_hyperprior, stem_nodes, node_logterms,
    node_posterior, newton_map, scan_rate, golden_section_max, corrected_angles

export StemWorkspace, stem_loglik!, stem_value_grad, global_logprior, angle_model, stem_etas, angle_nuisance, stem_grid,
    angle_vgh, angle_fit_map, angle_update_torsion!, angle_torsion_nodes, torsion_vgh, angle_fit_map_torsion!,
    stem_loglik, angle_loglikelihood, decode_orders, order_posterior, spiral_posterior, simulate_stems, constrained_parameters,
    unconstrained_parameters, prior_mean_parameters

"""
    StemWorkspace(T, k, n_max, ngrid_ϕ, ngrid_κ; logspace = false)

Buffers of `stem_loglik!` for stems of at most `n_max` organs, band k and a nuisance grid of `ngrid_ϕ` offsets and
`ngrid_κ` κ nodes, with element type T (Float64, or a ForwardDiff dual type for the Hessian pass): the banded-permanent
workspace, the cosine table of the nuisance combinations, the per-combination log-permanents and the gradient
accumulator `grad`. One per task; it is reused across stems and evaluations.
"""
struct StemWorkspace{T}
    perm::PermanentWorkspace{T}
    C::Array{Float64, 3} # cos(y_i - ϕ - o (j - 1) ψ) - 1 per (ϕ, o, ψ) combination, n_max × (2k + 1) × 4 ngrid_ϕ
    logp_c::Vector{T} # log p(ψ) per combination
    galpha_c::Vector{T} # ∂ log p(ψ) / ∂α per combination
    log_Z::Matrix{T} # log-permanent + n κ + log p(ψ), 4 ngrid_ϕ × ngrid_κ
    log_Zq::Vector{T} # per κ node
    grad::Vector{T} # gradient accumulator: ∂ℓ/∂η_d (2k + 1), then log μκ, σκ, α, π_o
end
function StemWorkspace(::Type{T}, k::Int, n_max::Int, ngrid_ϕ::Int, ngrid_κ::Int; logspace::Bool = false) where {T}
    Nc = 4 * ngrid_ϕ
    return StemWorkspace{T}(PermanentWorkspace(T, k, n_max; logspace = logspace), zeros(n_max, 2 * k + 1, Nc), zeros(T, Nc), zeros(T, Nc), zeros(T, Nc, ngrid_κ), zeros(T, ngrid_κ), zeros(T, 2 * k + 5))
end

# nuisance combination c of the grid ↦ (o, ψ, log p(ψ), ϕ), in the order of `stem_loglik!` (ϕ fastest, then o, then ψ)
function _combo(grid, c::Int, α)
    nϕ = grid.ngrid_ϕ
    j, iϕ = divrem(c - 1, nϕ)
    io, iψ = j % 2, j ÷ 2
    ψ, lp = (iψ == 0 ? (Golden_angle, log(α)) : (Lucas_angle, log1p(-α)))
    return (o = (io == 0 ? -1 : 1), ψ, lp, ϕ = grid.ϕ_grid[iϕ + 1])
end

# cosine table of the nuisance combinations (ϕ fastest, then o, then ψ) with log p(ψ) and ∂ log p(ψ) / ∂α of each: the
# cosines depend on the angles (so on the twist rate), not on κ or η
function _fill_cosines!(fws::StemWorkspace, grid, y::AbstractVector, α)
    (; k, ϕ_grid) = grid
    (; C, logp_c, galpha_c) = fws
    n = length(y)
    cidx = 0
    for (o, (ψ, logp, galpha)) in Iterators.product((-1, 1), ((Golden_angle, log(α), 1 / α), (Lucas_angle, log1p(-α), -1 / (1 - α))))
        for ϕ in ϕ_grid
            cidx += 1
            logp_c[cidx] = logp
            galpha_c[cidx] = galpha
            @inbounds for d in -k:k, i in max(1, 1 - d):min(n, n - d)
                C[i, d + k + 1, cidx] = cos(y[i] - ϕ - o * (i + d - 1) * ψ) - 1
            end
        end
    end
    return fws
end

# the two log terms of the outlier mixture at κ, (log(1 − π), log π + log I₀(κ) − κ), or nothing without outliers
_outlier_terms(::Nothing, κ) = nothing
_outlier_terms(outlier, κ) = (log1p(-outlier), log(outlier) + logbesseli(0, κ) - κ)

# edge log-weights of combination c at κ: η_d + log((1 − π) e^{κ (cos − 1)} + π I₀(κ) e^{−κ}) with outliers of probability
# π (the von Mises density and the uniform one, both divided by e^κ / (2π I₀(κ)), which the caller adds back), and
# η_d + κ (cos − 1) without; ot = `_outlier_terms(π, κ)`
function _fill_weights!(w, η, C, c::Int, κ, n::Int, k::Int, ot)
    if ot === nothing
        @inbounds for d in -k:k, i in max(1, 1 - d):min(n, n - d)
            w[i, d + k + 1] = η[d + k + 1] + κ * C[i, d + k + 1, c]
        end
    else
        la, lb = ot
        @inbounds for d in -k:k, i in max(1, 1 - d):min(n, n - d)
            w[i, d + k + 1] = η[d + k + 1] + logaddexp(la + κ * C[i, d + k + 1, c], lb)
        end
    end
    return w
end

"""
    stem_loglik!(fws, grid, y, η, logμκ, σκ, α, with_grad; prune = 1e-14, outlier = nothing)

Log-likelihood of one stem (angles y, already corrected for torsion) given the full log-propensity vector η (2k + 1,
η[d + k + 1] = η_d), summed over the nuisance grid (o, ψ, ϕ) × κ nodes, and, with `with_grad`, its gradient accumulated
into fws.grad: [∂ℓ/∂η_d (2k + 1, = E_post[N_d] − E_η[N_d]); ∂ℓ/∂log μκ; ∂ℓ/∂σκ; ∂ℓ/∂α; ∂ℓ/∂π_o]. `grid` has fields k,
ngrid_ϕ, ngrid_κ, ϕ_grid, κ_nodes, κ_logweights (`stem_grid(model)`). Generic in the number type T of the workspace.

`outlier = π_o`: each angle is drawn from the von Mises of its site with probability 1 − π_o, and uniformly on the circle
otherwise (a gross measurement error, or an organ the site model does not describe). Without it (`nothing`), the
emission is the von Mises alone and ∂ℓ/∂π_o is not written.

Pass 1 runs the forward pass on the whole (ϕ, o, ψ) × κ grid. Pass 2 runs forward–backward only on the grid points whose
posterior weight exceeds `prune` (the posterior over the grid is concentrated on a few points), and accumulates the
expected displacement counts and expected cosines.
"""
function stem_loglik!(
        fws::StemWorkspace{T}, grid, y::AbstractVector, η::AbstractVector, logμκ, σκ, α, with_grad::Bool;
        prune::Real = 1.0e-14, outlier = nothing
    ) where {T}
    (; k, ngrid_ϕ, ngrid_κ, κ_nodes, κ_logweights) = grid
    (; perm, C, logp_c, galpha_c, log_Z, log_Zq, grad) = fws
    n = length(y)
    w = perm.w
    K = 2 * k + 1
    Nc = 4 * ngrid_ϕ

    # cosines do not depend on κ: compute them once per plant; the outlier terms depend on κ alone
    _fill_cosines!(fws, grid, y, α)
    κs = [exp(logμκ + σκ * z) for z in κ_nodes]
    ots = [_outlier_terms(outlier, κ) for κ in κs]
    fill_weights!(c, q) = _fill_weights!(w, η, C, c, κs[q], n, k, ots[q])

    # pass 1: forward passes
    for q in 1:ngrid_κ
        κ = κs[q]
        for c in 1:Nc
            fill_weights!(c, q)
            log_Z[c, q] = forward!(perm, n) + n * κ + logp_c[c]
        end
        log_Zq[q] = κ_logweights[q] + logsumexp(view(log_Z, :, q)) - n * logbesseli(0, κ)
    end
    lse_q = logsumexp(log_Zq)
    # mean over o (½ each), p(ψ) (in log_Z) and ϕ (1 / ngrid_ϕ): the Nc combinations carry a total weight of 2
    ℓ = lse_q - log(2 * ngrid_ϕ) - n * log(2π)

    # pass 2: posterior weights π_qc = v_q u_qc and expected sufficient statistics
    if with_grad
        logprune = log(prune)
        for q in 1:ngrid_κ
            κ = κs[q]
            v = exp(log_Zq[q] - lse_q)
            lse_c = logsumexp(view(log_Z, :, q))
            # a κ node where every combination underflowed (very large κ, scaled space) has zero weight
            (_isneginf(lse_c) || v == 0) && continue
            Aκ = besseliratio(κ)
            gκ = -n * Aκ # ∂ / ∂κ of - n log I₀(κ)
            gπ = zero(T)
            # with outliers: ∂w/∂κ = r C + (1 − r)(A(κ) − 1) and ∂w/∂π = −r / (1 − π) + (1 − r) / π, r = P(von Mises | edge)
            la, lb = (ots[q] === nothing ? (zero(T), zero(T)) : ots[q])
            for c in 1:Nc
                lu = log_Z[c, q] - lse_c
                π_qc = v * exp(lu)
                gκ += exp(lu) * n # from + n κ
                grad[K + 3] += π_qc * galpha_c[c]
                (log(v) + lu < logprune) && continue
                fill_weights!(c, q)
                forward!(perm, n)
                rho = backward!(perm, n)
                cc, cπ = zero(T), zero(T)
                @inbounds for d_idx in 1:K
                    s = zero(T)
                    for i in 1:n
                        r = rho[i, d_idx]
                        s += r
                        if outlier === nothing
                            cc += r * C[i, d_idx, c]
                        else
                            a = la + κ * C[i, d_idx, c]
                            rv = exp(a - logaddexp(a, lb))
                            cc += r * (rv * C[i, d_idx, c] + (1 - rv) * (Aκ - 1))
                            cπ += r * ((1 - rv) / outlier - rv / (1 - outlier))
                        end
                    end
                    grad[d_idx] += π_qc * s
                end
                gκ += exp(lu) * cc
                gπ += exp(lu) * cπ
            end
            grad[K + 1] += v * κ * gκ
            grad[K + 2] += v * κ * κ_nodes[q] * gκ
            grad[K + 4] += v * gπ
        end
    end

    # η normaliser: - log perm(exp η), gradient - E_η[N_d]
    toeplitz_weights!(perm, η, n)
    ℓ -= forward!(perm, n)
    if with_grad
        rho = backward!(perm, n)
        for d_idx in 1:K, i in 1:n
            grad[d_idx] -= rho[i, d_idx]
        end
    end
    return ℓ
end

"""
    stem_value_grad(m, u, s; τ, with_grad, fws)

Log-likelihood of stem s (prior excluded) at the unconstrained parameters u with twist rate τ, and its gradient in u
(`nothing` without `with_grad`); `fws` a `StemWorkspace` of eltype(u) large enough for the stem. Methods:
`PhyllotaxisModelAD` (or its `TransformedLogDensityProblem`).
"""
function stem_value_grad end

"log prior of the unconstrained parameters u, Jacobian included (generic in the number type)"
function global_logprior end

"the underlying angle model (its fields data, torsion, k, ngrid_ϕ, ngrid_κ, logspace)"
function angle_model end

"""
    stem_etas(m, u, s)

Candidate log-propensities of stem s at u and their log prior weights, (etas, logw): one vector η of 2k + 1 with log
weight 0, or one per candidate rank offset.
"""
function stem_etas end

"(; logμκ, σκ, α, outlier) at u: the angle nuisance parameters (outlier = π_o, or `nothing` without outliers)"
function angle_nuisance end

"the nuisance grid (k, ngrid_ϕ, ngrid_κ, ϕ_grid, κ_nodes, κ_logweights) of a model"
function stem_grid end

"""
    constrained_parameters(model, u)

The model's parameters in their natural space, as a NamedTuple, from the unconstrained vector u: the nuisance fields
(logμκ, σκ, α, outlier) and, for the propensity model, η (the full 2k + 1 log-propensities).
"""
function constrained_parameters end

"`unconstrained_parameters(model; kwargs...)`: the unconstrained vector u from named constrained values"
function unconstrained_parameters end

"`prior_mean_parameters(model)`: the unconstrained u at the prior mean (the default starting point of the fits)"
function prior_mean_parameters end

"""
    simulate_stems(...)

Simulated stems (angles, true orders, twist rates). Methods: `simulate_stems(η; ...)` (orders from the banded permanent
with log-propensities η) and `simulate_stems(f::DelayLaw; ...)` (orders from fold commitment with i.i.d. delays f).
Both take the same angle and twist options.
"""
function simulate_stems end

_workspace(m, ::Type{T}, n) where {T} = (am = angle_model(m); StemWorkspace(T, am.k, n, am.ngrid_ϕ, am.ngrid_κ; logspace = am.logspace))
_chunks(S) = collect(Iterators.partition(1:S, max(1, cld(S, 2 * Threads.nthreads()))))

# Σ over stems of the log-likelihood and its gradient in u, threaded, generic in eltype(u)
function _data_value_grad(m, u::AbstractVector{T}, with_grad::Bool) where {T}
    am = angle_model(m)
    parts = fetch.(
        [
            Threads.@spawn begin
                local fws = _workspace(m, T, maximum(length, view(am.data, idxs)))
                local l, g = zero(T), zeros(T, length(u))
                for s in idxs
                    ls, gs = stem_value_grad(m, u, s; with_grad, fws)
                    l += ls
                    with_grad && (g .+= gs)
                end
                (l, g)
            end for idxs in _chunks(length(am.data))
        ]
    )
    return sum(first, parts), (with_grad ? sum(last, parts) : nothing)
end

struct AngleHessianTag end

# seed u with the identity: duals whose partials are the derivatives in u
_seed(v) = (P = length(v); [ForwardDiff.Dual{AngleHessianTag, Float64, P}(v[i], ForwardDiff.Partials(ntuple(j -> Float64(i == j), P))) for i in 1:P])
_hessian_from(gd) = [ForwardDiff.partials(gd[i], j) for i in eachindex(gd), j in eachindex(gd)]

"""
    angle_vgh(m, u, order)

Log posterior at u (data + `global_logprior`), with its gradient (order ≥ 1, analytic) and Hessian (order 2: one dual
pass over the analytic gradient). If the scaled pass fails on duals, the model's `logspace` flag decides; the result is
checked by the optimiser (non-finite = rejected step).
"""
function angle_vgh(m, u::AbstractVector, order::Int)
    lp = w -> global_logprior(m, w)
    if order == 0
        return (_data_value_grad(m, u, false)[1] + lp(u), nothing, nothing)
    elseif order == 1
        l, g = _data_value_grad(m, u, true)
        return (l + lp(u), g + ForwardDiff.gradient(lp, u), nothing)
    end
    ld, gd = _data_value_grad(m, _seed(u), true)
    H = _hessian_from(gd) + ForwardDiff.hessian(lp, u)
    return (ForwardDiff.value(ld) + lp(u), ForwardDiff.value.(gd) + ForwardDiff.gradient(lp, u), (H + H') / 2)
end

"MAP by Newton trust region with analytic gradient and dual-pass Hessian; returns (; u, H, result)"
function angle_fit_map(m; u0, options = MAP_OPTIONS)
    nf = newton_map((u, order) -> angle_vgh(m, u, order), u0; options)
    return (; u = nf.v, H = nf.H, result = nf.result)
end

_stem_ll(m, u, s, τ; fws = _workspace(m, Float64, length(angle_model(m).data[s]))) =
    first(stem_value_grad(m, map(Float64, u), s; τ, with_grad = false, fws))

"""
    stem_loglik(m, u, s; τ)

Log-likelihood of stem s alone (prior excluded) at the unconstrained parameters u (model parameters only), with twist
rate τ (the stem's current rate by default).
"""
stem_loglik(m, u::AbstractVector, s::Int; τ::Real = angle_model(m).torsion.τ[s]) = _stem_ll(m, u, s, τ)

"""
    angle_loglikelihood(m, u)
    angle_loglikelihood(m, tfit)

Total log-likelihood (prior excluded): at u with each stem's current twist rate, or, from a `fit_map_torsion!` result, with
the twist rates integrated out (the fit's `loglik`, recomputed at its parameters and nodes).
"""
angle_loglikelihood(m, u::AbstractVector) = _data_value_grad(m, map(Float64, u), false)[1]
function angle_loglikelihood(m, tfit::NamedTuple)
    fws = _workspace(m, Float64, maximum(length, angle_model(m).data))
    return sum(s -> logsumexp(_node_terms(m, tfit.fit.u, s, tfit.nodes, fws)), eachindex(angle_model(m).data))
end

# f(s, fws) for every stem, one task and one Float64 workspace per stem, longest stems first (a scan evaluates its
# stem's likelihood a few hundred times: the workspace is reused, and long stems do not finish last); the results in
# stem order
function _per_stem(f, m)
    data = angle_model(m).data
    tasks = Vector{Task}(undef, length(data))
    for s in sortperm(length.(data); rev = true)
        tasks[s] = Threads.@spawn f(s, _workspace(m, Float64, length(data[s])))
    end
    return [fetch(t) for t in tasks]
end

"each stem's rate set to the maximum of its own log-likelihood at u (scan + golden section)"
function angle_update_torsion!(m, u; scan = -25.0:0.5:25.0)
    tor = angle_model(m).torsion
    uf = map(Float64, u)
    tor.τ .= _per_stem(m) do s, fws
        scan_rate(τ -> _stem_ll(m, uf, s, τ; fws), torsion_step(tor.x[s]); scan)
    end
    return tor.τ
end

"adaptive quadrature nodes of every stem's rate at u (`stem_nodes`)"
function angle_torsion_nodes(m, u; kw...)
    tor = angle_model(m).torsion
    uf = map(Float64, u)
    res = _per_stem(m) do s, fws
        stem_nodes(τ -> _stem_ll(m, uf, s, τ; fws), torsion_step(tor.x[s]); kw...)
    end
    return TorsionNodes(getindex.(res, 1), getindex.(res, 2), getindex.(res, 3))
end

# integrated log-likelihood of stem s over its twist nodes at v = [u; m_ω; log s_ω] and its gradient in v, generic in
# eltype(v); only the nodes in `keep`
function _stem_integrated(m, v::AbstractVector{T}, s::Int, nodes::TorsionNodes, keep, fws) where {T}
    G = length(v) - 2
    u = v[1:G]
    mω, sω = v[G + 1], exp(v[G + 2])
    ω, logw = nodes.ω[s][keep], nodes.logw[s][keep]
    vals = [stem_value_grad(m, u, s; τ = ω[j], with_grad = true, fws) for j in eachindex(ω)]
    a = node_logterms(ω, logw, first.(vals), mω, sω)
    L = logsumexp(a)
    pw = exp.(a .- L) # posterior weight of each node
    g = zeros(T, G + 2)
    for j in eachindex(ω)
        g[1:G] .+= pw[j] .* vals[j][2]
        r = (ω[j] - mω) / sω
        g[G + 1] += pw[j] * r / sω
        g[G + 2] += pw[j] * (r^2 - 1)
    end
    return L, g
end

# Float64 node terms a_sj of stem s at v
function _node_terms(m, v, s, nodes, fws)
    G = length(v) - 2
    ℓ = [first(stem_value_grad(m, v[1:G], s; τ, with_grad = false, fws)) for τ in nodes.ω[s]]
    return node_logterms(nodes.ω[s], nodes.logw[s], ℓ, v[G + 1], exp(v[G + 2]))
end

"""
    torsion_vgh(m, prior, nodes, v, order)

Log posterior with the twist rates integrated out, v = [u; m_ω; log s_ω], up to the Hessian (one dual pass per stem over
its analytic gradient; nodes below e^-30 of the best skipped in the derivative passes).
"""
function torsion_vgh(m, prior::TorsionPrior, nodes::TorsionNodes, v::AbstractVector, order::Int)
    am = angle_model(m)
    G = length(v) - 2
    parts = reduce(
        vcat, fetch.(
            [
                Threads.@spawn begin
                    local nmax = maximum(length, view(am.data, idxs))
                    local fws = _workspace(m, Float64, nmax)
                    local fwd = (order == 2 ? _workspace(m, ForwardDiff.Dual{AngleHessianTag, Float64, G + 2}, nmax) : nothing)
                    map(idxs) do s
                        a = _node_terms(m, v, s, nodes, fws)
                        L0 = logsumexp(a)
                        order == 0 && return (L0, nothing, nothing)
                        keep = a .≥ maximum(a) - 30
                        order == 1 && return (L0, _stem_integrated(m, v, s, nodes, keep, fws)[2], nothing)
                        _, gd = _stem_integrated(m, _seed(v), s, nodes, keep, fwd)
                        (L0, ForwardDiff.value.(gd), _hessian_from(gd))
                    end
                end for idxs in _chunks(length(am.data))
            ]
        )
    )
    lp = w -> global_logprior(m, w[1:G]) + torsion_hyperprior(prior, w[G + 1], w[G + 2])
    l = lp(v) + sum(first, parts)
    order == 0 && return (l, nothing, nothing)
    g = ForwardDiff.gradient(lp, v) + sum(p -> p[2], parts)
    order == 1 && return (l, g, nothing)
    H = ForwardDiff.hessian(lp, v) + sum(p -> p[3], parts)
    return (l, g, (H + H') / 2)
end

"""
    angle_fit_map_torsion!(m; u0, prior, recenter, nodes_kw, options)

The fit behind `fit_map_torsion!`: fit without torsion; nodes; fit with each rate at its best mode; nodes; integrated fit
(Newton on `torsion_vgh`); `recenter` times: nodes at the new fit and refit. Returns (; u, H, result, m_ω, s_ω, τ,
τ_mean, τ_sd, loglik, nodes), u = [model u; m_ω; log s_ω].
"""
function angle_fit_map_torsion!(
        m; u0, prior::TorsionPrior = TorsionPrior(), recenter::Int = 1, nodes_kw = (;),
        options = MAP_OPTIONS
    )
    tor = angle_model(m).torsion
    tor.τ .= 0
    u = angle_fit_map(m; u0, options).u
    nodes = angle_torsion_nodes(m, u; nodes_kw...)
    tor.τ .= [first(md)[1] for md in nodes.modes] # best mode of each stem
    u = angle_fit_map(m; u0 = u, options).u
    nodes = angle_torsion_nodes(m, u; nodes_kw...)
    G = length(u)
    v0 = vcat(u, mean(tor.τ), log(max(std(tor.τ), 0.1)))
    nf = nothing
    for r in 0:recenter
        nf = newton_map((v, order) -> torsion_vgh(m, prior, nodes, v, order), v0; options)
        v0 = nf.v
        r < recenter && (nodes = angle_torsion_nodes(m, nf.v[1:G]; nodes_kw...))
    end
    v = nf.v
    fws = _workspace(m, Float64, maximum(length, angle_model(m).data))
    terms = [_node_terms(m, v, s, nodes, fws) for s in eachindex(tor.τ)]
    post = [node_posterior(nodes.ω[s], a, nodes.modes[s]) for (s, a) in enumerate(terms)]
    tor.τ .= getindex.(post, 3)
    return (;
        u = v, H = nf.H, result = nf.result, m_ω = v[G + 1], s_ω = exp(v[G + 2]), τ = copy(tor.τ),
        τ_mean = getindex.(post, 1), τ_sd = getindex.(post, 2), loglik = sum(logsumexp, terms), nodes,
    )
end

#################################################################################################################
# decoding: the organ order of each stem
#################################################################################################################

# State of stem s at twist rate τ, for every candidate log-propensity (rank offset): the cosine table of the nuisance
# combinations stays in fws (it depends on τ, not on the candidate); per candidate c, its log prior weight plus its stem
# log-likelihood (lc) and the within-candidate log posterior of each nuisance combination × κ node (P[c], Nc × ngrid_κ,
# normalised) and log perm(e^η_c) (lperm); per κ node, κ, its outlier terms (`_outlier_terms`) and n log I₀(κ).
# `_refresh_cosines!(fws, st)` restores the cosine table of a state after other twist rates were evaluated.
function _stem_state(m, u, s::Int, τ::Real, fws::StemWorkspace)
    grid = stem_grid(m)
    nu = angle_nuisance(m, u)
    y = corrected_angles(angle_model(m), s; τ)
    n = length(y)
    κs = [exp(nu.logμκ + nu.σκ * z) for z in grid.κ_nodes]
    ots = [_outlier_terms(nu.outlier, κ) for κ in κs]
    nlI0 = [n * logbesseli(0, κ) for κ in κs]
    etas, logw = stem_etas(m, u, s)
    lc = zeros(length(etas))
    P = Vector{Matrix{Float64}}(undef, length(etas))
    lperm = zeros(length(etas))
    for (c, η) in enumerate(etas)
        ℓ = stem_loglik!(fws, grid, y, η, nu.logμκ, nu.σκ, nu.α, false; outlier = nu.outlier)
        B = [fws.log_Z[j, q] + grid.κ_logweights[q] - nlI0[q] for j in axes(fws.log_Z, 1), q in 1:grid.ngrid_κ]
        lB = logsumexp(B)
        lc[c] = logw[c] + ℓ
        P[c] = B .- lB
        lperm[c] = lB - log(2 * grid.ngrid_ϕ) - n * log(2π) - ℓ
    end
    return (; y, n, etas, logw, lc, P, lperm, nu, grid, κs, ots, nlI0)
end
_refresh_cosines!(fws::StemWorkspace, st) = _fill_cosines!(fws, st.grid, st.y, st.nu.α)

# twist rule of stem s: the values of its rate and their log weights (with the population law)
function _twist_rule(m, s::Int, tfit)
    tfit === nothing && return ([angle_model(m).torsion.τ[s]], [0.0])
    nodes = tfit.nodes
    return nodes.ω[s], node_logterms(nodes.ω[s], nodes.logw[s], zeros(length(nodes.ω[s])), tfit.m_ω, tfit.s_ω)
end

_model_u(m, u::AbstractVector) = map(Float64, u)
_model_u(m, tfit::NamedTuple) = map(Float64, tfit.fit.u[1:(end - 2)])
_tfit(::AbstractVector) = nothing
_tfit(tfit::NamedTuple) = tfit

"""
    order_posterior(m, u; nsamples = 0, rng, prune = 1e-8)
    order_posterior(m, tfit; nsamples = 0, rng, prune = 1e-8)

Exact posterior of the organ order of each stem, summed over the nuisance grid (ϕ, o, ψ, κ node), the candidate rank
offsets and, from a `fit_map_torsion!` result `tfit`, the twist rate (its quadrature nodes, weighted by the fitted
population law); with plain parameters u, at each stem's current rate. Per stem: (; ρ, samples, τ_mean, τ_sd), with
ρ[i, d + k + 1] = P(σ[i] = i + d | angles) (the organ observed at position i is canonical organ i + d) and `nsamples`
exact draws of σ (a nuisance state drawn from its posterior, then σ by forward filtering, backward sampling).
States of posterior weight below `prune` are skipped (their mass is renormalised away).
"""
function order_posterior(m, ut; nsamples::Int = 0, rng::AbstractRNG = Random.default_rng(), prune::Real = 1.0e-8)
    am = angle_model(m)
    u, tfit = _model_u(m, ut), _tfit(ut)
    k = am.k
    S = length(am.data)
    seeds = rand(rng, UInt64, S)
    return _per_stem(m) do s, fws
        srng = Xoshiro(seeds[s])
        ω, lpω = _twist_rule(m, s, tfit)
        sts = [_stem_state(m, u, s, ω[j], fws) for j in eachindex(ω)]
        lj = [lpω[j] + logsumexp(sts[j].lc) for j in eachindex(ω)]
        Pj = exp.(lj .- logsumexp(lj))
        # every retained state (node j, candidate c, combination, κ node q) with its posterior weight
        n = length(am.data[s])
        ρ = zeros(n, 2k + 1)
        states, wts = Tuple{Int, Int, Int, Int}[], Float64[]
        for j in eachindex(ω)
            Pj[j] < prune && continue
            st = sts[j]
            _refresh_cosines!(fws, st)
            pc = exp.(st.lc .- logsumexp(st.lc))
            for c in eachindex(st.etas), q in 1:st.grid.ngrid_κ, cb in axes(st.P[c], 1)
                wgt = Pj[j] * pc[c] * exp(st.P[c][cb, q])
                wgt < prune && continue
                _fill_weights!(fws.perm.w, st.etas[c], fws.C, cb, st.κs[q], n, k, st.ots[q])
                forward!(fws.perm, n)
                ρ .+= wgt .* backward!(fws.perm, n)
                push!(states, (j, c, cb, q)); push!(wts, wgt)
            end
        end
        ρ ./= sum(wts)
        samples = Vector{Vector{Int}}()
        if nsamples > 0
            cw = cumsum(wts)
            draws = sort([states[min(searchsortedfirst(cw, rand(srng) * cw[end]), length(cw))] for _ in 1:nsamples])
            jlast = 0
            for (j, c, cb, q) in draws # sorted by twist node: one cosine table per node
                st = sts[j]
                (j != jlast) && (_refresh_cosines!(fws, st); jlast = j)
                _fill_weights!(fws.perm.w, st.etas[c], fws.C, cb, st.κs[q], n, k, st.ots[q])
                push!(samples, sample_permutation!(srng, fws.perm, n))
            end
            shuffle!(srng, samples)
        end
        τm = sum(Pj .* ω)
        (; ρ, samples, τ_mean = τm, τ_sd = sqrt(max(sum(Pj .* (ω .- τm) .^ 2), 0.0)))
    end
end

# best (order, nuisance state) of stem s at twist rate τ: max over the candidates, the nuisance combinations and the κ
# nodes of the joint log posterior (Viterbi on each combination within `Δ` of the best by marginal weight)
function _joint_best(m, u, s::Int, τ::Real, lpτ::Real, fws::StemWorkspace; Δ::Real = 30.0)
    st = _stem_state(m, u, s, τ, fws)
    (; n, nu, grid) = st
    k = grid.k
    best = (; score = -Inf, σ = collect(1:n), τ, ϕ = 0.0, o = 1, ψ = Golden_angle, κ = 0.0, candidate = 1)
    top = maximum(c -> st.lc[c] + maximum(st.P[c]), eachindex(st.etas))
    for c in eachindex(st.etas), q in 1:grid.ngrid_κ, cb in axes(st.P[c], 1)
        (st.lc[c] + st.P[c][cb, q] < top - Δ) && continue
        κ = st.κs[q]
        cmb = _combo(grid, cb, nu.α)
        w = _fill_weights!(fws.perm.w, st.etas[c], fws.C, cb, κ, n, k, st.ots[q])
        σ = map_permutation(fws.perm.g, w, n)
        v = sum(w[i, σ[i] - i + k + 1] for i in 1:n)
        # log p(σ, angles, state): edge weights + the constants removed from them, the state's prior (o: ½, ψ, ϕ: 1/ngrid_ϕ,
        # log κ: its log-normal density at the node), the candidate's prior and its normaliser, the twist's population law
        score = v + n * κ - st.nlI0[q] - n * log(2π) + cmb.lp - log(2 * grid.ngrid_ϕ) - 0.5 * grid.κ_nodes[q]^2 -
            st.lperm[c] + st.logw[c] + lpτ
        score > best.score && (best = (; score, σ, τ, cmb.ϕ, cmb.o, cmb.ψ, κ, candidate = c))
    end
    return best
end

"""
    spiral_posterior(m, u)
    spiral_posterior(m, tfit)

Posterior of the spiral of each stem, summed over the order, ϕ, κ, the rank offsets and (from a `fit_map_torsion!`
result) the twist rate: one (; p_lucas, p_ccw) per stem, P(ψ = Lucas angle | angles) and P(o = +1 | angles) (o = +1:
the site angles increase, a counter-clockwise spiral). Only forward passes (cheaper than `order_posterior`).
"""
function spiral_posterior(m, ut)
    u, tfit = _model_u(m, ut), _tfit(ut)
    return _per_stem(m) do s, fws
        ω, lpω = _twist_rule(m, s, tfit)
        L = fill(-Inf, 2, 2) # [ψ golden / Lucas, o = −1 / +1]
        for j in eachindex(ω)
            st = _stem_state(m, u, s, ω[j], fws)
            for c in eachindex(st.etas), q in 1:st.grid.ngrid_κ, cb in axes(st.P[c], 1)
                cmb = _combo(st.grid, cb, st.nu.α)
                iψ, io = (cmb.ψ == Golden_angle ? 1 : 2), (cmb.o == -1 ? 1 : 2)
                L[iψ, io] = logaddexp(L[iψ, io], lpω[j] + st.lc[c] + st.P[c][cb, q])
            end
        end
        P = exp.(L .- logsumexp(L))
        (; p_lucas = P[2, 1] + P[2, 2], p_ccw = P[1, 2] + P[2, 2])
    end
end

"""
    decode_orders(m, u; full = false)
    decode_orders(m, tfit; full = false, refine = true)

Most likely organ order of each stem: the joint maximum a posteriori of (σ, ϕ, o, ψ, κ node, rank offset, twist rate).
With plain parameters u, at each stem's current twist rate; from a `fit_map_torsion!` result `tfit`, the twist rate is
decoded with the order (prior: the fitted population law N(m_ω, s_ω²)): the best quadrature node, refined by golden
section within the sd of its mode (`refine`), since a different twist can change the order. Returns the vector of
orders σ (σ[i] = canonical index of the organ at position i), or with `full = true` one NamedTuple per stem
(; σ, τ, ϕ, o, ψ, κ, candidate, score).
"""
function decode_orders(m, ut; full::Bool = false, refine::Bool = true)
    am = angle_model(m)
    u, tfit = _model_u(m, ut), _tfit(ut)
    out = _per_stem(m) do s, fws
        tfit === nothing && return _joint_best(m, u, s, am.torsion.τ[s], 0.0, fws)
        lpτ(τ) = -0.5 * ((τ - tfit.m_ω) / tfit.s_ω)^2
        f(τ) = _joint_best(m, u, s, τ, lpτ(τ), fws)
        cand = [f(τ) for τ in unique(tfit.nodes.ω[s])]
        best = cand[argmax([c.score for c in cand])]
        if refine
            modes = tfit.nodes.modes[s]
            h = modes[argmin([abs(md[1] - best.τ) for md in modes])][2]
            _, refined = golden_section_max(f, best.τ - h, best.τ + h; tol = 1.0e-5 * h, score = r -> r.score)
            refined.score > best.score && (best = refined)
        end
        best
    end
    return full ? out : [b.σ for b in out]
end

# the angles of one stem of order σ (twist coordinate x): κ ~ LogNormal(logμκ, σκ), the golden spiral with probability
# α (else the Lucas one), uniform ϕ and orientation, twist τ ~ N(torsion_mean, torsion_sd²) degrees per unit of x, and
# each angle replaced with probability `outlier` by a uniform one (shared by the `simulate_stems` methods)
function _simulate_angles(
        rng::AbstractRNG, σ::AbstractVector{Int}, x::AbstractVector; logμκ, σκ, α, outlier, torsion_mean,
        torsion_sd
    )
    n = length(σ)
    κ = exp(logμκ + σκ * randn(rng))
    ψ = (rand(rng) < α ? Golden_angle : Lucas_angle)
    ϕ = 2π * rand(rng)
    o = (rand(rng) < 0.5 ? 1 : -1)
    τ = torsion_mean + torsion_sd * randn(rng)
    bad = [rand(rng) < outlier for _ in 1:n]
    y = [bad[p] ? 2π * rand(rng) : rand(rng, VonMises(mod2pi(ϕ + o * (σ[p] - 1) * ψ + deg2rad(τ) * x[p]), κ)) for p in 1:n]
    return (; y, τ, κ, outliers = findall(bad), ψ, o)
end

end # module AngleLikelihood
