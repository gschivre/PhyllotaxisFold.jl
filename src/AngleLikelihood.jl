"""
    AngleLikelihood

The angle likelihood shared by models A (free η) and B (fold η): per stem, the banded permanent summed over the nuisance
grid, with its analytic gradient in (η, log μκ, σκ, α) from forward–backward (expected displacement counts and expected
cosines). A model plugs in through two methods:

- `stem_value_grad(m, u, s; τ, with_grad, fws)`: the log-likelihood of stem s at unconstrained parameters u (twist rate
  τ) and its gradient in u (the model's map u ↦ η, κ, α and its chain rule);
- `global_logprior(m, u)`: the log prior of u, Jacobian included.

Everything else is generic: the log density and gradient (threaded over stems), the Hessian by one dual pass over the
analytic gradient (`angle_vgh`), the MAP (`angle_fit_map`), and stem torsion as an integrated random effect
(`torsion_vgh`, `angle_fit_map_torsion`).
"""
module AngleLikelihood

using Statistics
using LogExpFunctions, ForwardDiff, Optim
using ..Common: Golden_angle, Lucas_angle, logbesseli, besseliratio, _isneginf
using ..Permanents: PermanentWorkspace, forward!, backward!, toeplitz_weights!
using ..Torsion: TorsionPrior, TorsionNodes, torsion_step, torsion_hyperprior, stem_nodes, node_logterms, node_posterior,
    newton_map, scan_rate

export StemWorkspace, stem_loglik!, stem_value_grad, global_logprior, angle_model, angle_vgh, angle_fit_map,
       angle_update_torsion!, angle_torsion_nodes, torsion_vgh, angle_fit_map_torsion

struct StemWorkspace{T}
    perm::PermanentWorkspace{T}
    C::Array{Float64, 3} # cos(y_i - ϕ - o (j - 1) ψ) - 1 per (ϕ, o, ψ) combination, n_max × (2k + 1) × 4 ngrid_ϕ
    logp_c::Vector{T} # log p(ψ) per combination
    galpha_c::Vector{T} # ∂ log p(ψ) / ∂α per combination
    log_Z::Matrix{T} # log-permanent + n κ + log p(ψ), 4 ngrid_ϕ × ngrid_κ
    log_Zq::Vector{T} # per κ node
    grad::Vector{T} # gradient accumulator: ∂ℓ/∂η_d (2k + 1), then log μκ, σκ, α
    gradH::Matrix{T} # per-row gradient ∂ℓ/∂H[i, d] when the log-propensities are a matrix H (n_max × (2k + 1))
end
function StemWorkspace(::Type{T}, k::Int, n_max::Int, ngrid_ϕ::Int, ngrid_κ::Int; logspace::Bool = false) where {T}
    Nc = 4 * ngrid_ϕ
    return StemWorkspace{T}(PermanentWorkspace(T, k, n_max; logspace = logspace), zeros(n_max, 2 * k + 1, Nc), zeros(T, Nc), zeros(T, Nc), zeros(T, Nc, ngrid_κ), zeros(T, ngrid_κ), zeros(T, 2 * k + 4), zeros(T, n_max, 2 * k + 1))
end

"""
    stem_loglik!(fws, grid, y, η, logμκ, σκ, α, with_grad; prune = 1e-14)

Log-likelihood of one stem (angles y, already corrected for torsion) given the full log-propensity vector η (2k + 1,
η[d + k + 1] = η_d), summed over the nuisance grid (o, ψ, ϕ) × κ nodes, and, with `with_grad`, its gradient accumulated
into fws.grad: [∂ℓ/∂η_d (2k + 1, = E_post[N_d] − E_η[N_d]); ∂ℓ/∂log μκ; ∂ℓ/∂σκ; ∂ℓ/∂α]. `grid` has fields k, ngrid_ϕ,
ngrid_κ, ϕ_grid, κ_nodes, κ_logweights (a model, or `stem_grid(model)`). Generic in the number type T of the workspace.

η can also be an n × (2k + 1) matrix H of per-row log-propensities (row i: organ position i; a non-stationary order law,
e.g. a readiness spacing that follows the tempo along the stem). The weights are then H[i, d] + κ (cos − 1), the
normaliser is − log perm(e^H), and with `with_grad` the per-row gradient ∂ℓ/∂H[i, d] = ρ_post[i, d] − ρ_H[i, d] is
written into fws.gradH[1:n, :] (overwritten); fws.grad[1:2k+1] still receives its column sums.
"""
#
# Pass 1 runs the forward pass on the whole (ϕ, o, ψ) × κ grid. Pass 2 runs forward-backward only on
# the grid points whose posterior weight exceeds `prune` (the posterior over the grid is concentrated
# on a few points), and accumulates the expected displacement counts and expected cosines.
_eta(η::AbstractVector, i, d_idx) = η[d_idx]
_eta(η::AbstractMatrix, i, d_idx) = η[i, d_idx]

function stem_loglik!(fws::StemWorkspace{T}, grid, y::AbstractVector, η::AbstractVecOrMat, logμκ, σκ, α, with_grad::Bool; prune::Real = 1e-14) where {T}
    (; k, ngrid_ϕ, ngrid_κ, ϕ_grid, κ_nodes, κ_logweights) = grid
    (; perm, C, logp_c, galpha_c, log_Z, log_Zq, grad, gradH) = fws
    n = length(y)
    w = perm.w
    K = 2 * k + 1
    Nc = 4 * ngrid_ϕ
    rows = η isa AbstractMatrix
    rows && @assert size(η, 1) ≥ n && size(η, 2) == K
    (rows && with_grad) && fill!(view(gradH, 1:n, :), 0)

    # cosines do not depend on κ: compute them once per plant
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
    fill_weights! = (c, κ) -> begin
        @inbounds for d in -k:k, i in max(1, 1 - d):min(n, n - d)
            w[i, d + k + 1] = _eta(η, i, d + k + 1) + κ * C[i, d + k + 1, c]
        end
    end

    # pass 1: forward passes
    for q in 1:ngrid_κ
        κ = exp(logμκ + σκ * κ_nodes[q])
        for c in 1:Nc
            fill_weights!(c, κ)
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
            κ = exp(logμκ + σκ * κ_nodes[q])
            v = exp(log_Zq[q] - lse_q)
            lse_c = logsumexp(view(log_Z, :, q))
            # a κ node where every combination underflowed (very large κ, scaled space) has zero weight
            (_isneginf(lse_c) || v == 0) && continue
            gκ = -n * besseliratio(κ) # ∂ / ∂κ of - n log I₀(κ)
            for c in 1:Nc
                lu = log_Z[c, q] - lse_c
                π_qc = v * exp(lu)
                gκ += exp(lu) * n # from + n κ
                grad[K + 3] += π_qc * galpha_c[c]
                (log(v) + lu < logprune) && continue
                fill_weights!(c, κ)
                forward!(perm, n)
                rho = backward!(perm, n)
                cc = zero(T)
                @inbounds for d_idx in 1:K
                    s = zero(T)
                    for i in 1:n
                        r = rho[i, d_idx]
                        s += r
                        cc += r * C[i, d_idx, c]
                        rows && (gradH[i, d_idx] += π_qc * r)
                    end
                    grad[d_idx] += π_qc * s
                end
                gκ += exp(lu) * cc
            end
            grad[K + 1] += v * κ * gκ
            grad[K + 2] += v * κ * κ_nodes[q] * gκ
        end
    end

    # η normaliser: - log perm(exp η), gradient - E_η[N_d]
    rows ? (w[1:n, :] .= view(η, 1:n, :)) : toeplitz_weights!(perm, η, n)
    ℓ -= forward!(perm, n)
    if with_grad
        rho = backward!(perm, n)
        for d_idx in 1:K, i in 1:n
            grad[d_idx] -= rho[i, d_idx]
            rows && (gradH[i, d_idx] -= rho[i, d_idx])
        end
    end
    return ℓ
end

"""
    stem_value_grad(m, u, s; τ, with_grad, fws)

Log-likelihood of stem s (prior excluded) at the unconstrained parameters u with twist rate τ, and its gradient in u
(`nothing` without `with_grad`); `fws` a `StemWorkspace` of eltype(u) large enough for the stem. Methods: `FoldModel`,
`TransformedLogDensityProblem` (model A).
"""
function stem_value_grad end

"log prior of the unconstrained parameters u, Jacobian included (generic in the number type)"
function global_logprior end

"the underlying angle model (its fields data, torsion, k, ngrid_ϕ, ngrid_κ, logspace)"
function angle_model end

_workspace(m, ::Type{T}, n) where {T} = (am = angle_model(m); StemWorkspace(T, am.k, n, am.ngrid_ϕ, am.ngrid_κ; logspace = am.logspace))
_chunks(S) = collect(Iterators.partition(1:S, max(1, cld(S, 2 * Threads.nthreads()))))

# Σ over stems of the log-likelihood and its gradient in u, threaded, generic in eltype(u)
function _data_value_grad(m, u::AbstractVector{T}, with_grad::Bool) where {T}
    am = angle_model(m)
    parts = fetch.([Threads.@spawn begin
        local fws = _workspace(m, T, maximum(length, view(am.data, idxs)))
        local l, g = zero(T), zeros(T, length(u))
        for s in idxs
            ls, gs = stem_value_grad(m, u, s; with_grad, fws)
            l += ls
            with_grad && (g .+= gs)
        end
        (l, g)
    end for idxs in _chunks(length(am.data))])
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
function angle_fit_map(m; u0, options = Optim.Options(g_tol = 1e-5, iterations = 100))
    nf = newton_map((u, order) -> angle_vgh(m, u, order), u0; options)
    return (; u = nf.v, H = nf.H, result = nf.result)
end

stem_loglik(m, u, s, τ; fws = _workspace(m, Float64, length(angle_model(m).data[s]))) =
    first(stem_value_grad(m, map(Float64, u), s; τ, with_grad = false, fws))

# run f(s, fws) for every stem, one task and one Float64 workspace per stem, longest stems first (a scan evaluates its
# stem's likelihood a few hundred times: the workspace is reused, and long stems do not finish last)
function _per_stem(f, m)
    data = angle_model(m).data
    @sync for s in sortperm(length.(data); rev = true)
        Threads.@spawn f(s, _workspace(m, Float64, length(data[s])))
    end
end

"each stem's rate set to the maximum of its own log-likelihood at u (scan + golden section)"
function angle_update_torsion!(m, u; scan = -25.0:0.5:25.0)
    tor = angle_model(m).torsion
    uf = map(Float64, u)
    _per_stem(m) do s, fws
        tor.τ[s] = scan_rate(τ -> stem_loglik(m, uf, s, τ; fws), torsion_step(tor.x[s]); scan)
    end
    return tor.τ
end

"adaptive quadrature nodes of every stem's rate at u (`stem_nodes`)"
function angle_torsion_nodes(m, u; kw...)
    tor = angle_model(m).torsion
    uf = map(Float64, u)
    res = Vector{Any}(undef, length(tor.τ))
    _per_stem(m) do s, fws
        res[s] = stem_nodes(τ -> stem_loglik(m, uf, s, τ; fws), torsion_step(tor.x[s]); kw...)
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
    π = exp.(a .- L)
    g = zeros(T, G + 2)
    for j in eachindex(ω)
        g[1:G] .+= π[j] .* vals[j][2]
        r = (ω[j] - mω) / sω
        g[G + 1] += π[j] * r / sω
        g[G + 2] += π[j] * (r^2 - 1)
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
    parts = reduce(vcat, fetch.([Threads.@spawn begin
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
    end for idxs in _chunks(length(am.data))]))
    lp = w -> global_logprior(m, w[1:G]) + torsion_hyperprior(prior, w[G + 1], w[G + 2])
    l = lp(v) + sum(first, parts)
    order == 0 && return (l, nothing, nothing)
    g = ForwardDiff.gradient(lp, v) + sum(p -> p[2], parts)
    order == 1 && return (l, g, nothing)
    H = ForwardDiff.hessian(lp, v) + sum(p -> p[3], parts)
    return (l, g, (H + H') / 2)
end

"""
    angle_fit_map_torsion(m; u0, prior, recenter, nodes_kw, options)

The fit behind `fit_map_torsion`: fit without torsion; nodes; fit with each rate at its best mode; nodes; integrated fit
(Newton on `torsion_vgh`); `recenter` times: nodes at the new fit and refit. Returns (; u, H, result, m_ω, s_ω, τ,
τ_mean, τ_sd, loglik, nodes), u = [model u; m_ω; log s_ω].
"""
function angle_fit_map_torsion(m; u0, prior::TorsionPrior = TorsionPrior(), recenter::Int = 1, nodes_kw = (;),
        options = Optim.Options(g_tol = 1e-5, iterations = 100))
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
    return (; u = v, H = nf.H, result = nf.result, m_ω = v[G + 1], s_ω = exp(v[G + 2]), τ = copy(tor.τ),
        τ_mean = getindex.(post, 1), τ_sd = getindex.(post, 2), loglik = sum(logsumexp, terms), nodes)
end

end # module AngleLikelihood
