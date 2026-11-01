# AngleLikelihood
#
# The angle likelihood of the propensity model: per stem, the banded permanent summed over the nuisance grid, with its
# analytic gradient in (η, log μκ, σκ, α, `π_o`) from forward–backward (expected displacement counts and expected
# cosines). Optionally the emission is a mixture with outlier angles (probability `π_o`, uniform on the circle).
#
# The generic machinery, written against the model's fields and a few methods of `PropensityModel`: the log posterior
# and its gradient (threaded over stems, one task per stem, workspaces from a pool), the Hessian by one dual pass over
# the analytic gradient, the twist rates integrated out by adaptive quadrature, the MAP fits behind `fit_map` and their
# result `PhyllotaxisFit`, and the decoding of the organ orders (`decode_orders`: joint MAP; `order_posterior`: exact
# posterior marginals and samples; `spiral_posterior`), the twist rate integrated out.

#################################################################################################################
# the likelihood of one stem
#################################################################################################################

"""
    StemWorkspace(T, g, n_max, ngrid_ϕ, ngrid_κ; logspace = false)

Buffers of `stem_loglik!` for stems of at most `n_max` organs, the permutation graph g of band k and a nuisance grid of
`ngrid_ϕ` offsets and `ngrid_κ` κ nodes, with element type T (Float64, or a ForwardDiff dual type for the Hessian
pass): the banded-permanent workspace, the twist-corrected angles, the cosine table of the nuisance combinations, the
per-combination log-permanents and the gradient accumulator `grad`. One per task; reused across stems and
evaluations (`WorkspacePool`).
"""
struct StemWorkspace{T}
    perm::PermanentWorkspace{T}
    y::Vector{Float64} # angles of the current stem, twist removed
    C::Array{Float64, 3} # cos(y_i - ϕ - o (j - 1) ψ) - 1 per (ϕ, o, ψ) combination, n_max × (2k + 1) × 4 ngrid_ϕ
    logp_c::Vector{T} # log p(ψ) per combination
    galpha_c::Vector{T} # ∂ log p(ψ) / ∂α per combination
    log_Z::Matrix{T} # log-permanent + n κ + log p(ψ), 4 ngrid_ϕ × ngrid_κ
    log_Zq::Vector{T} # per κ node
    grad::Vector{T} # gradient accumulator: ∂ℓ/∂η_d (2k + 1), then log μκ, σκ, α, π_o
    eη::Vector{T} # exp.(η) of the current stem
    active::Matrix{Bool} # the grid points of non-negligible weight, for the dual passes (`_mark_active!`)
end
function StemWorkspace(::Type{T}, g::PermutationGraph, n_max::Int, ngrid_ϕ::Int, ngrid_κ::Int; logspace::Bool = false) where {T}
    Nc, K = 4 * ngrid_ϕ, 2 * g.k + 1
    return StemWorkspace{T}(
        PermanentWorkspace(T, g, n_max; logspace), zeros(n_max), zeros(n_max, K, Nc), zeros(T, Nc), zeros(T, Nc),
        zeros(T, Nc, ngrid_κ), zeros(T, ngrid_κ), zeros(T, K + 4), zeros(T, K), fill(true, Nc, ngrid_κ)
    )
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

# the two log terms of the outlier mixture at κ, (log(1 − π), log π + log I₀(κ) − κ), or nothing without outliers. The
# outlier probability has a concrete type set by the model (`Nothing` without outliers), so the branches on
# `ot === nothing` below are resolved at compile time. (Without outliers the weights are η_d + κ (cos − 1): writing them
# as a mixture with π = 0 would cost a logaddexp per entry and put log 0 in the derivatives.)
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

# the same weights in linear form, e^{η_d} ((1 − π) e^{κ (cos − 1)} + π I₀(κ) e^{−κ}), into the scaled table of the permanent
# workspace (with outliers only: the uniform term keeps every weight away from underflow); eη = exp.(η)
function _fill_linear!(ew, eη, C, c::Int, κ, n::Int, k::Int, ot)
    pa, pb = exp(ot[1]), exp(ot[2])
    @inbounds for d in -k:k, i in max(1, 1 - d):min(n, n - d)
        ew[i, d + k + 1] = eη[d + k + 1] * (pa * exp(κ * C[i, d + k + 1, c]) + pb)
    end
    return ew
end

# log-permanent of combination c at κ: one exponential per entry with outliers (linear weights), the log-weights and
# the exponentials of the scaled pass without
function _forward_combo!(perm, η, eη, C, c::Int, κ, n::Int, k::Int, ot)
    ot === nothing && (_fill_weights!(perm.w, η, C, c, κ, n, k, ot); return forward!(perm, n))
    _fill_linear!(perm.ew, eη, C, c, κ, n, k, ot)
    return forward_linear!(perm, n)
end

"""
    stem_loglik!(fws, grid, y, η, logμκ, σκ, α, with_grad; prune = 1e-14, outlier = nothing)

Log-likelihood of one stem (angles y, already corrected for torsion) given the full log-propensity vector η (2k + 1,
`η[d + k + 1] = η_d`), summed over the nuisance grid (o, ψ, ϕ) × κ nodes, and, with `with_grad`, its gradient accumulated
into `fws.grad`: `[∂ℓ/∂η_d (2k + 1, = E_post[N_d] − E_η[N_d]); ∂ℓ/∂log μκ; ∂ℓ/∂σκ; ∂ℓ/∂α; ∂ℓ/∂π_o]`. `grid` has fields k,
`ngrid_ϕ`, `ngrid_κ`, `ϕ_grid`, `κ_nodes`, `κ_logweights`. Generic in the number type T of the workspace.

`outlier = π_o`: each angle is drawn from the von Mises of its site with probability `1 − π_o`, and uniformly on the circle
otherwise (a gross measurement error, or an organ the site model does not describe). Without it (`nothing`), the
emission is the von Mises alone and `∂ℓ/∂π_o` is not written.

Pass 1 runs the forward pass on the whole (ϕ, o, ψ) × κ grid, or only on its points marked in `active` (a
Bool matrix of 4 `ngrid_ϕ` × `ngrid_κ`; the others get weight 0). Pass 2 runs forward–backward only on the grid points whose
posterior weight exceeds `prune` (the posterior over the grid is concentrated on a few points), and accumulates the
expected displacement counts and expected cosines.
"""
function stem_loglik!(
        fws::StemWorkspace{T}, grid, y::AbstractVector, η::AbstractVector, logμκ, σκ, α, with_grad::Bool;
        prune::Real = 1.0e-14, outlier = nothing, active::Union{Nothing, AbstractMatrix{Bool}} = nothing
    ) where {T}
    (; k, ngrid_ϕ, ngrid_κ, κ_nodes, κ_logweights) = grid
    (; perm, C, logp_c, galpha_c, log_Z, log_Zq, grad) = fws
    n = length(y)
    K = 2 * k + 1
    Nc = 4 * ngrid_ϕ

    # cosines do not depend on κ: compute them once per plant; the outlier terms depend on κ alone
    _fill_cosines!(fws, grid, y, α)
    κs = [exp(logμκ + σκ * z) for z in κ_nodes]
    ots = [_outlier_terms(outlier, κ) for κ in κs]
    eη = fws.eη
    eη .= exp.(η)
    forward_combo!(c, q) = _forward_combo!(perm, η, eη, C, c, κs[q], n, k, ots[q])

    # pass 1: forward passes
    for q in 1:ngrid_κ
        κ = κs[q]
        for c in 1:Nc
            # a combination of negligible weight (`active`, from a Float64 pass at the same point) is left out
            if active !== nothing && !active[c, q]
                log_Z[c, q] = T(-Inf)
                continue
            end
            log_Z[c, q] = forward_combo!(c, q) + n * κ + logp_c[c]
        end
        # (a κ node with no active point has weight 0: logsumexp of −Inf duals would give NaN partials)
        lq = logsumexp(view(log_Z, :, q))
        log_Zq[q] = (_isneginf(lq) ? T(-Inf) : κ_logweights[q] + lq - n * logbesseli(0, κ))
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
                forward_combo!(c, q)
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
                            rv = logistic(la + κ * C[i, d_idx, c] - lb) # e^a / (e^a + e^lb), one exponential
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

#################################################################################################################
# the model's side (methods in PropensityModel)
#################################################################################################################

"`stem_params(m, u)`: (; η, logμκ, σκ, α, outlier) at u (η the full 2k + 1 costs, outlier = `π_o` or `nothing`)"
function stem_params end
"`nmodel(m)`: the number of model parameters (those of the angle model, the twist hyperparameters excluded)"
function nmodel end
"`params_jacobian(m, u)`: Jacobian of the model parameters θ in u (the first `nmodel(m)` entries of u)"
function params_jacobian end
"`accumulate_gradθ!(gθ, m, grad, w)`: add w × the gradient in θ of one stem, read from the workspace accumulator `grad`"
function accumulate_gradθ! end
"`model_logprior(m, u)`: log prior of the model parameters at u (first `nmodel(m)` entries), Jacobian included"
function model_logprior end

"""
    constrained_parameters(model, u)

The model's parameters in their natural space, as a NamedTuple, from the unconstrained vector u: η (the full 2k + 1
log-propensities), logμκ, σκ, α, outlier (`π_o`, 0 without outliers) and, with a twist, `m_ω` and `s_ω`.
"""
function constrained_parameters end

"""
    unconstrained_parameters(model; η, logμκ, σκ, α, outlier, m_ω, s_ω)
    unconstrained_parameters(model, θ::AbstractVector)

The unconstrained vector u from named constrained values, or from the flat constrained vector θ (`coef`).
"""
function unconstrained_parameters end

"`prior_mean_parameters(model)`: the unconstrained u at the prior mean"
function prior_mean_parameters end


#################################################################################################################
# workspaces and threads
#################################################################################################################

"workspaces of one element type, taken by the tasks of an evaluation and given back (thread-safe)"
struct WorkspacePool{T}
    lock::ReentrantLock
    free::Vector{StemWorkspace{T}}
end
WorkspacePool{T}() where {T} = WorkspacePool{T}(ReentrantLock(), StemWorkspace{T}[])

_workspace(m, ::Type{T}) where {T} = StemWorkspace(T, m.graph, m.n_max, m.ngrid_ϕ, m.ngrid_κ; logspace = m.logspace)
function _acquire!(pool::WorkspacePool{T}, m) where {T}
    ws = lock(pool.lock) do
        isempty(pool.free) ? nothing : pop!(pool.free)
    end
    return ws === nothing ? _workspace(m, T) : ws
end
_release!(pool::WorkspacePool, ws) = (lock(() -> push!(pool.free, ws), pool.lock); nothing)
# the model keeps its workspaces between evaluations: Float64 ones, and those of each dual type (Hessian passes)
_pool(m, ::Type{Float64}) = m.pool
_pool(m, ::Type{T}) where {T} = lock(() -> get!(() -> WorkspacePool{T}(), m.dual_pools, T), m.pools_lock)::WorkspacePool{T}

# f(s, fws) for every stem, one task per stem, the longest stems first (they do not finish last), with a workspace of
# element type T from the pool; the results, of type R, in stem order
function _per_stem(f::F, m, ::Type{T}, ::Type{R} = Any) where {F, T, R}
    data = m.data
    pool = _pool(m, T)
    tasks = Vector{Task}(undef, length(data))
    for s in sortperm(length.(data); rev = true)
        tasks[s] = Threads.@spawn _with_workspace(f, pool, m, s)
    end
    return R[fetch(t)::R for t in tasks]
end
function _with_workspace(f::F, pool, m, s::Int) where {F}
    fws = _acquire!(pool, m)
    try
        return f(s, fws)
    finally
        _release!(pool, fws)
    end
end

#################################################################################################################
# log posterior, gradient and Hessian
#################################################################################################################

# angles of stem s with the twist of rate ω (degrees per unit of the coordinate) removed, in the workspace
function _corrected!(fws::StemWorkspace, m, s::Int, ω::Real)
    y0, x = m.data[s], m.torsion_x[s]
    y = view(fws.y, 1:length(y0))
    r = deg2rad(ω)
    @inbounds for i in eachindex(y0)
        y[i] = y0[i] - r * x[i]
    end
    return y
end

# log-likelihood of stem s at the parameters p (`stem_params`) and twist rate ω; with gθ, its gradient in the model
# parameters θ, times wgt, added to gθ
function _stem_ll!(m, p, s::Int, ω::Real, fws::StemWorkspace, gθ = nothing, wgt = 1; active = nothing)
    y = _corrected!(fws, m, s, ω)
    with_grad = gθ !== nothing
    with_grad && fill!(fws.grad, 0)
    ℓ = stem_loglik!(fws, m.grid, y, p.η, p.logμκ, p.σκ, p.α, with_grad; outlier = p.outlier, active)
    with_grad && accumulate_gradθ!(gθ, m, fws.grad, wgt)
    return ℓ
end

# the tag of the duals of the Hessian passes
struct AngleHessianTag end

# The dual (Hessian) passes cost ~ (number of parameters + 1) Float64 passes. A Float64 pass at the same point marks the
# grid points within e^-40 of the heaviest one (`active`): the dual pass leaves out the others, whose total weight is
# below (number of grid points) × e^-40 of the stem's likelihood: 4 ngrid_ϕ ngrid_κ = 2048 points with the default grid,
# so below 1e-14.
const ACTIVE_DROP = 40.0
function _mark_active!(active, fws64::StemWorkspace{Float64}, m, pf, s::Int, ω::Real)
    _stem_ll!(m, pf, s, ω, fws64)
    (; κ_nodes, κ_logweights) = m.grid
    n = length(m.data[s])
    off = [κ_logweights[q] - n * logbesseli(0, exp(pf.logμκ + pf.σκ * z)) for (q, z) in enumerate(κ_nodes)]
    top = maximum(fws64.log_Z[c, q] + off[q] for c in axes(active, 1), q in axes(active, 2))
    for q in axes(active, 2), c in axes(active, 1)
        active[c, q] = fws64.log_Z[c, q] + off[q] ≥ top - ACTIVE_DROP
    end
    return active
end
# the stem log-likelihood and its gradient in a dual workspace, on the grid points active at the Float64 parameters pf
function _stem_ll_dual!(m, p, pf, s::Int, ω::Real, fwd::StemWorkspace, gθ)
    _with_workspace((_, fws64) -> _mark_active!(fwd.active, fws64, m, pf, s, ω), m.pool, m, s)
    return _stem_ll!(m, p, s, ω, fwd, gθ; active = fwd.active)
end
# (only the package's own Hessian duals, over Float64 values: other number types, e.g. ForwardDiff run by a user over
# the log density, take the full grid)
const HessianDual = ForwardDiff.Dual{AngleHessianTag, Float64}
_stem_ll_any!(m, p, pf, s, ω, fws::StemWorkspace, gθ) = _stem_ll!(m, p, s, ω, fws, gθ)
_stem_ll_any!(m, p, pf, s, ω, fws::StemWorkspace{<:HessianDual}, gθ) = _stem_ll_dual!(m, p, pf, s, ω, fws, gθ)

_rate(ωs::Nothing, s) = 0.0
_rate(ωs::AbstractVector, s) = ωs[s]

# Σ over stems of the log-likelihood at fixed twist rates ωs (nothing: no twist) and its gradient in the model part of
# u, threaded, generic in eltype(u): the per-stem gradients are summed in θ and the chain rule applied once
function _data_value_grad(m, u::AbstractVector{T}, ωs, with_grad::Bool) where {T}
    G = nmodel(m)
    p = stem_params(m, u)
    pf = (T <: HessianDual ? stem_params(m, ForwardDiff.value.(u)) : p)
    parts = _per_stem(m, T, Tuple{T, Vector{T}}) do s, fws
        gθ = zeros(T, with_grad ? G : 0)
        ℓ = _stem_ll_any!(m, p, pf, s, _rate(ωs, s), fws, (with_grad ? gθ : nothing))
        (ℓ, gθ)
    end
    l = sum(first, parts)
    with_grad || return l, nothing
    return l, params_jacobian(m, u)' * sum(last, parts)
end

# ForwardDiff gradient and Hessian of the (cheap) prior terms; the chunk size is chosen at run time, so the result types
# are asserted
_grad(f, v::AbstractVector) = ForwardDiff.gradient(f, v)::Vector{Float64}
_hess(f, v::AbstractVector) = ForwardDiff.hessian(f, v)::Matrix{Float64}

# seed v with the identity: duals whose partials are the derivatives in v
_seed(v) = (P = length(v); [ForwardDiff.Dual{AngleHessianTag, Float64, P}(v[i], ForwardDiff.Partials(ntuple(j -> Float64(i == j), P))) for i in 1:P])
_hessian_from(gd) = [ForwardDiff.partials(gd[i], j) for i in eachindex(gd), j in eachindex(gd)]

"""
    angle_vgh(m, u, ωs, order)

Log posterior of the model parameters u at fixed twist rates ωs (`nothing`: no twist), with its gradient (order ≥ 1,
analytic) and Hessian (order 2: one dual pass over the analytic gradient).
"""
function angle_vgh(m, u::AbstractVector, ωs, order::Int)
    lp = w -> model_logprior(m, w)
    if order == 0
        return (_data_value_grad(m, u, ωs, false)[1] + lp(u), nothing, nothing)
    elseif order == 1
        l, g = _data_value_grad(m, u, ωs, true)
        return (l + lp(u), g + _grad(lp, u), nothing)
    end
    return _angle_vgh2(m, _seed(u), u, ωs, lp)
end
# the Hessian pass behind a function barrier (the dual width is the number of parameters, known at run time)
function _angle_vgh2(m, ud::Vector{<:ForwardDiff.Dual}, u, ωs, lp)
    ld, gd = _data_value_grad(m, ud, ωs, true)
    H = _hessian_from(gd) + _hess(lp, u)
    return (ForwardDiff.value(ld) + lp(u), ForwardDiff.value.(gd) + _grad(lp, u), (H + H') / 2)
end

# node terms a_j = log w_j + ℓ(ω_j) + log N(ω_j | m_ω, s_ω) of stem s (Float64), at the parameters p and v = [u; m_ω; log s_ω]
function _node_terms(m, p, v, s::Int, nodes::TorsionNodes, fws)
    G = nmodel(m)
    ℓ = [_stem_ll!(m, p, s, ω, fws) for ω in nodes.ω[s]]
    return node_logterms(nodes.ω[s], nodes.logw[s], ℓ, v[G + 1], exp(v[G + 2]))
end

# integrated log-likelihood of stem s over its twist nodes in `keep`, at v = [u; m_ω; log s_ω] (generic in eltype(v)),
# and its gradient: in θ (the chain rule is applied by the caller), and in (m_ω, log s_ω)
function _stem_integrated(m, p, pf, v::AbstractVector{T}, s::Int, nodes::TorsionNodes, keep, fws) where {T}
    G = nmodel(m)
    mω, sω = v[G + 1], exp(v[G + 2])
    ω, logw = nodes.ω[s][keep], nodes.logw[s][keep]
    gs = [zeros(T, G) for _ in eachindex(ω)]
    ℓ = [_stem_ll_any!(m, p, pf, s, ω[j], fws, gs[j]) for j in eachindex(ω)]
    a = node_logterms(ω, logw, ℓ, mω, sω)
    L = logsumexp(a)
    gθ, gt = zeros(T, G), zeros(T, 2)
    for j in eachindex(ω)
        pw = exp(a[j] - L) # posterior weight of the node
        gθ .+= pw .* gs[j]
        r = (ω[j] - mω) / sω
        gt[1] += pw * r / sω
        gt[2] += pw * (r^2 - 1)
    end
    return L, gθ, gt
end

"""
    twist_vgh(m, nodes, v, order)

Log posterior with the twist rates integrated out, `v = [u; m_ω; log s_ω]`, up to the Hessian (one dual pass per stem over
its analytic gradient; nodes below e^-30 of the best skipped in the derivative passes).
"""
function twist_vgh(m, nodes::TorsionNodes, v::AbstractVector, order::Int)
    G = nmodel(m)
    u = v[1:G]
    p = stem_params(m, u)
    lp = w -> model_logprior(m, w[1:G]) + torsion_hyperprior(m.torsion, w[G + 1], w[G + 2])
    if order == 0
        L = _per_stem((s, fws) -> logsumexp(_node_terms(m, p, v, s, nodes, fws)), m, Float64, Float64)
        return (lp(v) + sum(L), nothing, nothing)
    elseif order == 1
        parts = _per_stem(m, Float64, Tuple{Float64, Vector{Float64}, Vector{Float64}}) do s, fws
            a = _node_terms(m, p, v, s, nodes, fws)
            _, gθ, gt = _stem_integrated(m, p, p, v, s, nodes, a .≥ maximum(a) - 30, fws)
            (logsumexp(a), gθ, gt)
        end
        g = vcat(params_jacobian(m, u)' * sum(q -> q[2], parts), sum(q -> q[3], parts))
        return (lp(v) + sum(first, parts), g + _grad(lp, v), nothing)
    end
    return _twist_vgh2(m, nodes, v, _seed(v), p, lp)
end
# the Hessian pass behind a function barrier (dual width known at run time); the value and the choice of nodes from
# Float64 passes, the derivatives from a dual pass on the kept nodes
function _twist_vgh2(m, nodes::TorsionNodes, v, vd::Vector{Td}, p, lp) where {Td <: ForwardDiff.Dual}
    G = nmodel(m)
    pd = stem_params(m, vd[1:G])
    dpool = _pool(m, Td)
    parts = _per_stem(m, Float64, Tuple{Float64, Vector{Td}, Vector{Td}}) do s, fws
        a = _node_terms(m, p, v, s, nodes, fws)
        keep = a .≥ maximum(a) - 30
        _, gθ, gt = _with_workspace((_, fwd) -> _stem_integrated(m, pd, p, vd, s, nodes, keep, fwd), dpool, m, s)
        (logsumexp(a), gθ, gt)
    end
    gd = vcat(params_jacobian(m, vd[1:G])' * sum(q -> q[2], parts), sum(q -> q[3], parts))
    H = _hessian_from(gd) + _hess(lp, v)
    return (lp(v) + sum(first, parts), ForwardDiff.value.(gd) + _grad(lp, v), (H + H') / 2)
end

#################################################################################################################
# fitting
#################################################################################################################

"""
    PhyllotaxisFit

The result of `fit_map`: the MAP of a `PhyllotaxisModelAD` and what is needed to use and check it.

- `model`: the fitted model; `u`: the MAP in the unconstrained space (`[model parameters; m_ω; log s_ω]` with a twist);
  `θ`: the same as a NamedTuple of natural parameters (`constrained_parameters`).
- `H`: Hessian of the log posterior in u at the MAP (`-inv(H)`: the Laplace covariance of u);
  `logposterior`, `loglik` (the log-likelihood, priors excluded, twist rates integrated out) and `stem_loglik` (its
  terms, one per stem), `gradient` (of the log posterior in u at the MAP).
- `converged`, `iterations`, `result` (the Optim result).
- `twist`: `nothing` without twist; otherwise per stem the most probable rate `ω` (the centre of its heaviest mode),
  the posterior mean and sd of the rate (`ω_mean`, `ω_sd`, aliases included), the posterior mass of the heaviest
  mode (`mode_mass`), and the quadrature `nodes`.

Use it with `decode_orders`, `order_posterior`, `spiral_posterior`, `stem_loglik`, `corrected_angles`, and the
StatsBase accessors `coef`, `coefnames`, `vcov`, `stderror`, `coeftable`, `loglikelihood`, `dof`, `nobs`.
`show` prints the convergence checks and the parameter table.
"""
struct PhyllotaxisFit{M, Θ <: NamedTuple, W, R}
    model::M
    u::Vector{Float64}
    θ::Θ
    H::Matrix{Float64}
    logposterior::Float64
    loglik::Float64
    stem_loglik::Vector{Float64}
    gradient::Vector{Float64}
    converged::Bool
    iterations::Int
    twist::W
    result::R
end
function PhyllotaxisFit(m, nf, stem_ll::Vector{Float64}, twist)
    return PhyllotaxisFit(
        m, nf.v, constrained_parameters(m, nf.v), nf.H, nf.l, sum(stem_ll), stem_ll, nf.g, Optim.converged(nf.result),
        Optim.iterations(nf.result), twist, nf.result
    )
end

# Default convergence of the Newton fits (`fit_map(model; g_tol, iterations)`). The fit stops when no component of
# the gradient of the log posterior in u exceeds g_tol. The u left is then off the maximum by about H⁻¹ g, i.e. by
# g_tol × the posterior variance of u, and the log posterior by about ½ gᵀ H⁻¹ g: on usual data the posterior variances
# are small (sds of 0.01–1 in u), so stopping there is inconsequential, and Newton's quadratic convergence usually
# overshoots it anyway (final gradients of 1e-8 and below). The fits take 1–15 Newton steps: `iterations` is a safety
# cap, and a fit that reaches it is reported as not converged.
const MAP_G_TOL = 1.0e-5
const MAP_ITERATIONS = 100
map_options(; g_tol::Real = MAP_G_TOL, iterations::Integer = MAP_ITERATIONS) = Optim.Options(; g_tol, iterations)

"""
    newton_map(vgh, v0; options)

Maximise f by Newton trust region, `vgh(v, order)` returning (f, ∇f or nothing, ∇²f or nothing) up to `order`. A
non-finite evaluation is a rejected step. Returns (; v, l, g, H, result): the maximiser, and f, ∇f and ∇²f there (the
last Hessian evaluation, re-evaluated if it was not at the maximiser).
"""
function newton_map(vgh, v0::AbstractVector; options = map_options())
    P = length(v0)
    last_v, last_l, last_g, last_H = fill(NaN, P), NaN, zeros(P), zeros(P, P)
    fgh! = (F, G, H, v) -> begin
        order = (H !== nothing ? 2 : G !== nothing ? 1 : 0)
        l, g, h = vgh(v, order)
        ok = isfinite(l) && (order < 1 || all(isfinite, g)) && (order < 2 || all(isfinite, h))
        if order ≥ 2
            H .= (ok ? .-(h .+ h') ./ 2 : 0.0)
            if ok
                last_v .= v
                last_l = l
                last_g .= g
                last_H .= (h .+ h') ./ 2
            end
        end
        (G !== nothing) && (G .= (ok ? .-g : 0.0))
        return ok ? -l : Inf
    end
    res = optimize(NLSolversBase.only_fgh!(fgh!), collect(Float64, v0), NewtonTrustRegion(), options)
    v = Optim.minimizer(res)
    if v != last_v
        l, g, h = vgh(v, 2)
        last_l, last_g, last_H = l, collect(Float64, g), (h .+ h') ./ 2
    end
    return (; v, l = last_l, g = last_g, H = last_H, result = res)
end

_fixed_fit(m, ωs, u0, options) = newton_map((u, order) -> angle_vgh(m, u, ωs, order), u0; options)

# quadrature nodes of every stem's twist rate at the model parameters u
function _place_nodes(m, u; kw...)
    p = stem_params(m, map(Float64, u))
    res = _per_stem(m, Float64) do s, fws
        stem_nodes(ω -> _stem_ll!(m, p, s, ω, fws), torsion_step(m.torsion_x[s]); kw...)
    end
    return TorsionNodes(getindex.(res, 1), getindex.(res, 2), getindex.(res, 3))
end

# the MAP of a model without twist
function _fit_no_twist(m, u0, options)
    nf = _fixed_fit(m, nothing, u0, options)
    return PhyllotaxisFit(m, nf, _stem_values(m, nf.v, nothing), nothing)
end

# the MAP with the twist rates integrated out: fit without twist; nodes; fit with each rate at its best mode; nodes;
# integrated fit (Newton on `twist_vgh`); `recenter` times: nodes at the new fit and refit. The nodes of the fit are kept
# in the model (its log density is then the objective of the fit).
function _fit_twist(m, u0, options, recenter::Int, nodes_kw)
    G = nmodel(m)
    u = _fixed_fit(m, nothing, u0[1:G], options).v
    nodes = _place_nodes(m, u; nodes_kw...)
    ωs = [first(md)[1] for md in nodes.modes] # best mode of each stem
    u = _fixed_fit(m, ωs, u, options).v
    nodes = _place_nodes(m, u; nodes_kw...)
    v0 = (length(u0) == G + 2 ? vcat(u, u0[(G + 1):end]) : vcat(u, mean(ωs), log(length(ωs) > 1 ? max(std(ωs), 0.1) : 0.1)))
    nf = nothing
    for r in 0:recenter
        nf = newton_map((v, order) -> twist_vgh(m, nodes, v, order), v0; options)
        v0 = nf.v
        r < recenter && (nodes = _place_nodes(m, nf.v[1:G]; nodes_kw...))
    end
    v = nf.v
    p = stem_params(m, v[1:G])
    terms = _per_stem((s, fws) -> _node_terms(m, p, v, s, nodes, fws), m, Float64, Vector{Float64})
    post = [node_posterior(nodes.ω[s], a, nodes.modes[s]) for (s, a) in enumerate(terms)]
    m.nodes[] = nodes
    twist = (;
        ω = getindex.(post, 3), ω_mean = getindex.(post, 1), ω_sd = getindex.(post, 2), mode_mass = getindex.(post, 4),
        nodes,
    )
    return PhyllotaxisFit(m, nf, map(logsumexp, terms), twist)
end

#################################################################################################################
# log-likelihoods at given parameters
#################################################################################################################

# the twist nodes to integrate with: nothing for a model without twist, else those of the fit (or of the model, for
# parameters given as a vector)
_nodes(m, nodes) = nodes
function _nodes(m)
    _has_twist(m) || return nothing
    nodes = m.nodes[]
    nodes === nothing && throw(ArgumentError("the twist quadrature nodes are placed by fit_map: fit the model first"))
    return nodes
end
_nodes(fit::PhyllotaxisFit) = (fit.twist === nothing ? nothing : fit.twist.nodes)

# log-likelihood of stem s at u: with nodes, the twist rate integrated out; ω given: at that fixed rate
function _stem_value(m, p, u, s, nodes, fws, ω)
    ω === nothing || return _stem_ll!(m, p, s, ω, fws)
    nodes === nothing && return _stem_ll!(m, p, s, 0.0, fws)
    return logsumexp(_node_terms(m, p, u, s, nodes, fws))
end
function _stem_values(m, u, nodes, ω = nothing)
    uf = map(Float64, u)
    p = stem_params(m, uf)
    return _per_stem((s, fws) -> _stem_value(m, p, uf, s, nodes, fws, ω), m, Float64, Float64)
end

"""
    stem_loglik(fit)
    stem_loglik(model, u, s; ω)

Log-likelihood of each stem at a fit (priors excluded, twist rate integrated out), or of stem s alone at the
unconstrained parameters u: with a twist, integrated over the model's quadrature nodes (placed by `fit_map`), or at a
fixed rate ω (degrees per unit of the coordinate) if given.
"""
stem_loglik(fit::PhyllotaxisFit) = fit.stem_loglik
function stem_loglik(m, u::AbstractVector, s::Int; ω::Union{Nothing, Real} = nothing)
    uf = map(Float64, u)
    nodes = (ω === nothing ? _nodes(m) : nothing)
    return _with_workspace((_, fws) -> _stem_value(m, stem_params(m, uf), uf, s, nodes, fws, ω), m.pool, m, s)
end

"""
    angle_loglikelihood(model, u)

Total log-likelihood (priors excluded) at the unconstrained parameters u, the twist rates integrated out over the
model's quadrature nodes (placed by `fit_map`). At a fit: `loglikelihood(fit)`.
"""
angle_loglikelihood(m, u::AbstractVector) = sum(_stem_values(m, u, _nodes(m)))

#################################################################################################################
# decoding: the organ order of each stem
#################################################################################################################

corrected_angles(m, s::Int; ω::Real = 0.0) = torsion_corrected(m.data[s], m.torsion_x[s], ω)
corrected_angles(fit::PhyllotaxisFit, s::Int) = corrected_angles(fit.model, s; ω = (fit.twist === nothing ? 0.0 : fit.twist.ω[s]))

# State of stem s at twist rate ω: the cosine table of the nuisance combinations stays in fws; the stem log-likelihood
# (lc), the log posterior of each nuisance combination × κ node (P, Nc × ngrid_κ, normalised) and log perm(e^η) (lperm);
# per κ node, κ, its outlier terms (`_outlier_terms`) and n log I₀(κ). `_refresh_cosines!(fws, st)` restores the cosine
# table of a state after other twist rates were evaluated.
function _stem_state(m, p, s::Int, ω::Real, fws::StemWorkspace)
    grid = m.grid
    y = _corrected!(fws, m, s, ω)
    n = length(y)
    κs = [exp(p.logμκ + p.σκ * z) for z in grid.κ_nodes]
    ots = [_outlier_terms(p.outlier, κ) for κ in κs]
    nlI0 = [n * logbesseli(0, κ) for κ in κs]
    ℓ = stem_loglik!(fws, grid, y, p.η, p.logμκ, p.σκ, p.α, false; outlier = p.outlier)
    B = [fws.log_Z[j, q] + grid.κ_logweights[q] - nlI0[q] for j in axes(fws.log_Z, 1), q in 1:grid.ngrid_κ]
    lB = logsumexp(B)
    P = B .- lB
    lperm = lB - log(2 * grid.ngrid_ϕ) - n * log(2π) - ℓ
    return (; y = collect(y), n, η = p.η, lc = ℓ, P, lperm, p, grid, κs, ots, nlI0)
end
_refresh_cosines!(fws::StemWorkspace, st) = _fill_cosines!(fws, st.grid, st.y, st.p.α)

# twist rule of stem s: the values of its rate and their log weights (with the population law at v)
# each stem at a fixed twist rate of its own (the keyword ω of the decoding functions)
struct FixedRates
    ω::Vector{Float64}
end
_rule(m, ω::Nothing) = _nodes(m)
function _rule(m, ω::AbstractVector)
    length(ω) == length(m.data) || throw(DimensionMismatch("ω: one twist rate per stem"))
    return FixedRates(collect(Float64, ω))
end
_twist_rule(m, v, s::Int, r::FixedRates) = ([r.ω[s]], [0.0])
function _twist_rule(m, v, s::Int, nodes)
    nodes === nothing && return ([0.0], [0.0])
    G = nmodel(m)
    return nodes.ω[s], node_logterms(nodes.ω[s], nodes.logw[s], zeros(length(nodes.ω[s])), v[G + 1], exp(v[G + 2]))
end

"""
    order_posterior(fit; nsamples = 0, rng, prune = 1e-8)
    order_posterior(model, u; ω, nsamples = 0, rng, prune = 1e-8)

Exact posterior of the organ order of each stem, summed over the nuisance grid (ϕ, o, ψ, κ node) and the twist rate
(its quadrature nodes, weighted by the fitted population law). Per stem: (; ρ, samples, `ω_mean`, `ω_sd`), with
ρ[i, d + k + 1] = P(σ[i] = i + d | angles) (the organ observed at position i is canonical organ i + d) and `nsamples`
exact draws of σ (a nuisance state drawn from its posterior, then σ by forward filtering, backward sampling).
States of posterior weight below `prune` are skipped (their mass is renormalised away). Given parameters u, `ω` (one rate per stem, degrees per unit of the
coordinate) fixes each stem's twist instead of integrating it.
"""
order_posterior(fit::PhyllotaxisFit; kw...) = _order_posterior(fit.model, fit.u, _nodes(fit); kw...)
order_posterior(m, u::AbstractVector; ω = nothing, kw...) = _order_posterior(m, u, _rule(m, ω); kw...)
function _order_posterior(m, u, nodes; nsamples::Int = 0, rng::AbstractRNG = Random.default_rng(), prune::Real = 1.0e-8)
    k = m.k
    uf = map(Float64, u)
    p = stem_params(m, uf)
    seeds = rand(rng, UInt64, length(m.data))
    return _per_stem(m, Float64) do s, fws
        srng = Xoshiro(seeds[s])
        ω, lpω = _twist_rule(m, uf, s, nodes)
        sts = [_stem_state(m, p, s, ω[j], fws) for j in eachindex(ω)]
        lj = [lpω[j] + sts[j].lc for j in eachindex(ω)]
        Pj = exp.(lj .- logsumexp(lj))
        # every retained state (node j, combination, κ node q) with its posterior weight
        n = length(m.data[s])
        ρ = zeros(n, 2k + 1)
        states, wts = Tuple{Int, Int, Int}[], Float64[]
        for j in eachindex(ω)
            Pj[j] < prune && continue
            st = sts[j]
            _refresh_cosines!(fws, st)
            for q in 1:st.grid.ngrid_κ, cb in axes(st.P, 1)
                wgt = Pj[j] * exp(st.P[cb, q])
                wgt < prune && continue
                _fill_weights!(fws.perm.w, st.η, fws.C, cb, st.κs[q], n, k, st.ots[q])
                forward!(fws.perm, n)
                ρ .+= wgt .* backward!(fws.perm, n)
                push!(states, (j, cb, q)); push!(wts, wgt)
            end
        end
        ρ ./= sum(wts)
        samples = Vector{Vector{Int}}()
        if nsamples > 0
            cw = cumsum(wts)
            draws = sort([states[min(searchsortedfirst(cw, rand(srng) * cw[end]), length(cw))] for _ in 1:nsamples])
            jlast = 0
            for (j, cb, q) in draws # sorted by twist node: one cosine table per node
                st = sts[j]
                (j != jlast) && (_refresh_cosines!(fws, st); jlast = j)
                _fill_weights!(fws.perm.w, st.η, fws.C, cb, st.κs[q], n, k, st.ots[q])
                push!(samples, sample_permutation!(srng, fws.perm, n))
            end
            shuffle!(srng, samples)
        end
        ωm = sum(Pj .* ω)
        (; ρ, samples, ω_mean = ωm, ω_sd = sqrt(max(sum(Pj .* (ω .- ωm) .^ 2), 0.0)))
    end
end

# best (order, nuisance state) of stem s at twist rate ω: max over the nuisance combinations and the κ nodes of the
# joint log posterior (Viterbi on each combination within `Δ` of the best by marginal weight)
function _joint_best(m, p, s::Int, ω::Real, lpω::Real, fws::StemWorkspace; Δ::Real = 30.0)
    st = _stem_state(m, p, s, ω, fws)
    (; n, grid) = st
    k = grid.k
    best = (; score = -Inf, σ = collect(1:n), ω, ϕ = 0.0, o = 1, ψ = Golden_angle, κ = 0.0)
    top = maximum(st.P)
    for q in 1:grid.ngrid_κ, cb in axes(st.P, 1)
        (st.P[cb, q] < top - Δ) && continue
        κ = st.κs[q]
        cmb = _combo(grid, cb, p.α)
        w = _fill_weights!(fws.perm.w, st.η, fws.C, cb, κ, n, k, st.ots[q])
        σ = map_permutation(fws.perm.g, w, n)
        v = sum(w[i, σ[i] - i + k + 1] for i in 1:n)
        # log p(σ, angles, state): edge weights + the constants removed from them, the state's prior (o: ½, ψ, ϕ: 1/ngrid_ϕ,
        # log κ: its log-normal density at the node), the normaliser of the costs, the twist's population law
        score = v + n * κ - st.nlI0[q] - n * log(2π) + cmb.lp - log(2 * grid.ngrid_ϕ) - 0.5 * grid.κ_nodes[q]^2 -
            st.lperm + lpω
        score > best.score && (best = (; score, σ, ω, cmb.ϕ, cmb.o, cmb.ψ, κ))
    end
    return best
end

"""
    spiral_posterior(fit)
    spiral_posterior(model, u; ω)

Posterior of the spiral of each stem, summed over the order, ϕ, κ and the twist rate: one (; `p_lucas`, `p_ccw`) per
stem, P(ψ = Lucas angle | angles) and P(o = +1 | angles) (o = +1: the site angles increase, a counter-clockwise
spiral). Only forward passes (cheaper than `order_posterior`). Given parameters u, `ω` (one rate per stem, degrees per unit of the
coordinate) fixes each stem's twist instead of integrating it.
"""
spiral_posterior(fit::PhyllotaxisFit) = _spiral_posterior(fit.model, fit.u, _nodes(fit))
spiral_posterior(m, u::AbstractVector; ω = nothing) = _spiral_posterior(m, u, _rule(m, ω))
function _spiral_posterior(m, u, nodes)
    uf = map(Float64, u)
    p = stem_params(m, uf)
    return _per_stem(m, Float64) do s, fws
        ω, lpω = _twist_rule(m, uf, s, nodes)
        L = fill(-Inf, 2, 2) # [ψ golden / Lucas, o = −1 / +1]
        for j in eachindex(ω)
            st = _stem_state(m, p, s, ω[j], fws)
            for q in 1:st.grid.ngrid_κ, cb in axes(st.P, 1)
                cmb = _combo(st.grid, cb, p.α)
                iψ, io = (cmb.ψ == Golden_angle ? 1 : 2), (cmb.o == -1 ? 1 : 2)
                L[iψ, io] = logaddexp(L[iψ, io], lpω[j] + st.lc + st.P[cb, q])
            end
        end
        P = exp.(L .- logsumexp(L))
        (; p_lucas = P[2, 1] + P[2, 2], p_ccw = P[1, 2] + P[2, 2])
    end
end

"""
    decode_orders(fit; full = false, refine = true)
    decode_orders(model, u; ω, full = false, refine = true)

Most likely organ order of each stem: the joint maximum a posteriori of (σ, ϕ, o, ψ, κ node, twist rate). With a
twist, the rate is decoded with the order (prior: the fitted population law `N(m_ω, s_ω²)`): the best quadrature node,
refined by golden section within the sd of its mode (`refine`), since a different twist can change the order.
Returns the vector of orders σ (σ[i] = canonical index of the organ at position i), or with `full = true` one
NamedTuple per stem (; σ, ω, ϕ, o, ψ, κ, score). Given parameters u, `ω` (one rate per stem, degrees per unit of the
coordinate) fixes each stem's twist instead of integrating it.
"""
decode_orders(fit::PhyllotaxisFit; kw...) = _decode_orders(fit.model, fit.u, _nodes(fit); kw...)
decode_orders(m, u::AbstractVector; ω = nothing, kw...) = _decode_orders(m, u, _rule(m, ω); kw...)
function _decode_orders(m, u, nodes; full::Bool = false, refine::Bool = true)
    uf = map(Float64, u)
    p = stem_params(m, uf)
    G = nmodel(m)
    out = _per_stem(m, Float64) do s, fws
        nodes === nothing && return _joint_best(m, p, s, 0.0, 0.0, fws)
        nodes isa FixedRates && return _joint_best(m, p, s, nodes.ω[s], 0.0, fws)
        mω, sω = uf[G + 1], exp(uf[G + 2])
        f(ω) = _joint_best(m, p, s, ω, -0.5 * ((ω - mω) / sω)^2, fws)
        cand = [f(ω) for ω in unique(nodes.ω[s])]
        best = cand[argmax([c.score for c in cand])]
        if refine
            modes = nodes.modes[s]
            h = modes[argmin([abs(md[1] - best.ω) for md in modes])][2]
            _, refined = golden_section_max(f, best.ω - h, best.ω + h; tol = 1.0e-5 * h, score = r -> r.score)
            refined.score > best.score && (best = refined)
        end
        best
    end
    return full ? out : [b.σ for b in out]
end
