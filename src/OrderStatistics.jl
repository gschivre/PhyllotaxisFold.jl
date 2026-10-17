"""
    OrderStatistics

Level-0 organ order: site i becomes ready at iγ (readiness spacing γ in passage sd units) and commits after an i.i.d.
passage z ~ g, so the organ order is σ = sortperm(iγ + z_i). This module gives

- `displacement_marginal`: the exact stationary law of the displacement d = σ[p] − p of a bulk organ
- `match_propensity`: the Toeplitz log-propensities η whose permanent model has these bulk marginals (KL projection)
- `propensity_table` / `propensity!`: η tabulated on a (shape, log γ) grid and its bicubic interpolation with Jacobian
- `loglik_order_exact`: the exact Level-0 log P(σ) (reference for the permanent approximation)
- `build_tables` / `build_ramp_tables`, `save_tables` / `load_tables`: cached passage + propensity tables
"""
module OrderStatistics

using Statistics, Serialization, LinearAlgebra
using LogExpFunctions
using ..Common: _keys_w, _keys_dw, _keys_index
using ..Permanents: PermanentWorkspace, toeplitz_weights!, forward!, backward!
using ..PassageTime: PassageTable, passage_table, ramp_passage_table, _interp_z, _logS_z, _table_dict, _table_from_dict

export displacement_marginal, zero_mean_tilt, gauge!, match_propensity, PropensityTable, propensity_table, propensity!,
       loglik_order_exact, order_diagnostics, build_tables, build_ramp_tables, save_tables, load_tables

"""
    displacement_marginal(tab, iμ, γ, k; Dmax, J)

Exact stationary distribution of the displacement d = σ[p] - p of a bulk organ at Level 0 (readiness
spacing γ in sd units, i.i.d. passage z ~ tab[:, iμ]). Returns the probabilities of d = -k..k.
Given its own passage z, a later site j commits before it with prob F(z - jγ) and an earlier site j
commits after it with prob S(z + jγ); the rank shift is a difference of Poisson-binomial counts.
"""
function displacement_marginal(tab::PassageTable, iμ::Int, γ::Real, k::Int; Dmax::Int = k + 3, J::Int = 40)
    (; z_grid, logg, logS, logF) = tab
    colg, colS, colF = view(logg, :, iμ), view(logS, :, iμ), view(logF, :, iμ)
    W = 2 * Dmax + 3 # bins: -Dmax-1 (overflow) ... Dmax+1 (overflow)
    P = zeros(W)
    Q = zeros(W)
    out = zeros(W)
    Δz = step(z_grid)
    gmax = maximum(colg)
    for (iz, z) in enumerate(z_grid)
        lg = colg[iz]
        (lg < gmax - 50) && continue
        fill!(P, 0.0)
        P[Dmax + 2] = 1.0
        for j in 1:J
            a = exp(_interp_z(colF, z_grid, z - j * γ)) # later site j overtakes: rank +1
            b = exp(_logS_z(colS, z_grid, z + j * γ)) # earlier site j falls behind: rank -1
            (a < 1e-300 && b < 1e-300) && continue
            fill!(Q, 0.0)
            for r in 1:W
                p = P[r]
                (p == 0.0) && continue
                rp = clamp(r + 1, 1, W)
                rm = clamp(r - 1, 1, W)
                if r == 1 || r == W # overflow bins are absorbing
                    Q[r] += p
                    continue
                end
                Q[r] += p * ((1 - a) * (1 - b) + a * b)
                Q[rp] += p * a * (1 - b)
                Q[rm] += p * (1 - a) * b
            end
            P, Q = Q, P
        end
        out .+= exp(lg) .* P
    end
    out .*= Δz
    # rank shift D = r - (Dmax + 2) maps to displacement d = -D
    pd = zeros(2 * k + 1)
    for d in -k:k
        pd[d + k + 1] = out[Dmax + 2 - d]
    end
    return pd
end

"exponential tilt t_d ∝ p_d e^{a d} with zero mean (the tilt is the gauge of the permanent model)"
function zero_mean_tilt(p::AbstractVector, k::Int)
    ds = -k:k
    f(a) = sum(p[i] * exp(a * ds[i]) * ds[i] for i in eachindex(p))
    a_lo, a_hi = -5.0, 5.0
    for _ in 1:200
        a = 0.5 * (a_lo + a_hi)
        (f(a) > 0) ? (a_hi = a) : (a_lo = a)
    end
    a = 0.5 * (a_lo + a_hi)
    t = [p[i] * exp(a * ds[i]) for i in eachindex(p)]
    return t ./ sum(t)
end

"""
    gauge!(η, k)

Remove the two gauge directions of the Toeplitz permanent model: a constant shift (η_0 = 0) and a linear tilt
η_d → η_d + a d, which leaves every permutation probability unchanged because Σ_p d_p = 0 (fixed by η_1 = η_{−1}).
"""
function gauge!(η::AbstractVector, k::Int)
    η0 = η[k + 1]
    a = 0.5 * (η[k + 2] - η[k])
    for d in -k:k
        η[d + k + 1] -= η0 + a * d
    end
    return η
end

"""
    match_propensity(target, k; n, trim, maxiter, tol)

Find the Toeplitz log-propensities η whose bulk displacement marginals in the banded-permanent model
equal `target` (zero-mean distribution over d = -k..k). This is the moment-matching / maximum-entropy
condition, i.e. the KL projection of the true Level-0 order law onto the permanent family. Two phases:
damped iterative scaling η_d += λ log(t_d / m_d) (the damping λ starts at `damping` and is halved, down to 0.02,
whenever the error has not decreased for 50 iterations: heavy-tailed targets make the undamped scaling oscillate),
then, if it has not reached `tol` after `nscale` iterations, Gauss–Newton steps on r = log t − log m with a
finite-difference Jacobian (pseudo-inverse: the two gauge directions are its null space) and step halving, until
the least-squares gradient vanishes. At finite n the trimmed bulk marginals of the permanent are not exactly zero-mean,
so a zero-mean target is reached only up to a floor (≈ 1e-5–1e-4 at n = 60 for heavy-tailed targets, halving at
n = 100): the polish returns the least-squares optimum. Returns (η, iterations, err) with err the maximum
|log t_d − log m_d| (weighted 1e-3 where t_d ≤ 1e-12).
"""
function match_propensity(target::AbstractVector, k::Int; ws::PermanentWorkspace = PermanentWorkspace(k, 60), n::Int = ws.n_max, trim::Int = 2 * k, maxiter::Int = 3000, tol::Real = 1e-7, damping::Real = 0.8, floor_p::Real = 1e-250, nscale::Int = 300, η0 = nothing)
    K = 2 * k + 1
    t = max.(target, floor_p)
    lt = log.(t)
    wt = [t[d] > 1e-12 ? 1.0 : 1e-3 for d in 1:K]
    η = gauge!(η0 === nothing ? log.(t ./ t[k + 1]) : copy(η0), k)
    m = zeros(K)
    r = zeros(K)
    resid!(r, η) = (_bulk_marginals!(m, ws, η, n, trim, k); for d in 1:K; r[d] = wt[d] * (lt[d] - log(max(m[d], 1e-300))); end; maximum(abs, r))
    err = Inf
    best, stall, λ = Inf, 0, damping
    it = 0
    while it < min(nscale, maxiter)
        it += 1
        err = resid!(r, η)
        (err < tol) && return (η, it, err)
        for d in 1:K
            η[d] += λ * r[d] / wt[d]
        end
        gauge!(η, k)
        if err < best
            best, stall = err, 0
        elseif (stall += 1) ≥ 50
            λ, stall = max(λ / 2, 0.02), 0
        end
    end
    # Gauss–Newton polish (steps accepted on the L2 norm of r, the quantity it minimises)
    err = resid!(r, η)
    nr = norm(r)
    J = zeros(K, K)
    r1 = zeros(K)
    h = 1e-6
    while it < maxiter && err ≥ tol
        it += 1
        for j in 1:K
            η[j] += h
            resid!(r1, η)
            η[j] -= h
            for d in 1:K
                J[d, j] = (r1[d] - r[d]) / h
            end
        end
        norm(J' * r) < 1e-12 && break # least-squares optimum: the rest of r is out of reach at finite n
        Δ = -(pinv(J; rtol = 1e-5) * r) # drops the two gauge directions (singular values ~1e-7)
        α = 1.0
        accepted = false
        while α > 1e-4
            ηn = gauge!(η .+ α .* Δ, k)
            en = resid!(r1, ηn)
            if norm(r1) < nr
                η, err, nr, accepted = ηn, en, norm(r1), true
                r .= r1
                break
            end
            α /= 2
        end
        accepted || break
    end
    return (η, it, err)
end

# bulk displacement marginals m (positions trim+1..n−trim, normalised) of the banded permanent with log-propensities η
function _bulk_marginals!(m, ws, η, n, trim, k)
    toeplitz_weights!(ws, η, n)
    forward!(ws, n)
    rho = backward!(ws, n)
    fill!(m, 0.0)
    for i in (trim + 1):(n - trim), d_idx in 1:(2 * k + 1)
        m[d_idx] += rho[i, d_idx]
    end
    m ./= sum(m)
    return m
end

struct PropensityTable
    k::Int
    μ_grid::Vector{Float64} # shape grid (plateau μ or ramp log ν), the same as the passage table
    logγ_grid::Vector{Float64}
    η::Array{Float64, 3} # (2k + 1) × nμ × nγ
    pd::Array{Float64, 3} # true displacement marginals (before band renormalisation)
    maxerr::Float64
end

"""
    propensity_table(tab, k; logγ_grid, n)

Tabulate η(shape, log γ) on the shape grid of the passage table and a grid of log γ: exact marginals
(`displacement_marginal`), zero-mean tilt, then `match_propensity`. Threaded over grid points.
"""
function propensity_table(tab::PassageTable, k::Int; logγ_grid = collect(range(log(0.4), log(8.0); length = 49)), n::Int = 60)
    nμ = length(tab.μ_grid)
    nγ = length(logγ_grid)
    η = zeros(2 * k + 1, nμ, nγ)
    pd = zeros(2 * k + 1, nμ, nγ)
    errs = zeros(nμ, nγ)
    # one workspace per task (indexing buffers by threadid() is unsafe: a task may migrate between threads)
    chunks = Iterators.partition(1:(nμ * nγ), max(1, cld(nμ * nγ, 4 * Threads.nthreads())))
    @sync for idxs in chunks
        Threads.@spawn begin
            local ws = PermanentWorkspace(k, n)
            for idx in idxs
                iμ, iγ = Tuple(CartesianIndices((nμ, nγ))[idx])
                p = displacement_marginal(tab, iμ, exp(logγ_grid[iγ]), k)
                pd[:, iμ, iγ] .= p
                t = zero_mean_tilt(p ./ sum(p), k)
                ηm, _, err = match_propensity(t, k; ws, n)
                η[:, iμ, iγ] .= ηm
                errs[iμ, iγ] = err
            end
        end
    end
    return PropensityTable(k, copy(tab.μ_grid), collect(Float64, logγ_grid), η, pd, maximum(errs))
end

"""
    propensity!(η, ∂η, ptab, μ, logγ)

Bicubic interpolation of η(μ, log γ) and of its Jacobian ∂η[:, 1] = ∂η/∂μ, ∂η[:, 2] = ∂η/∂log γ (∂η = nothing to skip).
"""
function propensity!(η::AbstractVector, ∂η::Union{AbstractMatrix, Nothing}, ptab::PropensityTable, μ::Real, logγ::Real)
    (; μ_grid, logγ_grid) = ptab
    T = ptab.η
    nμ, nγ = length(μ_grid), length(logγ_grid)
    iμ, tμ, hμ = _keys_index(μ_grid, μ)
    iγ, tγ, hγ = _keys_index(logγ_grid, logγ)
    wμ, dwμ = _keys_w(tμ), _keys_dw(tμ)
    wγ, dwγ = _keys_w(tγ), _keys_dw(tγ)
    fill!(η, 0.0)
    (∂η !== nothing) && fill!(∂η, 0.0)
    for a in 1:4, b in 1:4
        ia = clamp(iμ + a - 1, 1, nμ) # grid nodes iμ - 1 .. iμ + 2 (1-based: iμ .. iμ + 3)
        ib = clamp(iγ + b - 1, 1, nγ)
        ω = wμ[a] * wγ[b]
        for d_idx in eachindex(η)
            η[d_idx] += ω * T[d_idx, ia, ib]
        end
        if ∂η !== nothing
            ωμ = dwμ[a] * wγ[b] / hμ
            ωγ = wμ[a] * dwγ[b] / hγ
            for d_idx in eachindex(η)
                ∂η[d_idx, 1] += ωμ * T[d_idx, ia, ib]
                ∂η[d_idx, 2] += ωγ * T[d_idx, ia, ib]
            end
        end
    end
    return η
end

"""
    loglik_order_exact(σ, logg, z_grid, γ)

Exact Level-0 log P(σ) for sites 1..n with readiness iγ and i.i.d. standardised passage density g:
P(σ) = ∫_{τ_1 < … < τ_n} ∏_p g(τ_p - σ[p] γ) dτ, by a forward recursion on a time grid with step Δz. Two times in the
same grid cell are ordered with probability ½ (midpoint rule): counting such ties as ordered biased log P(σ) upwards at
first order in Δz (Gaussian delays of sd 0.43 on a 0.05 grid: +0.025 for the identity of 6 sites, +0.28 for an order
with three displacements); the error is now second order (≤ 0.01 there).
"""
function loglik_order_exact(σ::AbstractVector{Int}, logg::AbstractVector, z_grid, γ::Real)
    n = length(σ)
    Δz = step(z_grid)
    zmin, zmax = first(z_grid), last(z_grid)
    tmin = γ * minimum(σ) + zmin
    L = round(Int, (γ * (maximum(σ) - minimum(σ)) + (zmax - zmin)) / Δz) + 2
    A = ones(L) # normalised cumulative of the previous forward message
    α = zeros(L)
    ll = 0.0
    for p in 1:n
        tp = γ * σ[p]
        fill!(α, 0.0)
        ka = max(1, ceil(Int, (tp + zmin - tmin) / Δz) + 1)
        kb = min(L, floor(Int, (tp + zmax - tmin) / Δz) + 1)
        @inbounds for kk in ka:kb
            α[kk] = exp(_interp_z(logg, z_grid, tmin + (kk - 1) * Δz - tp)) * A[kk]
        end
        c = sum(α) * Δz
        (c > 0) || return -Inf
        ll += log(c)
        acc = 0.0
        @inbounds for kk in 1:L
            a = α[kk] * Δz / c
            A[kk] = acc + a / 2 # P(previous time < a time in cell kk): the cells before, half of cell kk
            acc += a
        end
    end
    return ll
end

"""
    order_diagnostics(orders)

Fraction of displaced organs, and counts of the patterns 231 (one very late organ) and 312 (one very early organ) over
consecutive triplets of positions holding three consecutive canonical indices.
"""
function order_diagnostics(orders::AbstractVector)
    displaced = mean(mean(σ .!= eachindex(σ)) for σ in orders)
    n231 = n312 = 0
    for σ in orders, p in 1:(length(σ) - 2)
        a, b, c = σ[p], σ[p + 1], σ[p + 2]
        m = min(a, b, c)
        (max(a, b, c) == m + 2 && a + b + c == 3 * m + 3) || continue
        ((a, b, c) == (m + 1, m + 2, m)) && (n231 += 1)
        ((a, b, c) == (m + 2, m, m + 1)) && (n312 += 1)
    end
    return (; displaced, n231, n312)
end

#################################################################################################################
# cached tables
#################################################################################################################

"save (; tab, ptab) as plain arrays, independent of the struct definitions"
function save_tables(path::AbstractString, tabs)
    (; tab, ptab) = tabs
    serialize(path, Dict{Symbol, Any}(:tab => _table_dict(tab),
        :ptab => Dict{Symbol, Any}(:k => ptab.k, :μ_grid => ptab.μ_grid, :logγ_grid => ptab.logγ_grid, :η => ptab.η, :pd => ptab.pd, :maxerr => ptab.maxerr)))
end
"load tables written by `save_tables`: (; tab::PassageTable, ptab::PropensityTable)"
function load_tables(path::AbstractString)
    d = deserialize(path)
    p = d[:ptab]
    return (; tab = _table_from_dict(d[:tab]), ptab = PropensityTable(p[:k], p[:μ_grid], p[:logγ_grid], p[:η], p[:pd], p[:maxerr]))
end

"""
    build_tables(; k, μ_grid, logγ_grid, h, Δz, cache)

Plateau passage table and its propensity table, (; tab, ptab). With `cache`, read it if it matches the grids, otherwise
compute and write it (≈ 1 min with 18 threads).
"""
function build_tables(; k::Int = 4, μ_grid = collect(-1.0:0.25:10.0), logγ_grid = collect(range(log(0.4), log(8.0); length = 49)), h::Real = 0.02, Δz::Real = 0.04, cache::Union{String, Nothing} = nothing)
    if cache !== nothing && isfile(cache)
        tabs = load_tables(cache)
        (tabs.ptab.k == k && tabs.ptab.μ_grid == μ_grid && tabs.ptab.logγ_grid == logγ_grid) && return tabs
    end
    tab = passage_table(μ_grid; Δz = Δz, h = h)
    ptab = propensity_table(tab, k; logγ_grid = logγ_grid)
    tabs = (; tab, ptab)
    (cache !== nothing) && save_tables(cache, tabs)
    return tabs
end

"ramp family (model M2): passage and propensity tables on a grid of log ν (same caching as `build_tables`)"
function build_ramp_tables(; k::Int = 4, logν_grid = collect(-3.0:0.25:3.0), logγ_grid = collect(range(log(0.4), log(8.0); length = 49)), h::Real = 0.02, Δz::Real = 0.04, cache::Union{String, Nothing} = nothing)
    if cache !== nothing && isfile(cache)
        tabs = load_tables(cache)
        (tabs.ptab.k == k && tabs.ptab.μ_grid == logν_grid && tabs.ptab.logγ_grid == logγ_grid) && return tabs
    end
    tab = ramp_passage_table(logν_grid; Δz = Δz, h = h)
    ptab = propensity_table(tab, k; logγ_grid = logγ_grid)
    tabs = (; tab, ptab)
    (cache !== nothing) && save_tables(cache, tabs)
    return tabs
end

end # module OrderStatistics
