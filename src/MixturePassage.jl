"""
    MixturePassage

Model B-mix: the plateau margin varies from site to site. Site j has margin μ_j = μ̄ + s_μ ζ_j, ζ_j ~ N(0, 1), restricted
to [μ_c, μ_max] (a margin below μ_c, deep under the fold, means a wait of more than ≈ 20 plastochrons: such an organ falls
outside the stem window, and its mass is renormalised away). Given μ_j the delay follows the plateau passage law; the
delays stay i.i.d. across sites, with the *mixture* law

    g_mix(s; μ̄, s_μ) = ∫ g_μ(s) N(μ; μ̄, s_μ²) dμ / Z        (absolute noise time s, NOT standardised per μ)

so everything downstream of a passage law (displacement marginals, KL projection onto η, the angle model) is unchanged.
A fold turns a spread of margins into a heavy late tail (a margin near the fold commits late), which flattens the costs
of long displacements.

The mixture law is standardised by its median and a robust scale, IQR/1.349 (= sd for a Gaussian): with a heavy tail the
sd mostly measures the rare long delays, while the IQR keeps measuring the bulk. γ is the readiness spacing in that
scale. At s_μ = 0 the law is the plateau law at μ̄ (only the scale convention differs from model B).

- `mixture_passage_column(tab, μ̄, sμ)`: the standardised law as a one-column `PassageTable`
- `MixturePropensityTable`, `mixture_propensity_table`: η on a (μ̄, s_μ, log γ) grid
- `propensity!(η, ∂η, mtab, μ̄, sμ, logγ)`: tricubic (Keys) interpolation with its Jacobian (2k + 1) × 3
- `mixture_sampler`: draws of the delay (absolute noise time) for simulations
- `build_mixture_tables`, `save_mixture_tables`, `load_mixture_tables`
"""
module MixturePassage

using Random, Statistics, Serialization
using Distributions, LogExpFunctions
using ..Common: LOGFLOOR, _keys_w, _keys_dw, _keys_index
using ..Permanents: PermanentWorkspace
using ..PassageTime: PassageTable, passage_table, _interp_z, _table_dict, _table_from_dict
using ..OrderStatistics: displacement_marginal, zero_mean_tilt, match_propensity
import ..OrderStatistics: propensity!

export mixture_passage_column, MixturePropensityTable, mixture_propensity_table, mixture_sampler, build_mixture_tables,
       save_mixture_tables, load_mixture_tables, margin_weights, reproject, inband_mass

"""
    margin_weights(tab, μ̄, sμ)

Weights of the margins of the passage table `tab` (its μ grid, uniform) for the Normal(μ̄, s_μ²) margin restricted to
the grid range, normalised to 1 (s_μ should be at least the grid step). At s_μ = 0, the weight is shared linearly by
the two nodes around μ̄.
"""
function margin_weights(tab::PassageTable, μ̄::Real, sμ::Real)
    μs = tab.μ_grid
    if sμ == 0
        h = μs[2] - μs[1]
        w = zeros(length(μs))
        x = clamp((μ̄ - μs[1]) / h, 0.0, length(μs) - 1.0)
        i = min(floor(Int, x), length(μs) - 2)
        w[i + 1], w[i + 2] = 1 - (x - i), x - i
        return w
    end
    w = pdf.(Normal(μ̄, sμ), μs)
    return w ./ sum(w)
end

"""
    mixture_passage_column(tab, μ̄, sμ; Δz = 0.04, zmax = 400, ns = 200_000)

The mixture passage law, standardised by its median and IQR/1.349, on a z grid from its left edge (s = 0) to `zmax`
(or the end of the slowest margin's tail), as a one-column `PassageTable` (mean = median, sd = scale, in noise units).
Built on a uniform absolute-time grid of `ns` points; the survival is accumulated from the right (precise far tail).
"""
function mixture_passage_column(tab::PassageTable, μ̄::Real, sμ::Real; Δz::Real = 0.04, zmax::Real = 400.0, ns::Int = 200_000)
    w = margin_weights(tab, μ̄, sμ)
    keep = findall(>(1e-12), w)
    smax = maximum(tab.mean[i] + 30 * tab.sd[i] for i in keep)
    sgrid = range(smax / ns, smax; length = ns)
    ds = step(sgrid)
    g = zeros(ns)
    for i in keep
        m, sd = tab.mean[i], tab.sd[i]
        col = view(tab.logg, :, i)
        wi = w[i] / sd
        @inbounds for (j, s) in enumerate(sgrid)
            g[j] += wi * exp(_interp_z(col, tab.z_grid, (s - m) / sd))
        end
    end
    g ./= sum(g) * ds
    F = cumsum(g) .* ds
    S = reverse(cumsum(reverse(g))) .* ds .- 0.5 .* g .* ds
    q(p) = sgrid[clamp(searchsortedfirst(F, p), 1, ns)]
    med = q(0.5)
    sc = (q(0.75) - q(0.25)) / 1.349
    zlo = max(-8.0, (first(sgrid) - med) / sc)
    zg = range(floor(zlo / Δz) * Δz, min(zmax, (smax - med) / sc); step = Δz)
    function at(v, s)
        s ≤ first(sgrid) && return (v === S ? 1.0 : 0.0)
        x = clamp((s - first(sgrid)) / ds + 1, 1, ns - 1)
        i = floor(Int, x)
        t = x - i
        return (1 - t) * v[i] + t * v[i + 1]
    end
    lg = [max(log(max(at(g, med + sc * z) * sc, 0.0)), LOGFLOOR) for z in zg]
    lF = [max(log(max(at(F, med + sc * z), 0.0)), LOGFLOOR) for z in zg]
    lS = [min(max(log(max(at(S, med + sc * z), 0.0)), LOGFLOOR), 0.0) for z in zg]
    return PassageTable([Float64(μ̄)], zg, reshape(lg, :, 1), reshape(lS, :, 1), reshape(lF, :, 1), [med], [sc], -Inf, Inf)
end

"""
    mixture_sampler(tab, μ̄, sμ)

Sampler of the mixture delay in absolute noise time: rng ↦ s (a margin drawn from the weights, then a passage time at
that margin). Returns (sampler, scale): `scale` = IQR/1.349 of the law, the unit of γ.
"""
function mixture_sampler(tab::PassageTable, μ̄::Real, sμ::Real)
    w = margin_weights(tab, μ̄, sμ)
    cw = cumsum(w)
    cdfs = map(eachindex(tab.μ_grid)) do i
        c = cumsum(exp.(view(tab.logg, :, i)))
        c ./ c[end]
    end
    sc = mixture_passage_column(tab, μ̄, sμ; ns = 50_000).sd[1]
    smp = rng -> begin
        i = clamp(searchsortedfirst(cw, rand(rng)), 1, length(cw))
        z = tab.z_grid[clamp(searchsortedfirst(cdfs[i], rand(rng)), 1, length(tab.z_grid))] - step(tab.z_grid) * rand(rng)
        max(tab.mean[i] + tab.sd[i] * z, 0.0)
    end
    return smp, sc
end

struct MixturePropensityTable
    k::Int
    μ_grid::Vector{Float64} # mean margin μ̄ (uniform)
    s_grid::Vector{Float64} # margin sd s_μ (uniform, from 0)
    logγ_grid::Vector{Float64}
    η::Array{Float64, 4} # (2k + 1) × nμ × ns × nγ
    pd::Array{Float64, 4} # true displacement marginals
    scale::Matrix{Float64} # IQR/1.349 of each mixture law (noise units), nμ × ns
    μc::Float64 # lowest margin kept
    maxerr::Float64
end

"""
    mixture_propensity_table(tab, k; μ_grid, s_grid, logγ_grid, n, J)

η(μ̄, s_μ, log γ) on the grids: mixture law (`mixture_passage_column`), exact bulk displacement marginals with J
neighbours (a delay can overtake many sites), zero-mean tilt, KL projection (`match_propensity`). Threaded over (μ̄, s_μ).
"""
function mixture_propensity_table(tab::PassageTable, k::Int; μ_grid = collect(-0.5:0.25:6.0), s_grid = collect(0.0:0.25:3.0),
        logγ_grid = collect(range(log(0.4), log(8.0); length = 49)), n::Int = 60, J::Int = 60, progress = nothing)
    nμ, ns, nγ = length(μ_grid), length(s_grid), length(logγ_grid)
    η = zeros(2k + 1, nμ, ns, nγ)
    pd = zeros(2k + 1, nμ, ns, nγ)
    scale = zeros(nμ, ns)
    errs = zeros(nμ, ns, nγ)
    done = Threads.Atomic{Int}(0)
    @sync for idx in 1:(nμ * ns)
        Threads.@spawn begin
            iμ, is = Tuple(CartesianIndices((nμ, ns))[idx])
            col = mixture_passage_column(tab, μ_grid[iμ], s_grid[is])
            scale[iμ, is] = col.sd[1]
            ws = PermanentWorkspace(k, n; logspace = false)
            for iγ in 1:nγ
                p = displacement_marginal(col, 1, exp(logγ_grid[iγ]), k; J)
                pd[:, iμ, is, iγ] .= p
                ηm, _, err = match_propensity(zero_mean_tilt(p ./ sum(p), k), k; ws, n)
                η[:, iμ, is, iγ] .= ηm
                errs[iμ, is, iγ] = err
            end
            Threads.atomic_add!(done, 1)
            progress !== nothing && progress(done[], nμ * ns)
        end
    end
    return MixturePropensityTable(k, collect(Float64, μ_grid), collect(Float64, s_grid), collect(Float64, logγ_grid), η, pd, scale,
        first(tab.μ_grid), maximum(errs))
end

"""
    propensity!(η, ∂η, mtab::MixturePropensityTable, μ̄, sμ, logγ)

Tricubic (Keys) interpolation of η(μ̄, s_μ, log γ) and of its Jacobian ∂η[:, 1:3] in (μ̄, s_μ, log γ) (∂η = nothing to
skip). Generic in the number types (dual numbers pass through).
"""
function propensity!(η::AbstractVector, ∂η::Union{AbstractMatrix, Nothing}, mt::MixturePropensityTable, μ̄::Real, sμ::Real, logγ::Real)
    T = mt.η
    nμ, ns, nγ = length(mt.μ_grid), length(mt.s_grid), length(mt.logγ_grid)
    iμ, tμ, hμ = _keys_index(mt.μ_grid, μ̄)
    is, ts, hs = _keys_index(mt.s_grid, sμ)
    iγ, tγ, hγ = _keys_index(mt.logγ_grid, logγ)
    wμ, wsg, wγ = _keys_w(tμ), _keys_w(ts), _keys_w(tγ)
    dμ, dsg, dγ = _keys_dw(tμ), _keys_dw(ts), _keys_dw(tγ)
    fill!(η, 0)
    (∂η !== nothing) && fill!(∂η, 0)
    for a in 1:4, b in 1:4, c in 1:4
        ia, ib, ic = clamp(iμ + a - 1, 1, nμ), clamp(is + b - 1, 1, ns), clamp(iγ + c - 1, 1, nγ)
        ω = wμ[a] * wsg[b] * wγ[c]
        for d in eachindex(η)
            η[d] += ω * T[d, ia, ib, ic]
        end
        if ∂η !== nothing
            ω1, ω2, ω3 = dμ[a] * wsg[b] * wγ[c] / hμ, wμ[a] * dsg[b] * wγ[c] / hs, wμ[a] * wsg[b] * dγ[c] / hγ
            for d in eachindex(η)
                v = T[d, ia, ib, ic]
                ∂η[d, 1] += ω1 * v
                ∂η[d, 2] += ω2 * v
                ∂η[d, 3] += ω3 * v
            end
        end
    end
    return η
end

"""
    reproject(mtab; n = 60)

The same table with η recomputed from its stored exact marginals (`match_propensity`, e.g. after a change of the
projection); returns a new table with its maximum projection error.
"""
function reproject(mt::MixturePropensityTable; n::Int = 60, progress = nothing)
    k = mt.k
    η = similar(mt.η)
    nμ, ns, nγ = length(mt.μ_grid), length(mt.s_grid), length(mt.logγ_grid)
    errs = zeros(nμ, ns, nγ)
    done = Threads.Atomic{Int}(0)
    @sync for idx in 1:(nμ * ns)
        Threads.@spawn begin
            iμ, is = Tuple(CartesianIndices((nμ, ns))[idx])
            ws = PermanentWorkspace(k, n; logspace = false) # scaled linear space (log-space fallback when needed)
            for iγ in 1:nγ
                p = mt.pd[:, iμ, is, iγ]
                ηm, _, err = match_propensity(zero_mean_tilt(p ./ sum(p), k), k; ws, n)
                η[:, iμ, is, iγ] .= ηm
                errs[iμ, is, iγ] = err
            end
            Threads.atomic_add!(done, 1)
            progress !== nothing && progress(done[], nμ * ns)
        end
    end
    return MixturePropensityTable(k, mt.μ_grid, mt.s_grid, mt.logγ_grid, η, mt.pd, mt.scale, mt.μc, maximum(errs))
end

"""
    inband_mass(mtab, μ̄, sμ, logγ)

Probability that a bulk organ is displaced by at most k (tricubic interpolation of the exact marginals). The permanent
model renormalises the rest away: report it at a fit (it should stay close to 1).
"""
function inband_mass(mt::MixturePropensityTable, μ̄::Real, sμ::Real, logγ::Real)
    iμ, tμ, _ = _keys_index(mt.μ_grid, μ̄)
    is, ts, _ = _keys_index(mt.s_grid, sμ)
    iγ, tγ, _ = _keys_index(mt.logγ_grid, logγ)
    wμ, wsg, wγ = _keys_w(tμ), _keys_w(ts), _keys_w(tγ)
    nμ, ns, nγ = length(mt.μ_grid), length(mt.s_grid), length(mt.logγ_grid)
    v = 0.0
    for a in 1:4, b in 1:4, c in 1:4
        v += wμ[a] * wsg[b] * wγ[c] * sum(view(mt.pd, :, clamp(iμ + a - 1, 1, nμ), clamp(is + b - 1, 1, ns), clamp(iγ + c - 1, 1, nγ)))
    end
    return v
end

"""
    save_mixture_tables(path, (; tab, mtab); with_passage = false)

Save as plain arrays. By default without the fine passage table `tab` (large; it is only needed to simulate, and
`passage_table(collect(μc:Δμ:μmax))` rebuilds it, with the values stored in the file).
"""
function save_mixture_tables(path::AbstractString, tabs; with_passage::Bool = false)
    (; tab, mtab) = tabs
    serialize(path, Dict{Symbol, Any}(:tab => (with_passage ? _table_dict(tab) : nothing),
        :passage_grid => (first(tab.μ_grid), tab.μ_grid[2] - tab.μ_grid[1], last(tab.μ_grid)),
        :mtab => Dict{Symbol, Any}(:k => mtab.k, :μ_grid => mtab.μ_grid, :s_grid => mtab.s_grid, :logγ_grid => mtab.logγ_grid,
            :η => mtab.η, :pd => mtab.pd, :scale => mtab.scale, :μc => mtab.μc, :maxerr => mtab.maxerr)))
end
"""
    load_mixture_tables(path; with_passage = false)

(; tab, mtab) from `save_mixture_tables`; `tab` is the stored passage table, or with `with_passage = true` and none
stored, the one rebuilt on the stored margin grid (a few minutes), otherwise `nothing`.
"""
function load_mixture_tables(path::AbstractString; with_passage::Bool = false)
    d = deserialize(path)
    m = d[:mtab]
    mtab = MixturePropensityTable(m[:k], m[:μ_grid], m[:s_grid], m[:logγ_grid], m[:η], m[:pd], m[:scale], m[:μc], m[:maxerr])
    tab = (d[:tab] !== nothing ? _table_from_dict(d[:tab]) :
        with_passage ? (g = d[:passage_grid]; passage_table(collect(g[1]:g[2]:g[3]))) : nothing)
    return (; tab, mtab)
end

"""
    build_mixture_tables(; k = 4, μc = -1.0, μmax = 16.0, Δμ = 0.05, μ_grid, s_grid, logγ_grid, cache, progress)

Fine plateau passage table on [μc, μmax] (step Δμ, the quadrature of the margin integral) and the mixture propensity
table, (; tab, mtab). With `cache`, read it if it matches (the passage table rebuilt if not stored), otherwise compute
and write it (≈ 45 min on 18 threads for the default grids: 35 × 17 laws × 49 γ).
"""
function build_mixture_tables(; k::Int = 4, μc::Real = -1.0, μmax::Real = 16.0, Δμ::Real = 0.05, μ_grid = collect(-0.5:0.25:8.0),
        s_grid = collect(0.0:0.25:4.0), logγ_grid = collect(range(log(0.4), log(8.0); length = 49)), cache::Union{String, Nothing} = nothing,
        progress = nothing)
    if cache !== nothing && isfile(cache)
        t = load_mixture_tables(cache; with_passage = true)
        (t.mtab.k == k && t.mtab.μ_grid == μ_grid && t.mtab.s_grid == s_grid && t.mtab.logγ_grid == logγ_grid && t.mtab.μc == μc) && return t
    end
    tab = passage_table(collect(μc:Δμ:μmax))
    mtab = mixture_propensity_table(tab, k; μ_grid, s_grid, logγ_grid, progress)
    tabs = (; tab, mtab)
    cache !== nothing && save_mixture_tables(cache, tabs)
    return tabs
end

end # module MixturePassage
