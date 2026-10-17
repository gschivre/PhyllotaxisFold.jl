"""
    Level1

Commitment-paced readiness ("Level 1", docs `next_session_level1.md` of the analysis): site n becomes ready only after
the older primordia that frame it on the contact parastichies have committed,

    t_n = R + w τ_{n−p} + (1 − w) τ_{n−q},      τ_n = t_n + ε_n,      ε_n i.i.d. passage delays,

with lags p < q (e.g. the contact pair (3, 5)), weight w and a recovery time R. Unlike Level 0 (t_n = n T_P, a clock),
a late commitment delays the organs it frames: errors propagate, the absolute phase diffuses (unit root of the
recursion) and nearby commitment times are correlated. Times are in units of the passage sd. The mean plastochron is
T̄ = (R + E[ε]) / (w p + (1 − w) q); γ = T̄ / sd(ε) fixes R given the shape (R ≥ 0 requires γ ≥ E[ε]/sd(ε) / (wp + (1 − w)q)).

This module is stage 0 of the Level-1 plan: a simulator with the Level-0 limit for comparison, and the statistics that
separate the two architectures (no likelihood yet):

- `simulate_paced`: organ orders and commitment times of either architecture
- `timeline_structure`: Var(τ_{(p+m)} − τ_{(p)} − m T̄) against m (bounded at Level 0, growing at Level 1)
- `interval_autocorrelation`: autocorrelation of the sorted commitment intervals (the internodes, up to growth)
- `cascade_lengths`: lengths of the runs of consecutive displaced positions in orders
"""
module Level1

using Random, Statistics
using ..PassageTime: PassageTable, passage_sampler

export simulate_paced, level1_recovery, timeline_structure, interval_autocorrelation, cascade_lengths

"""
    level1_recovery(tab, μ, γ; p = 3, q = 5, w = 0.5)

The recovery time R (passage sd units) that gives the mean plastochron γ (sd units) under Level 1 at shape μ:
R = γ (w p + (1 − w) q) − E[ε]/sd(ε). Negative when γ is too small for the shape (readiness would precede the
commitment of the framing organs).
"""
level1_recovery(tab::PassageTable, μ::Real, γ::Real; p::Int = 3, q::Int = 5, w::Real = 0.5) =
    γ * (w * p + (1 - w) * q) - _cv_inv(tab, μ)
_cv_inv(tab, μ) = (i = argmin(abs.(tab.μ_grid .- μ)); tab.mean[i] / tab.sd[i]) # E[ε] / sd(ε) at the nearest grid margin

"""
    simulate_paced(tab, μ, γ; architecture = :level1, p = 3, q = 5, w = 0.5, plants = 100, organs = 50, burnin = 200, rng)

Simulate commitment times and organ orders. `architecture = :level0`: readiness on the lattice n γ (model B);
`:level1`: the recursion above, started on the lattice and run for `burnin` organs before the `organs` kept (the stem
is the kept sites: σ = sortperm of their commitment times). Delays ε from the plateau law at μ (`passage_sampler`, sd
units, mean E[ε]/sd(ε)). Returns (; orders, times) with times[s] the sorted commitment times of stem s (sd units).
"""
function simulate_paced(tab::PassageTable, μ::Real, γ::Real; architecture::Symbol = :level1, p::Int = 3, q::Int = 5, w::Real = 0.5,
        plants::Int = 100, organs::Int = 50, burnin::Int = 200, rng::AbstractRNG = Random.default_rng())
    @assert architecture in (:level0, :level1)
    @assert 1 ≤ p < q
    smp = passage_sampler(tab, μ)
    c = _cv_inv(tab, μ)
    ε() = c + smp(rng) # delay in sd units, mean E[ε]/sd
    R = level1_recovery(tab, μ, γ; p, q, w)
    architecture === :level1 && R < 0 && @warn "R = $(round(R; digits = 2)) < 0: γ too small for Level 1 at this shape"
    orders = Vector{Vector{Int}}(undef, plants)
    times = Vector{Vector{Float64}}(undef, plants)
    N = burnin + organs
    τ = zeros(N)
    for s in 1:plants
        for n in 1:N
            t = (architecture === :level0 || n ≤ q ? n * γ : R + w * τ[n - p] + (1 - w) * τ[n - q])
            τ[n] = t + ε()
        end
        kept = τ[(burnin + 1):N]
        orders[s] = sortperm(kept)
        times[s] = sort(kept)
    end
    return (; orders, times, R)
end

"""
    timeline_structure(times, T̄; mmax = 20)

Structure function of the commitment timeline: for m = 1..mmax, the variance over stems and positions of
τ_(p+m) − τ_(p) − m T̄ (sorted times). Level 0 levels off (≈ twice the variance of a delay, in T̄² units 2/γ²); Level 1
grows with m (phase diffusion).
"""
function timeline_structure(times::AbstractVector, T̄::Real; mmax::Int = 20)
    return [var(reduce(vcat, [t[(m + 1):end] .- t[1:(end - m)] .- m * T̄ for t in times])) for m in 1:mmax]
end

"""
    interval_autocorrelation(times; maxlag = 8)

Autocorrelation of the sorted commitment intervals at lags 1..maxlag, pooled over stems (each stem's intervals centred
on its own mean). Level 0: ≈ −0.5 at lag 1 and ≈ 0 beyond for precise clocks; Level 1: structure at the lags p and q.
"""
function interval_autocorrelation(times::AbstractVector; maxlag::Int = 8)
    I = [diff(t) .- mean(diff(t)) for t in times]
    v = mean(reduce(vcat, [x .^ 2 for x in I]))
    return [mean(reduce(vcat, [x[(l + 1):end] .* x[1:(end - l)] for x in I])) / v for l in 1:maxlag]
end

"""
    cascade_lengths(orders)

Lengths of the runs of consecutive displaced positions (σ[p] ≠ p) in each order, pooled: a single swap is a run of 2,
a 231 pattern a run of 3, and longer runs are cascades of permutations.
"""
function cascade_lengths(orders::AbstractVector)
    out = Int[]
    for σ in orders
        run = 0
        for p in eachindex(σ)
            if σ[p] != p
                run += 1
            elseif run > 0
                push!(out, run); run = 0
            end
        end
        run > 0 && push!(out, run)
    end
    return out
end

end # module Level1
