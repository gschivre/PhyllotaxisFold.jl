"""
    CommitmentTiming

The fold-commit (Level-0) timing law of organ order, scored through the posterior orders of the propensity model.

Site i becomes ready at time i (plastochron units) and commits after an i.i.d. delay eᵢ ~ f; the organ order is
σ = sortperm(i + eᵢ). The propensity model fitted to the angles gives, per stem, exact posterior samples of σ
(`order_posterior`). Both models share the angle likelihood p(y | σ), so for any delay law f

    p(y | f) / p(y | A) = E_{σ ∼ p(σ | y, A)} [ P_f(σ) / P_A(σ) ]

(P_A: the banded-permanent law with the fitted costs η; P_f: the exact Level-0 order law). Any delay law is thereby
scored against the propensity model, and against any other law, without refitting the angle model.

- `DelayLaw`: f = (1 − π_L − π_E) N(0, s²) + π_L (N(0, s²) + Exp(m_L)) + π_E (N(0, s²) − Exp(m_E))
- `loglik_order_exact`: the exact Level-0 log P(σ) (forward recursion on a delay grid, midpoint rule for ties)
- `order_samples`, `stem_terms`, `relative_loglik`: the re-weighting estimate, per stem and summed
- `fit_delay_law`: maximum of the summed terms for the :core, :late or :both law
- `delay_law_inference`: robust (sandwich, clustered by stem) standard errors of s and of the tail masses
- `simulate_stems(f::DelayLaw; ...)`: stems whose order comes from Level 0 with delay law f (for validation)
"""
module CommitmentTiming

using Random, Statistics, LinearAlgebra
using Distributions, LogExpFunctions, Optim
using ..Common: Golden_angle, Lucas_angle
using ..Permanents: PermanentWorkspace, logprob_permutation
import ..AngleLikelihood: simulate_stems

export DelayLaw, DELAY_GRID, delay_logdensity, tail_masses, loglik_order_exact, order_diagnostics, order_samples,
       stem_terms, relative_loglik, fit_delay_law, delay_law_inference, simulate_stems

"default delay grid (plastochrons): wide enough for tails of a few plastochrons, fine enough for a core sd of 0.3"
const DELAY_GRID = range(-12.0, 30.0; step = 0.05)

"""
    DelayLaw(s; πL = 0, mL = 1, πE = 0, mE = 1)

Commitment delay in plastochrons: a Gaussian core of sd `s`; a fraction `πL` of the sites is late by an extra
exponential delay of mean `mL`, a fraction `πE` early by an exponential advance of mean `mE`.
"""
struct DelayLaw
    s::Float64
    πL::Float64
    mL::Float64
    πE::Float64
    mE::Float64
end
DelayLaw(s::Real; πL::Real = 0.0, mL::Real = 1.0, πE::Real = 0.0, mE::Real = 1.0) = DelayLaw(s, πL, mL, πE, mE)

function Base.rand(rng::AbstractRNG, f::DelayLaw)
    u, e = rand(rng), f.s * randn(rng)
    u < f.πL && return e + f.mL * randexp(rng)
    u < f.πL + f.πE && return e - f.mE * randexp(rng)
    return e
end

_normlogpdf(x) = -x^2 / 2 - log(2π) / 2
# log density of N(0, s²) convolved with ±Exp(mean m) at z (exponentially modified Gaussian)
function _logemg(z, s, m, sign)
    λ = 1 / m
    x = sign * z
    return log(λ) + λ * (λ * s^2 / 2 - x) + logcdf(Normal(), (x - λ * s^2) / s)
end

"log density of f on `grid`, normalised numerically there"
function delay_logdensity(f::DelayLaw; grid = DELAY_GRID)
    lg = map(grid) do z
        terms = [log1p(-f.πL - f.πE) + _normlogpdf(z / f.s) - log(f.s)]
        f.πL > 0 && push!(terms, log(f.πL) + _logemg(z, f.s, f.mL, 1))
        f.πE > 0 && push!(terms, log(f.πE) + _logemg(z, f.s, f.mE, -1))
        logsumexp(terms)
    end
    lg .-= log(sum(exp.(lg)) * step(grid))
    return max.(lg, -700.0)
end

"""
    tail_masses(f; a = 1, grid)

P(e > a), P(e < −a) and their sum: the fractions of sites committed more than `a` plastochrons late or early. Unlike
π and m one by one, these are well identified from organ orders (a frequent short tail and a rarer long one give nearly
the same orders).
"""
function tail_masses(f::DelayLaw; a::Real = 1.0, grid = DELAY_GRID)
    g = exp.(delay_logdensity(f; grid)) .* step(grid)
    # each grid value stands for a cell of width Δz centred on it: the cell that contains ±a counts for its part beyond
    cdf_at(x) = (i = clamp(floor(Int, (x - first(grid)) / step(grid) + 0.5) + 1, 1, length(grid));
        sum(g[1:(i - 1)]) + g[i] * ((x - first(grid)) / step(grid) + 1.5 - i))
    late, early = 1 - cdf_at(a), cdf_at(-a)
    return (; late, early, off = late + early)
end

"""
    loglik_order_exact(σ, logg, z_grid, γ)

Exact Level-0 log P(σ) for sites 1..n with readiness iγ and i.i.d. delay density g:
P(σ) = ∫_{τ_1 < … < τ_n} ∏_p g(τ_p − σ[p] γ) dτ, by a forward recursion on a time grid with step Δz. Two times in the
same grid cell are ordered with probability ½ (midpoint rule): counting such ties as ordered biased log P(σ) upwards at
first order in Δz (Gaussian delays of sd 0.43 on a 0.05 grid: +0.28 for an order with three displacements); the error
is second order (≤ 0.01 there).
"""
function loglik_order_exact(σ::AbstractVector{Int}, logg::AbstractVector, z_grid, γ::Real)
    n = length(σ)
    Δz = step(z_grid)
    zmin, zmax = first(z_grid), last(z_grid)
    tmin = γ * minimum(σ) + zmin
    L = round(Int, (γ * (maximum(σ) - minimum(σ)) + (zmax - zmin)) / Δz) + 2
    A = ones(L) # P(previous time < a time in each cell)
    α = zeros(L)
    ll = 0.0
    for p in 1:n
        tp = γ * σ[p]
        fill!(α, 0.0)
        ka = max(1, ceil(Int, (tp + zmin - tmin) / Δz) + 1)
        kb = min(L, floor(Int, (tp + zmax - tmin) / Δz) + 1)
        @inbounds for kk in ka:kb
            α[kk] = exp(_interp(logg, z_grid, tmin + (kk - 1) * Δz - tp)) * A[kk]
        end
        c = sum(α) * Δz
        (c > 0) || return -Inf
        ll += log(c)
        acc = 0.0
        @inbounds for kk in 1:L
            a = α[kk] * Δz / c
            A[kk] = acc + a / 2 # the cells before, half of cell kk
            acc += a
        end
    end
    return ll
end
loglik_order_exact(σ::AbstractVector{Int}, f::DelayLaw; grid = DELAY_GRID) = loglik_order_exact(σ, delay_logdensity(f; grid), grid, 1.0)

@inline function _interp(col::AbstractVector, z_grid, z)
    x = (z - first(z_grid)) / step(z_grid) + 1
    nz = length(z_grid)
    x ≤ 1 && return col[1]
    x ≥ nz && return col[nz]
    i = floor(Int, x)
    t = x - i
    return (1 - t) * col[i] + t * col[i + 1]
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
# re-weighting of the propensity model's posterior orders
#################################################################################################################

"""
    order_samples(post, η)

Per stem, from `order_posterior(m, fit; nsamples)`: the distinct sampled orders, their sample frequencies `w` and
log P_A(σ) under the propensity model's full cost vector η (length 2k + 1).
"""
function order_samples(post, η::AbstractVector)
    k = (length(η) - 1) ÷ 2
    map(post) do p
        u = unique(p.samples)
        cnt = [count(==(σ), p.samples) for σ in u]
        ws = PermanentWorkspace(k, length(first(u)))
        logA = [logprob_permutation(ws, η, σ) for σ in u]
        all(isfinite, logA) || throw(ArgumentError("an order has a displacement beyond the band k = $(k) of η"))
        (; orders = u, w = cnt ./ sum(cnt), logA)
    end
end

"""
    stem_terms(samples, f; grid)
    stem_terms(samples, logP)

Per stem, log p(y | f) − log p(y | A) = log Σ_j w_j P_f(σ_j) / P_A(σ_j) over its sampled orders, for a delay law f
(exact Level-0 order law) or any order law given as a function σ ↦ log P(σ). Threaded over stems.
"""
function stem_terms(samples, logP::Function)
    out = zeros(length(samples))
    Threads.@threads for i in eachindex(samples)
        s = samples[i]
        out[i] = logsumexp([log(s.w[j]) + logP(s.orders[j]) - s.logA[j] for j in eachindex(s.orders)])
    end
    return out
end
function stem_terms(samples, f::DelayLaw; grid = DELAY_GRID)
    lg = delay_logdensity(f; grid)
    return stem_terms(samples, σ -> loglik_order_exact(σ, lg, grid, 1.0))
end

"log p(y | f) − log p(y | A) summed over stems (positive: f explains the orders better than the propensity model)"
relative_loglik(samples, f; kwargs...) = sum(stem_terms(samples, f; kwargs...))

# unconstrained parameters of the variants. :both in contrast form: log s, logit(π_L + π_E), logit(π_L / (π_L + π_E)),
# log m_L, log m_E
function _law(v, variant::Symbol)
    variant === :core && return DelayLaw(exp(v[1]))
    variant === :late && return DelayLaw(exp(v[1]); πL = logistic(v[2]), mL = exp(v[3]))
    πt, fL = logistic(v[2]), logistic(v[3])
    return DelayLaw(exp(v[1]); πL = πt * fL, mL = exp(v[4]), πE = πt * (1 - fL), mE = exp(v[5]))
end
function _v(f::DelayLaw, variant::Symbol)
    variant === :core && return [log(f.s)]
    variant === :late && return [log(f.s), logit(clamp(f.πL, 1e-6, 1 - 1e-6)), log(f.mL)]
    πt = f.πL + f.πE
    return [log(f.s), logit(clamp(πt, 1e-6, 1 - 1e-6)), logit(clamp(f.πL / max(πt, 1e-12), 1e-6, 1 - 1e-6)), log(f.mL), log(f.mE)]
end

"""
    fit_delay_law(samples, variant = :both; init, grid, iterations = 3000)

Delay law maximising `relative_loglik` (Nelder–Mead on unconstrained parameters). `variant`: :core (s), :late
(s, π_L, m_L) or :both (all five). `init`: a starting `DelayLaw`. Returns (; law, v, variant, rel_ll, converged).
"""
function fit_delay_law(samples, variant::Symbol = :both; init::DelayLaw = DelayLaw(0.5; πL = 0.05, mL = 1.0, πE = 0.05, mE = 1.0),
        grid = DELAY_GRID, iterations::Int = 3000)
    obj(v) = -relative_loglik(samples, _law(v, variant); grid)
    r = optimize(obj, _v(init, variant), NelderMead(), Optim.Options(; iterations, f_reltol = 1e-9))
    v = Optim.minimizer(r)
    return (; law = _law(v, variant), v, variant, rel_ll = -Optim.minimum(r), converged = Optim.converged(r))
end

_derived(v, variant, grid) = (f = _law(v, variant); t = tail_masses(f; grid); [f.s, t.late, t.early, t.off])

"""
    delay_law_inference(samples, fit; h = 1e-3, grid)

Robust standard errors of the core sd s and of the tail masses P(e > 1), P(e < −1), P(|e| > 1) at a `fit_delay_law`
result: sandwich covariance H⁻¹ (Σ_s g_s g_sᵀ) H⁻¹ of the unconstrained parameters (per-stem scores g_s, Hessian H of
the summed terms, by central differences; clustered by stem, so heterogeneity between stems is allowed), then the delta
method. `se_model` is the model-based (−H⁻¹) version. pinv: a tail with π ≈ 0 leaves its distance unidentified.
"""
function delay_law_inference(samples, fit; h::Real = 1e-3, grid = DELAY_GRID)
    v, variant = fit.v, fit.variant
    P = length(v)
    scores(w) = reduce(hcat, [(e = zeros(P); e[i] = h;
        (stem_terms(samples, _law(w .+ e, variant); grid) .- stem_terms(samples, _law(w .- e, variant); grid)) ./ (2h)) for i in 1:P])
    G = scores(v)
    H = reduce(hcat, [(e = zeros(P); e[i] = h; (vec(sum(scores(v .+ e); dims = 1)) .- vec(sum(scores(v .- e); dims = 1))) ./ (2h)) for i in 1:P])
    Hi = pinv(-(H + H') / 2; rtol = 1e-8)
    V = Hi * (G' * G) * Hi
    J = reduce(hcat, [(e = zeros(P); e[i] = 1e-4; (_derived(v .+ e, variant, grid) .- _derived(v .- e, variant, grid)) ./ 2e-4) for i in 1:P])
    est = _derived(v, variant, grid)
    se(C) = sqrt.(max.(diag(J * C * J'), 0.0))
    names = (:s, :late, :early, :off)
    return (; names, estimate = NamedTuple{names}(Tuple(est)), se = NamedTuple{names}(Tuple(se(V))),
        se_model = NamedTuple{names}(Tuple(se(Hi))), V)
end

#################################################################################################################
# simulation
#################################################################################################################

"""
    simulate_stems(f::DelayLaw; plants = 100, organs = 50, x = nothing, logμκ = 1.5, σκ = 0, α = 0.95, outlier = 0,
        torsion_mean = 0, torsion_sd = 0, rng)

Stems under Level-0 fold commitment: the order of each stem is σ = sortperm(i + eᵢ) with eᵢ ~ f, and the angles
y_p ~ vM(ϕ + o (σ[p] − 1) ψ + τ_s x_p, κ_s) as in model A's simulator (κ_s ~ LogNormal(logμκ, σκ), golden with
probability α, uniform ϕ and orientation, uniform outlier angles with probability `outlier`). `organs` is one count or
one per stem; `x` the twist coordinate of each stem (e.g. mm of stem below each organ; default: the positions
0, 1, …), and the twist τ_s ~ N(torsion_mean, torsion_sd²) is in degrees per unit of x. Returns (; angles, orders,
delays, torsion, κ, outliers, ψ, o).
"""
function simulate_stems(f::DelayLaw; plants::Int = 100, organs = 50, x = nothing, logμκ::Real = 1.5, σκ::Real = 0.0, α::Real = 0.95,
        outlier::Real = 0.0, torsion_mean::Real = 0.0, torsion_sd::Real = 0.0, rng::AbstractRNG = Random.default_rng())
    ns = organs isa Integer ? fill(organs, plants) : collect(organs)
    xs = x === nothing ? [collect(0.0:(n - 1)) for n in ns] : x
    @assert length(ns) == plants && length(xs) == plants && all(length.(xs) .== ns)
    out = map(1:plants) do s
        n = ns[s]
        e = [rand(rng, f) for _ in 1:n]
        σ = sortperm((1:n) .+ e)
        κ = exp(logμκ + σκ * randn(rng))
        ψ = rand(rng) < α ? Golden_angle : Lucas_angle
        ϕ, o, τ = 2π * rand(rng), (rand(rng) < 0.5 ? 1 : -1), torsion_mean + torsion_sd * randn(rng)
        bad = [rand(rng) < outlier for _ in 1:n]
        y = [bad[p] ? 2π * rand(rng) : rand(rng, VonMises(mod2pi(ϕ + o * (σ[p] - 1) * ψ + deg2rad(τ) * xs[s][p]), κ)) for p in 1:n]
        (; y, σ, e, τ, κ, bad = findall(bad), ψ, o)
    end
    return (; angles = [z.y for z in out], orders = [z.σ for z in out], delays = [z.e for z in out], torsion = [z.τ for z in out],
        κ = [z.κ for z in out], outliers = [z.bad for z in out], ψ = [z.ψ for z in out], o = [z.o for z in out])
end

end # module CommitmentTiming
