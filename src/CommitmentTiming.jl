# CommitmentTiming
#
# Laws of commitment delays (independent delays, one law for every site) as laws of organ order, scored through the
# posterior orders of the propensity model.
#
# Site i becomes ready at time i (plastochron units) and commits after an i.i.d. delay eᵢ ~ f; the organ order is
# σ = sortperm(i + eᵢ). The propensity model fitted to the angles gives, per stem, exact posterior samples of σ
# (`order_posterior`). Both models share the angle likelihood p(y | σ), so for any delay law f
#
#     p(y | f) / p(y | A) = E_{σ ∼ p(σ | y, A)} [ P_f(σ) / P_A(σ) ]
#
# (A: the propensity model; `P_A`: its banded-permanent law with the fitted costs η; `P_f`: the exact order law of the
# delays f). Any delay law is thereby scored against the propensity model, and against any other law, without refitting
# the angle model.
#
# - `DelayLaw`: the abstract type of a delay law. A law is defined by its log density (`delay_logpdf`) and, to be
#   fitted, its free parameters (`free_parameters`, `with_free_parameters`); everything below works for any law
# - `GaussianExpTails`, the default law: `f = (1 − π_L − π_E) N(0, σc²) + π_L (N(0, σc²) + Exp(m_L)) + π_E (N(0, σc²) − Exp(m_E))`
# - `loglik_order_exact`: the exact `log P_f(σ)` (forward recursion on a delay grid, midpoint rule for ties)
# - `order_samples`, `stem_terms`, `relative_loglik`: the re-weighting estimate, per stem and summed
# - `fit_delay_law`: maximum of the summed terms over the free parameters of a law (Newton trust region, ForwardDiff),
#   started from a law of the family (for the default law, a named variant: `GaussianExpTails(:core)`, `:late`,
#   `:early` or `:both`); given several groups of stems, one law per group with shared leading parameters (the core sd
#   of the default law)
# - `overtaking`: the overtaking profile `q_d = P(site i + d commits before site i)` of a law, the quantity the orders
#   identify; `overtaking_profile`: the same read directly from the posterior orders, with no delay law;
#   `compare_profiles`: Wald test of two groups' profiles
# - `delay_law_inference`: robust (sandwich, clustered by stem) standard errors of the reported parameters (σc for the
#   default law), the tail masses and the profile
# - `shared_core_test`: likelihood ratio of shared parameters (one core sd) for all groups against one set per group
# - `simulate_stems(f::DelayLaw; ...)`: stems whose order comes from independent delays f (for validation)
#
# Everything between the parameters and the summed terms is generic in the number type, so the gradient and the Hessian
# are exact (ForwardDiff); the methods of a new law must be generic too.

"""
default delay grid (plastochrons), the default of `delay_grid`. The density is renormalised on the grid, so the mass
it cuts off must be negligible (π e^{−30/m} for a tail of mass π and mean m). The grid is symmetric because late and
early tails are treated alike.
"""
const DELAY_GRID = range(-30.0, 30.0; step = 0.05)

#################################################################################################################
# delay laws: the interface
#################################################################################################################

"""
    DelayLaw

A law f of commitment delays, in plastochrons. Any subtype can be scored on the orders, fitted, compared and simulated
by the functions of this module, through these methods:

| method                         |                         | used by                                             |
|--------------------------------|-------------------------|-----------------------------------------------------|
| `delay_logpdf(f, z)`           | required                | everything (the law on its grid)                    |
| `free_parameters(f)`           | required to fit         | `fit_delay_law`, `delay_law_inference`              |
| `with_free_parameters(f, v)`   | required to fit         | `fit_delay_law`, `delay_law_inference`              |
| `reported_parameters(f)`       | optional (default none) | `delay_law_inference`                               |
| `nshared_parameters(f)`        | optional (default 0)    | joint fits of groups, `shared_core_test`            |
| `delay_grid(f)`                | optional (default `DELAY_GRID`) | the default `grid` of every function below  |
| `rand(rng, f)`                 | optional (default: from the density on the grid) | `simulate_stems` |

The fits pass ForwardDiff duals through the parameters, so the methods must be generic in their number type. The
tail masses (`tail_masses`) and the overtaking profile (`overtaking`) follow from the density, with no extra method.
`GaussianExpTails` is the default law.
"""
abstract type DelayLaw end

"""
    delay_logpdf(f::DelayLaw, z)

Log density of the delay law `f` at the delay `z` (plastochrons), up to an additive constant. The constant may depend
on the parameters of f: the density is renormalised on the delay grid (`delay_logdensity`). `-Inf` outside the support
is allowed. The one method a `DelayLaw` must define; generic in the number type of f's parameters.
"""
function delay_logpdf end

"""
    free_parameters(f::DelayLaw) -> Vector

The parameters of `f` that `fit_delay_law` adjusts, on an unconstrained scale (the log of a scale, the logit of a
fraction, ...), so that every real vector is a valid law. The starting point of a fit started at f. Must agree with
`with_free_parameters`: `with_free_parameters(f, free_parameters(f))` is f. Required to fit a law.
"""
function free_parameters end

"""
    with_free_parameters(f::DelayLaw, v)

The law of the same family as `f` at the free parameters `v` (`free_parameters`), in the number type of v (Float64,
or ForwardDiff duals inside the fits). f serves as a template: it sets the family and the parameters it leaves fixed.
Required to fit a law.
"""
function with_free_parameters end

"""
    reported_parameters(f::DelayLaw) -> NamedTuple

Parameters of `f` that `delay_law_inference` reports, with their robust standard errors, before the tail masses and
the overtaking profile: a NamedTuple of numbers computed from f (generic in their type; e.g. `(; σc = f.σc)` for
`GaussianExpTails`). Optional: none by default.
"""
reported_parameters(::DelayLaw) = (;)

"""
    nshared_parameters(f::DelayLaw) -> Int

Number of leading free parameters (`free_parameters`) that the groups of a joint fit share
(`fit_delay_law(groups, f)`, `shared_core_test`); the others are fitted per group. For `GaussianExpTails`, 1: the
core sd. Optional: 0 by default, which rules out joint fits.
"""
nshared_parameters(::DelayLaw) = 0

"""
    delay_grid(f::DelayLaw) -> AbstractRange

The grid of delays (plastochrons) on which `f` is evaluated and renormalised: the default `grid` of every function
that takes one (fits, scores, profile, tail masses) and the grid of the fallback sampler. Optional: `DELAY_GRID`
(−30 to 30, step 0.05) by default. A law with longer tails or a narrower core defines its own, e.g.
`delay_grid(::MyLaw) = range(-60.0, 60.0; step = 0.05)`. The density must have decayed at both ends, and the step
must fit a whole number of times in one plastochron (0.05, 0.1, 0.25, …).
"""
delay_grid(::DelayLaw) = DELAY_GRID

# fallback sampler: a grid cell drawn from the density on delay_grid(f), then a uniform point in it. It evaluates the
# density on the whole grid for each draw: a law simulated often should define its own rand
function Base.rand(rng::AbstractRNG, f::DelayLaw)
    grid = delay_grid(f)
    c = cumsum(exp.(_logdensity(f, grid)))
    i = min(searchsortedfirst(c, rand(rng) * last(c)), length(c))
    return grid[i] + (rand(rng) - 0.5) * step(grid)
end
Base.rand(f::DelayLaw) = rand(Random.default_rng(), f)

# log density on the grid, normalised there (generic in the number type of the law: Float64, or duals inside the fits)
function _logdensity(f, grid)
    lg = [delay_logpdf(f, z) for z in grid]
    isconcretetype(eltype(lg)) || (lg = convert(Vector{mapreduce(typeof, promote_type, lg)}, lg)) # e.g. -Inf among duals
    lg = lg .- (logsumexp(lg) + log(step(grid)))
    return max.(lg, -700.0)
end

#################################################################################################################
# the default law: a Gaussian core with exponential tails
#################################################################################################################

"""
    GaussianExpTails(σc; πL = 0, mL = 1, πE = 0, mE = 1)
    GaussianExpTails(σc, πL, mL, πE, mE)
    GaussianExpTails(variant::Symbol; σc = 0.5, πL = 0.05, mL = 1, πE = 0.05, mE = 1)
    GaussianExpTails(variant::Symbol, f::GaussianExpTails)

The default delay law, in plastochrons: a Gaussian core of sd `σc`; a fraction `πL` of the sites is late by an extra
exponential delay of mean `mL`, a fraction `πE` early by an exponential advance of mean `mE`:

    f = (1 − π_L − π_E) N(0, σc²) + π_L (N(0, σc²) ∗ Exp(m_L)) + π_E (N(0, σc²) ∗ (−Exp(m_E)))

The parameters share one floating-point type (promoted; ForwardDiff duals inside the fits).

A fit started at this law frees the tails it has, and keeps a tail of mass 0 absent. Free parameters
(`free_parameters`): with both tails, `log σc`, `logit(π_L + π_E)`, `logit(π_L / (π_L + π_E))`, `log m_L`, `log m_E` (the tail
mass and the late share are separate coordinates because the first is better determined than the second); with one
tail, log σc, logit π, log m; with none, log σc. The core sd is the parameter shared by the groups of a joint fit, and
the one reported by `delay_law_inference`.

The named variants are starting laws for `fit_delay_law`, with the tails they fit and nothing else:

| variant  | tails | free parameters        |
|----------|-------|------------------------|
| `:core`  | none  | σc                     |
| `:late`  | late  | σc, `π_L`, `m_L`       |
| `:early` | early | σc, `π_E`, `m_E`       |
| `:both`  | both  | σc, `π_L`, `m_L`, `π_E`, `m_E` |

`GaussianExpTails(:late)` starts from the keyword values; `GaussianExpTails(:both, f)` starts from the law `f` (e.g.
a fitted one), with the tails of the variant switched on (a tail of mass 0 in f starts at 10⁻⁶) and the others off.
`:core`, `:late` and `:both` are nested, so likelihood ratios between their fits test for tails, and on which side.
"""
struct GaussianExpTails{T <: Real} <: DelayLaw
    σc::T
    πL::T
    mL::T
    πE::T
    mE::T
    GaussianExpTails{T}(σc, πL, mL, πE, mE) where {T <: Real} = new{T}(σc, πL, mL, πE, mE)
end
function GaussianExpTails(σc::Real, πL::Real, mL::Real, πE::Real, mE::Real)
    p = promote(float(σc), πL, mL, πE, mE)
    return GaussianExpTails{typeof(first(p))}(p...)
end
GaussianExpTails(σc::Real; πL::Real = 0.0, mL::Real = 1.0, πE::Real = 0.0, mE::Real = 1.0) = GaussianExpTails(σc, πL, mL, πE, mE)

function Base.rand(rng::AbstractRNG, f::GaussianExpTails)
    u, e = rand(rng), f.σc * randn(rng)
    u < f.πL && return e + f.mL * randexp(rng)
    u < f.πL + f.πE && return e - f.mE * randexp(rng)
    return e
end

_normlogpdf(x) = -x^2 / 2 - log(2π) / 2
_normlogcdf(x) = x < -1 ? log(erfcx(-x / sqrt(2)) / 2) - x^2 / 2 : log1p(-erfc(x / sqrt(2)) / 2)
# log density of N(0, σ²) convolved with ±Exp(mean m) at z (exponentially modified Gaussian)
function _logemg(z, σ, m, sign)
    λ = 1 / m
    x = sign * z
    return log(λ) + λ * (λ * σ^2 / 2 - x) + _normlogcdf((x - λ * σ^2) / σ)
end

function delay_logpdf(f::GaussianExpTails, z)
    l = log1p(-f.πL - f.πE) - log(f.σc) + _normlogpdf(z / f.σc)
    f.πL > 0 && (l = logaddexp(l, log(f.πL) + _logemg(z, f.σc, f.mL, 1)))
    f.πE > 0 && (l = logaddexp(l, log(f.πE) + _logemg(z, f.σc, f.mE, -1)))
    return l
end

function free_parameters(f::GaussianExpTails)
    L, E = f.πL > 0, f.πE > 0
    L && E && return [log(f.σc), logit(f.πL + f.πE), logit(f.πL / (f.πL + f.πE)), log(f.mL), log(f.mE)]
    L && return [log(f.σc), logit(f.πL), log(f.mL)]
    E && return [log(f.σc), logit(f.πE), log(f.mE)]
    return [log(f.σc)]
end
function with_free_parameters(f::GaussianExpTails, v)
    σc, L, E = exp(v[1]), f.πL > 0, f.πE > 0
    if L && E
        πt, fL = logistic(v[2]), logistic(v[3])
        return GaussianExpTails(σc, πt * fL, exp(v[4]), πt * (1 - fL), exp(v[5]))
    end
    L && return GaussianExpTails(σc; πL = logistic(v[2]), mL = exp(v[3]))
    E && return GaussianExpTails(σc; πE = logistic(v[2]), mE = exp(v[3]))
    return GaussianExpTails(σc)
end
reported_parameters(f::GaussianExpTails) = (; σc = f.σc)
nshared_parameters(::GaussianExpTails) = 1

# a named variant: the tails of f switched on (mass at least 1e-6) or off
function GaussianExpTails(variant::Symbol, f::GaussianExpTails)
    c(p) = clamp(p, 1.0e-6, 1 - 1.0e-6)
    variant === :core && return GaussianExpTails(f.σc)
    variant === :late && return GaussianExpTails(f.σc; πL = c(f.πL), mL = f.mL)
    variant === :early && return GaussianExpTails(f.σc; πE = c(f.πE), mE = f.mE)
    variant === :both || throw(ArgumentError("variant must be :core, :late, :early or :both"))
    πt, fL = c(f.πL + f.πE), c(f.πL / max(f.πL + f.πE, 1.0e-12))
    return GaussianExpTails(f.σc, πt * fL, f.mL, πt * (1 - fL), f.mE)
end
function GaussianExpTails(
        variant::Symbol; σc::Real = 0.5, πL::Real = 0.05, mL::Real = 1.0, πE::Real = 0.05, mE::Real = 1.0
    )
    return GaussianExpTails(variant, GaussianExpTails(σc, πL, mL, πE, mE))
end

#################################################################################################################
# properties of a law and the exact order likelihood
#################################################################################################################

"log density of f on `grid`, normalised numerically there"
delay_logdensity(f::DelayLaw; grid = delay_grid(f)) = _logdensity(f, grid)

function _tail_masses(f, a, grid)
    g = exp.(_logdensity(f, grid)) .* step(grid)
    # each grid value stands for a cell of width Δz centred on it: the cell that contains ±a counts for its part beyond
    cdf_at(x) = (
        i = clamp(floor(Int, (x - first(grid)) / step(grid) + 0.5) + 1, 1, length(grid));
        sum(g[1:(i - 1)]) + g[i] * ((x - first(grid)) / step(grid) + 1.5 - i)
    )
    late, early = 1 - cdf_at(a), cdf_at(-a)
    return (; late, early, off = late + early)
end

"""
    tail_masses(f; a = 1, grid)

P(e > a), P(e < −a) and their sum: the fractions of sites committed more than `a` plastochrons late or early. They
are properties of f itself, read from the orders only through the shape of the law: better determined than π and m
one by one, but not identified like the overtaking profile (`overtaking`).
"""
tail_masses(f::DelayLaw; a::Real = 1.0, grid = delay_grid(f)) = _tail_masses(f, a, grid)

# q_d = P(e_i − e_{i+d} > d) = Σ_y g(y) P(e > y + d): each grid value stands for a cell, and the cell of y + d counts
# for one half (ties). d must be a multiple of the grid step (an integer d with a step that divides 1).
function _overtaking(f, ds, grid)
    g = exp.(_logdensity(f, grid)) .* step(grid)
    sur = reverse(cumsum(reverse(g))) .- g ./ 2 # P(e > z) at each grid value, half of its own cell
    return map(ds) do d
        sh = round(Int, d / step(grid))
        isapprox(sh * step(grid), d; atol = 1.0e-9) || throw(ArgumentError("d must be a multiple of the grid step"))
        sum(g[i] * sur[i + sh] for i in 1:(length(g) - sh))
    end
end

"""
    overtaking(f; d = 1:4, grid)

The overtaking profile of a delay law: `q_d = P(site i + d commits before site i) = P(e_i − e_{i+d} > d)`, for each lag
in `d`. With independent delays, `q_d` is the probability that the organs of any two sites d apart are inverted on the
stem, whatever the other sites do: unlike the parameters of f, it is identified by the orders (`q_1`: neighbour swaps;
`q_2`, `q_3`: the longer displacements that tails produce). It depends on f only through the law of a difference of two
delays, so it carries no late/early asymmetry.
"""
overtaking(f::DelayLaw; d = 1:4, grid = delay_grid(f)) = _overtaking(f, d, grid)

"""
    loglik_order_exact(σ, logg, z_grid, γ)

Exact log P(σ) for sites 1..n with readiness iγ and i.i.d. delay density g:
`P(σ) = ∫_{τ_1 < … < τ_n} ∏_p g(τ_p − σ[p] γ) dτ`, by a forward recursion on a time grid with step Δz. Two times in the
same grid cell are ordered with probability ½ (midpoint rule): counting such ties as ordered would bias log P(σ) upwards
at first order in Δz, while the midpoint rule leaves an error of second order. Grid cells where g is more than e^18
below its peak are skipped. g is interpolated linearly between grid points (exact when γ is a multiple of the grid step, as for γ = 1). Generic in the element type of `logg` (ForwardDiff duals).
"""
function loglik_order_exact(σ::AbstractVector{Int}, logg::AbstractVector{T}, z_grid, γ::Real) where {T <: Real}
    n = length(σ)
    Δz = step(z_grid)
    # the support of g on the grid (a choice of cells, made on the values: it carries no derivative)
    lv = _fullvalue.(logg)
    top = maximum(lv)
    g = exp.(logg) # once per law, not once per cell and organ
    zmin, zmax = z_grid[findfirst(>(top - _LOGG_DROP), lv)], z_grid[findlast(>(top - _LOGG_DROP), lv)]
    tmin = γ * minimum(σ) + zmin
    L = round(Int, (γ * (maximum(σ) - minimum(σ)) + (zmax - zmin)) / Δz) + 2
    # A[kk] = P(the organ below commits before a time in cell kk), stored on the window [pa, pb] of that organ only:
    # 0 below it, 1 above it (no organ below: 1 everywhere)
    A = zeros(T, L)
    α = zeros(T, L)
    pa, pb = 1, 0
    ll = zero(T)
    for p in 1:n
        tp = γ * σ[p]
        ka = max(1, ceil(Int, (tp + zmin - tmin) / Δz) + 1)
        kb = min(L, floor(Int, (tp + zmax - tmin) / Δz) + 1)
        c = zero(T)
        @inbounds for kk in ka:kb
            Ak = kk < pa ? zero(T) : kk > pb ? one(T) : A[kk]
            α[kk] = _interp(g, z_grid, tmin + (kk - 1) * Δz - tp) * Ak
            c += α[kk]
        end
        c *= Δz
        (c > 0) || return T(-Inf)
        ll += log(c)
        acc = zero(T)
        @inbounds for kk in ka:kb
            a = α[kk] * Δz / c
            A[kk] = acc + a / 2 # the cells before, half of cell kk
            acc += a
        end
        pa, pb = ka, kb
    end
    return ll
end
const _LOGG_DROP = 18.0
loglik_order_exact(σ::AbstractVector{Int}, f::DelayLaw; grid = delay_grid(f)) = loglik_order_exact(σ, _logdensity(f, grid), grid, 1.0)

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
`log P_A(σ)` under the propensity model's full cost vector η (length 2k + 1).
"""
function order_samples(post, η::AbstractVector)
    k = (length(η) - 1) ÷ 2
    return map(post) do p
        cm = countmap(p.samples)
        u = unique(p.samples) # in order of first appearance
        cnt = [cm[σ] for σ in u]
        ws = PermanentWorkspace(k, length(first(u)))
        logA = [logprob_permutation(ws, η, σ) for σ in u]
        all(isfinite, logA) || throw(ArgumentError("an order has a displacement beyond the band k = $(k) of η"))
        (; orders = u, w = cnt ./ sum(cnt), logA)
    end
end

"""
    stem_terms(samples, f; grid)
    stem_terms(samples, logP)

Per stem, log p(y | f) − log p(y | A) `= log Σ_j w_j P_f(σ_j) / P_A(σ_j)` over its sampled orders, for a delay law f
(exact order law of f) or any order law given as a callable σ ↦ log P(σ). Threaded over stems.
"""
function stem_terms(samples, logP)
    term(s) = logsumexp([log(s.w[j]) + logP(s.orders[j]) - s.logA[j] for j in eachindex(s.orders)])
    return fetch.([Threads.@spawn term(s) for s in samples]) # concrete element type (Float64 or a dual)
end
stem_terms(samples, f::DelayLaw; grid = delay_grid(f)) = _stem_terms(samples, f, grid)
_stem_terms(samples, f, grid) = (lg = _logdensity(f, grid); stem_terms(samples, σ -> loglik_order_exact(σ, lg, grid, 1.0)))

"log p(y | f) − log p(y | A) summed over stems (positive: f explains the orders better than the propensity model)"
relative_loglik(samples, f; kwargs...) = sum(stem_terms(samples, f; kwargs...))

#################################################################################################################
# fitting and uncertainty
#################################################################################################################

const _NQ = 4 # the overtaking profile reported by delay_law_inference: q_1 … q_4
# the quantities reported by delay_law_inference: the law's reported parameters, the tail masses, the profile
_names(f) = (keys(reported_parameters(f))..., :late, :early, :off, (Symbol(:q, d) for d in 1:_NQ)...)
_derived(f, grid) = (t = _tail_masses(f, 1.0, grid); [values(reported_parameters(f))..., t.late, t.early, t.off, _overtaking(f, 1:_NQ, grid)...])

# a law with mass beyond the grid: the cut-off mass would bias the fit (delay_grid). In general, the mass of the
# outermost cells shows a density that has not decayed; for exponential tails the cut-off mass is known exactly
function _check_grid(f::DelayLaw, grid)
    g = exp.(_logdensity(f, grid)) .* step(grid)
    edge = max(first(g), last(g))
    edge > 1.0e-6 && @warn "the delay density has not decayed at the edge of the grid (mass $(round(edge; sigdigits = 2)) in an edge cell): widen `grid`" f
    return f
end
function _check_grid(f::GaussianExpTails, grid)
    cut = max(f.πL * exp(-last(grid) / f.mL), f.πE * exp(first(grid) / f.mE))
    cut > 1.0e-4 && @warn "a delay tail extends beyond the grid (cut-off mass $(round(cut; sigdigits = 2))): widen `grid`" f
    return f
end

# minimum of obj by Newton's method with a trust region, exact gradient and Hessian by ForwardDiff
function _newton(obj, v0; iterations::Int = 200)
    v = collect(Float64, v0)
    # f_abstol: along a ridge where the likelihood no longer changes (a core collapsing below the grid step), stop
    return _finite_result(optimize(_ad_objective(obj, v), v, NewtonTrustRegion(), Optim.Options(; iterations, g_tol = 1.0e-6, f_abstol = 1.0e-6)))
end

# joint fit: v = [the k shared parameters; the q others of group 1; of group 2; ...]
_group_v(v, g, k, q) = vcat(v[1:k], v[(k + 1 + (g - 1) * q):(k + g * q)])
# the law of group g at the joint parameters v
function _group_law(init, v, g)
    k = nshared_parameters(init)
    return with_free_parameters(init, _group_v(v, g, k, length(free_parameters(init)) - k))
end

"""
    fit_delay_law(samples, init::DelayLaw = GaussianExpTails(:both); grid, iterations = 200)
    fit_delay_law(groups::Vector, init::DelayLaw = GaussianExpTails(:both); grid, iterations = 200)

Delay law maximising `relative_loglik`, by Newton's method with a trust region (exact gradient and Hessian by
ForwardDiff) on its free parameters (`free_parameters`), starting from the law `init`, which also sets the family
and which parameters are free (any `DelayLaw` that defines `free_parameters` and `with_free_parameters`). Returns
(; law, v, init, grid, `rel_ll`, converged), with `v` the free parameters of the fitted law and `grid` the delay
grid used (`delay_grid(init)` by default).

For the default law, the variants name the tails to fit: `fit_delay_law(samples, GaussianExpTails(:late))`, or
`GaussianExpTails(:both, f)` to start from the values of a law `f` (see `GaussianExpTails`).

Given several groups of stems (a vector of `order_samples` results, e.g. a wild type and its mutant), fits one law per
group, with the first `nshared_parameters(init)` free parameters shared and the others per group. For the default law
this is a **shared core sd** with group-specific tails: the hypothesis that the groups differ in their off-time organs,
not in the precision of a typical commitment, and the way to estimate the tails of a group whose own core is not
identified (heavy tails). For another law, the shared parameters are those its author declares with
`nshared_parameters`: a parameter with the same meaning in every group (a scale, say). A law that declares none (the
default) cannot be fitted jointly, and the call throws an `ArgumentError`. Returns (; laws, v, init, grid, `rel_ll`,
`rel_ll_groups`, converged, `shared_core = true`).
"""
function fit_delay_law(samples, init::DelayLaw = GaussianExpTails(:both); grid = delay_grid(init), iterations::Int = 200)
    _is_groups(samples) && return _fit_shared(samples, init; grid, iterations)
    r = _newton(v -> -sum(_stem_terms(samples, with_free_parameters(init, v), grid)), free_parameters(init); iterations)
    v = Optim.minimizer(r)
    law = _check_grid(with_free_parameters(init, v), grid)
    return (; law, v, init, grid, rel_ll = -Optim.minimum(r), converged = Optim.converged(r))
end

_is_groups(x) = !isempty(x) && first(x) isa AbstractVector

function _fit_shared(groups, init; grid, iterations)
    G, k = length(groups), nshared_parameters(init)
    k > 0 || throw(ArgumentError("a joint fit of groups shares parameters: define nshared_parameters for $(nameof(typeof(init)))"))
    v0 = free_parameters(init)
    obj(v) = -sum(sum(_stem_terms(groups[g], _group_law(init, v, g), grid)) for g in 1:G)
    r = _newton(obj, vcat(v0[1:k], repeat(v0[(k + 1):end], G)); iterations)
    v = Optim.minimizer(r)
    laws = [_check_grid(_group_law(init, v, g), grid) for g in 1:G]
    rel_ll_groups = [relative_loglik(groups[g], laws[g]; grid) for g in 1:G]
    return (; laws, v, init, grid, rel_ll = -Optim.minimum(r), rel_ll_groups, converged = Optim.converged(r), shared_core = true)
end

# sandwich covariance of the unconstrained parameters, clustered by stem: H⁻¹ (Σ_s g_s g_sᵀ) H⁻¹, with the per-stem
# scores g_s and the Hessian H of the summed terms exact (ForwardDiff); then the delta method for `derived`
function _sandwich(terms, derived, v)
    Gs = ForwardDiff.jacobian(terms, v)
    H = ForwardDiff.hessian(w -> sum(terms(w)), v)
    Hi = pinv(-(H + H') / 2; rtol = 1.0e-8)
    V = Hi * (Gs' * Gs) * Hi
    J = ForwardDiff.jacobian(derived, v)
    se(C) = sqrt.(max.(diag(J * C * J'), 0.0))
    return (; est = derived(v), se = se(V), se_model = se(Hi), V, Vd = J * V * J')
end

"""
    delay_law_inference(samples, fit; grid)
    delay_law_inference(groups, fit; grid)

`grid` defaults to the grid of the fit (`fit.grid`).

Robust standard errors at a `fit_delay_law` result: sandwich covariance `H⁻¹ (Σ_s g_s g_sᵀ) H⁻¹` of the unconstrained
parameters (per-stem scores `g_s`, Hessian H of the summed terms, exact by ForwardDiff; clustered by stem, so
heterogeneity between stems is allowed), then the delta method. `se_model` is the model-based (−H⁻¹) version. pinv: a
tail with π ≈ 0 leaves its distance unidentified.

`estimate` and `se` hold the law's `reported_parameters` (the core sd σc for `GaussianExpTails`), the tail masses
P(e > 1), P(e < −1), P(|e| > 1), and the overtaking profile `q_1` … `q_4` (`overtaking`); `names` lists them.
`profile = (; d, q, se, V)` repeats the profile with its covariance, for `compare_profiles`. The profile is identified
by the orders; the parameters, the tail masses and the late/early split are readings of the parametric law,
identified only through its shape.

For a joint fit of several groups, `estimate`, `se` and `profile` are vectors (one entry per group), and
`Δprofile` and `Δoff` give each group minus group 1, with standard errors from the joint covariance (the shared
parameters correlate the groups); `Δprofile` also carries its Wald test.
"""
function delay_law_inference(samples, fit; grid = get(fit, :grid, delay_grid(fit.init)))
    init = fit.init
    names = _names(init)
    K = length(names)
    nt(x) = NamedTuple{names}(Tuple(x))
    io = K - _NQ # the off-time fraction's entry in the derived vector
    qi = (io + 1):K # the profile's entries
    if get(fit, :shared_core, false)
        G = length(samples)
        terms(w) = reduce(vcat, [_stem_terms(samples[g], _group_law(init, w, g), grid) for g in 1:G])
        derived(w) = reduce(vcat, [_derived(_group_law(init, w, g), grid) for g in 1:G])
        r = _sandwich(terms, derived, fit.v)
        at(g, i) = K * (g - 1) .+ i
        blk(x, g) = nt(x[at(g, 1:K)])
        profile = [(; d = 1:_NQ, q = r.est[at(g, qi)], se = r.se[at(g, qi)], V = r.Vd[at(g, qi), at(g, qi)]) for g in 1:G]
        Δprofile = map(2:G) do g
            a, b = at(1, qi), at(g, qi)
            _wald(r.est[b] .- r.est[a], r.Vd[b, b] .+ r.Vd[a, a] .- r.Vd[b, a] .- r.Vd[a, b], 1:_NQ)
        end
        Δoff = [(estimate = r.est[at(g, io)] - r.est[io], se = sqrt(max(r.Vd[at(g, io), at(g, io)] + r.Vd[io, io] - 2r.Vd[at(g, io), io], 0.0))) for g in 2:G]
        return (;
            names, estimate = [blk(r.est, g) for g in 1:G], se = [blk(r.se, g) for g in 1:G],
            se_model = [blk(r.se_model, g) for g in 1:G], profile, Δprofile, Δoff, r.V,
        )
    end
    r = _sandwich(w -> _stem_terms(samples, with_free_parameters(init, w), grid), w -> _derived(with_free_parameters(init, w), grid), fit.v)
    profile = (; d = 1:_NQ, q = r.est[qi], se = r.se[qi], V = r.Vd[qi, qi])
    return (; names, estimate = nt(r.est), se = nt(r.se), se_model = nt(r.se_model), profile, r.V)
end

# difference Δ with covariance V over lags d: estimates, standard errors and the Wald χ² test (pinv: a lag with no
# overtaking at all carries no information)
function _wald(Δ, V, d)
    V = (V + V') / 2
    chi2 = max(Δ' * pinv(V; rtol = 1.0e-10) * Δ, 0.0)
    df = rank(V; rtol = 1.0e-10)
    return (; d, diff = Δ, se = sqrt.(max.(diag(V), 0.0)), V, chi2, df, p = df > 0 ? ccdf(Chisq(df), chi2) : 1.0)
end

"""
    overtaking_profile(samples; d = 1:4)

The overtaking profile read directly from the posterior orders, with no delay law: `q_d` is the fraction of pairs of
sites d apart whose organs are inverted on the stem, averaged over each stem's posterior samples (`order_samples`)
and pooled over stems (Σ expected inversions / Σ pairs). Standard errors and the covariance `V` across lags are
clustered by stem (linearised ratio estimator), so they allow heterogeneity between stems. Returns (; d, q, se, V).

Under independent delays this estimates `overtaking(f)` for the true law, whatever f is: a check of the fitted law,
and a comparison between groups (`compare_profiles`) that needs no delay law at all.
"""
function overtaking_profile(samples; d = 1:4)
    ds = collect(d)
    S = length(samples)
    c = zeros(S, length(ds)) # expected inversions per stem and lag
    m = zeros(S, length(ds)) # pairs per stem and lag
    for (s, smp) in enumerate(samples)
        n = length(first(smp.orders))
        for (σ, w) in zip(smp.orders, smp.w)
            pos = invperm(σ)
            for (k, dd) in enumerate(ds)
                c[s, k] += w * count(pos[i + dd] < pos[i] for i in 1:(n - dd))
            end
        end
        m[s, :] .= max.(n .- ds, 0)
    end
    M = vec(sum(m; dims = 1))
    q = vec(sum(c; dims = 1)) ./ M
    R = (c .- m .* q') ./ M' # linearised contribution of each stem
    V = S / max(S - 1, 1) .* (R' * R)
    return (; d = ds, q, se = sqrt.(diag(V)), V)
end

"""
    compare_profiles(a, b; d = 1:3)

Wald test of two independent groups' overtaking profiles (each from `overtaking_profile` or the `profile` of
`delay_law_inference`), over the lags `d`: `diff` = b − a with its standard errors, χ² with one degree of freedom per
informative lag, and its p-value. The default stops at d = 3: an overtaking at distance d needs two displacements that
differ by more than d, so with the default band k = 4 the lag 4 rests on displacements near the edge of the band, where
the band and the prior of the propensity model weigh most. For groups fitted jointly with a shared core,
use the `Δprofile` of `delay_law_inference` instead (it accounts for their correlation).
"""
function compare_profiles(a, b; d = 1:3)
    ia, ib = [findfirst(==(x), a.d) for x in d], [findfirst(==(x), b.d) for x in d]
    (any(isnothing, ia) || any(isnothing, ib)) && throw(ArgumentError("lags $(d) not in both profiles"))
    return _wald(b.q[ib] .- a.q[ia], b.V[ib, ib] .+ a.V[ia, ia], collect(d))
end

"""
    shared_core_test(groups, init::DelayLaw = GaussianExpTails(:both); grid)

Likelihood ratio of shared parameters for all groups (`fit_delay_law(groups, init)`: one core sd for the default law)
against one set per group (`fit_delay_law` on each), with the other parameters (the tails) per group in both. For any
other law, the shared parameters are those it declares with `nshared_parameters` (none by default: an `ArgumentError`).
Returns (; separate, shared, lr, df, p): `p` from χ² with (G − 1) × `nshared_parameters(init)` degrees of freedom,
which can be optimistic when the tails are heavy (a permutation of the group labels among stems calibrates it).
"""
function shared_core_test(groups, init::DelayLaw = GaussianExpTails(:both); grid = delay_grid(init))
    separate = [fit_delay_law(g, init; grid) for g in groups]
    shared = fit_delay_law(groups, init; grid)
    lr = max(2 * (sum(f.rel_ll for f in separate) - shared.rel_ll), 0.0)
    df = (length(groups) - 1) * nshared_parameters(init)
    return (; separate, shared, lr, df, p = ccdf(Chisq(df), lr))
end
