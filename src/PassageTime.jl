"""
    PassageTime

Commitment-time (first-passage) law of the noisy fold, in noise units:

- plateau (model M1):  dX = (μ + X²) ds + dW, margin μ constant
- linear ramp (M2):    dX = (μ₀ + ν s + X²) ds + dW, margin rising at rate ν from deep below the fold

Both are solved on the Fokker–Planck equation in X (finite volumes, Scharfetter–Gummel fluxes, absorption at the exit).
The plateau is stepped exactly in time (matrix exponential); the ramp by implicit Euler with Richardson extrapolation.
Each law is stored standardised, z = (s − mean) / sd, on a fixed z grid (`PassageTable`; for the ramp, `μ_grid` holds
log ν). Readers: `FoldPassageTime` (a Distributions.jl distribution), `passage_column`, `passage_sampler` and
`passage_logg` (bicubic log density in (shape, z), smooth for gradients and Hessians).
"""
module PassageTime

using LinearAlgebra, Random, Serialization
using Distributions, LogExpFunctions
using Bijectors: Bijectors
using ..Common: LOGFLOOR, _keys_w, _keys_index

export PassageTable, fp_generator, fp_moments, passage_table, ramp_start, ramp_passage_column, ramp_passage_table,
       FoldPassageTime, passage_column, passage_sampler, passage_logg, save_passage_table, load_passage_table

#################################################################################################################
# plateau: dX = (μ + X²) ds + dW
#################################################################################################################

# Bernoulli function x / (exp(x) - 1) used by the Scharfetter–Gummel flux
bernoulli_fn(x) = (abs(x) < 1e-8 ? 1.0 - x / 2 : x / expm1(x))

# face flux coefficients (a, b) such that J = a q_left + b q_right for cell masses q = h p,
# with J = V p - D ∂X p, V = μ + X², D = 1/2 (Scharfetter–Gummel: exact for V, D constant on the face)
function _sg_face(Xf, μ, hc, h)
    D = 0.5
    V = μ + Xf ^ 2
    Pe = V * hc / D
    return (D / hc * bernoulli_fn(-Pe) / h, -D / hc * bernoulli_fn(Pe) / h)
end

"""
    fp_generator(μ; h, X0, Xc, Xb)

Finite-volume (Scharfetter–Gummel) generator of the cell masses q on X ∈ [X0 - 2, Xc], uniform cells
of width ≈ h, zero flux at the left end and absorption at Xc. Returns (L, c_out, j0): dq/ds = L q, the
passage density is f(s) = c_out q_N(s), and the initial mass sits in cell j0.

X0 = -Inf, Xc = Inf (reference family) is replaced by X0 = -Xb, Xc = Xb: outside |X| > Xb the
motion is deterministic up to a timing variance ≈ 2 ∫_Xb^∞ X⁻⁶ dX = 0.4 Xb⁻⁵ (2e-6 for Xb = 10),
so the passage law equals the reference one up to a constant shift, irrelevant once standardised.
The X coordinate (rather than θ = 2 atan X) keeps the upwind numerical diffusion where the speed
X² is large, so its contribution to the timing variance, ≈ h ∫ X⁻⁴ dX, stays negligible.
"""
function fp_generator(μ::Real; h::Real = 0.02, X0::Real = -Inf, Xc::Real = Inf, Xb::Real = 10.0)
    X0 = max(X0, -Xb)
    Xc = min(Xc, Xb)
    Xl = X0 - 2.0
    N = round(Int, (Xc - Xl) / h)
    h = (Xc - Xl) / N
    dl = zeros(N - 1)
    dd = zeros(N)
    du = zeros(N - 1)
    for j in 1:(N - 1)
        a, b = _sg_face(Xl + j * h, μ, h, h) # face between cells j and j + 1
        dd[j] -= a
        du[j] -= b
        dl[j] += a
        dd[j + 1] += b
    end
    c_out, _ = _sg_face(Xc, μ, h / 2, h) # Dirichlet q = 0 at Xc, half-cell distance
    dd[N] -= c_out
    j0 = clamp(floor(Int, (X0 - Xl) / h) + 1, 1, N)
    return (Tridiagonal(dl, dd, du), c_out, j0)
end

"exact mean and sd of the passage time of the discretised process (moments of the phase-type law: −1ᵀL⁻¹q₀, 2·1ᵀL⁻²q₀)"
function fp_moments(L::Tridiagonal, j0::Int)
    N = size(L, 1)
    q0 = zeros(N)
    q0[j0] = 1.0
    F = lu(L)
    u1 = F \ q0 # = L⁻¹ q0
    u2 = F \ u1
    m = -sum(u1)
    m2 = 2 * sum(u2)
    return (m, sqrt(max(m2 - m ^ 2, 0.0)))
end

struct PassageTable
    μ_grid::Vector{Float64} # plateau margin μ, or log ν for a ramp table
    z_grid::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int}
    logg::Matrix{Float64} # log density of z = (s - m) / sd, length(z_grid) × length(μ_grid)
    logS::Matrix{Float64} # log survival of z
    logF::Matrix{Float64} # log CDF of z
    mean::Vector{Float64} # passage time mean (t_S units)
    sd::Vector{Float64} # passage time sd (t_S units)
    X0::Float64
    Xc::Float64
end

# The left tail of the density is resolved only down to ~1e-13 of its maximum (round-off of exp(L Δs)).
# Below, extrapolate log g quadratically from the first reliable point, but never before s = 0 (i_s0).
function _repair_left_tail!(logg::AbstractVector, Δz, i_s0::Int; rel_tol = 1e-11)
    lmax = maximum(logg)
    i_c = findfirst(x -> x > lmax + log(rel_tol), logg)
    (i_c === nothing) && return logg
    i_c += 2
    (i_c - 1 ≤ i_s0) && return logg
    b = max((logg[i_c + 1] - logg[i_c - 1]) / (2 * Δz), 1.0)
    c = min((logg[i_c + 1] - 2 * logg[i_c] + logg[i_c - 1]) / Δz ^ 2, 0.0)
    for i in i_s0:(i_c - 1)
        δ = (i - i_c) * Δz
        logg[i] = max(logg[i_c] + b * δ + 0.5 * c * δ ^ 2, LOGFLOOR)
    end
    return logg
end

"""
    passage_table(μ_grid; Δz, z_range, h, X0, Xc)

Tabulate the standardised passage law (log density, log survival, log CDF) on a fixed z grid for
each plateau margin of μ_grid. The time stepping is exact (matrix exponential of the generator), so
the only discretisation is the spatial one (cell width h in X).
"""
function passage_table(μ_grid::AbstractVector; Δz::Real = 0.04, z_range = (-8.0, 40.0), h::Real = 0.02, X0::Real = -Inf, Xc::Real = Inf)
    z_grid = range(z_range[1], z_range[2]; step = Δz)
    nz = length(z_grid)
    nμ = length(μ_grid)
    logg = fill(LOGFLOOR, nz, nμ)
    logS = zeros(nz, nμ)
    logF = fill(LOGFLOOR, nz, nμ)
    m_vec = zeros(nμ)
    sd_vec = zeros(nμ)
    nblas = BLAS.get_num_threads()
    (Threads.nthreads() > 1) && BLAS.set_num_threads(1)
    Threads.@threads for iμ in 1:nμ
        L, c_out, j0 = fp_generator(μ_grid[iμ]; h = h, X0 = X0, Xc = Xc)
        N = size(L, 1)
        m, sd = fp_moments(L, j0)
        m_vec[iμ], sd_vec[iμ] = m, sd
        Ld = Matrix(L)
        q = zeros(N)
        q[j0] = 1.0
        # first grid point at s ≥ 0
        i_first = max(1, ceil(Int, (-m / sd - z_grid[1]) / Δz - 1e-9) + 1)
        s_first = m + sd * z_grid[i_first]
        q = exp(Ld .* s_first) * q
        E = exp(Ld .* (sd * Δz))
        q_next = similar(q)
        for iz in i_first:nz
            S = sum(q)
            f = c_out * q[N]
            logg[iz, iμ] = (f > 0 ? max(log(f * sd), LOGFLOOR) : LOGFLOOR)
            logS[iz, iμ] = (S > 0 ? max(log(S), LOGFLOOR) : LOGFLOOR)
            logF[iz, iμ] = log1mexp(min(logS[iz, iμ], -1e-300))
            mul!(q_next, E, q)
            q, q_next = q_next, q
        end
        _repair_left_tail!(view(logg, :, iμ), Δz, i_first)
        # CDF from the repaired density in the left tail (cumulative trapezoid), where 1 - S is not resolved
        acc = -Inf
        for iz in 1:nz
            iz > 1 && (acc = logaddexp(acc, logaddexp(logg[iz - 1, iμ], logg[iz, iμ]) + log(Δz / 2)))
            logF[iz, iμ] < -25 && (logF[iz, iμ] = max(acc, LOGFLOOR))
        end
    end
    BLAS.set_num_threads(nblas)
    return PassageTable(collect(Float64, μ_grid), z_grid, logg, logS, logF, m_vec, sd_vec, Float64(X0), Float64(Xc))
end

#################################################################################################################
# linear ramp: dX = (μ₀ + ν s + X²) ds + dW
#################################################################################################################
#
# The margin rises at rate ν (noise units). The ramp starts deep below the fold, at μ₀ = -max(8, 3 ν^(2/3)), so the
# cell tracks its low state |μ|^(3/2) ≫ ν (adiabatic) and noise escape at the start is negligible (rate ∝
# exp(-8/3 |μ₀|^(3/2))). The standardised passage law then depends on ν only (the universal ramp); the start only adds
# a lag (a shallower start changes only log g < -19). A ramp table is a PassageTable whose `μ_grid` holds log ν:
# everything downstream (propensity table, angle models) uses it unchanged, with log ν in place of μ.

ramp_start(ν) = -max(8.0, 3.0 * ν ^ (2 / 3))

# Scharfetter–Gummel generator at margin μ written in place (same scheme as fp_generator); returns c_out
function _fp_coeffs!(dl, dd, du, μ, Xl, h)
    N = length(dd)
    fill!(dl, 0.0); fill!(dd, 0.0); fill!(du, 0.0)
    for j in 1:(N - 1)
        a, b = _sg_face(Xl + j * h, μ, h, h)
        dd[j] -= a; du[j] -= b; dl[j] += a; dd[j + 1] += b
    end
    c_out, _ = _sg_face(Xl + N * h, μ, h / 2, h)
    dd[N] -= c_out
    return c_out
end

# solve (I - Δs L) x = q in place of q (Thomas algorithm; I - Δs L is an M-matrix, so x stays ≥ 0)
function _ie_solve!(q, dl, dd, du, Δs, c)
    N = length(q)
    b1 = 1 - Δs * dd[1]
    c[1] = -Δs * du[1] / b1
    q[1] /= b1
    for j in 2:N
        a = -Δs * dl[j - 1]
        m = 1 - Δs * dd[j] - a * c[j - 1]
        (j < N) && (c[j] = -Δs * du[j] / m)
        q[j] = (q[j] - a * q[j - 1]) / m
    end
    for j in (N - 1):-1:1
        q[j] -= c[j] * q[j + 1]
    end
    return q
end

# implicit Euler run on s ∈ [0, nsteps Δs]: log density and log survival at s_n = n Δs
function _ramp_run(ν, Δs, nsteps; h = 0.02, Xb = 10.0)
    μ0 = ramp_start(ν)
    X0 = -sqrt(-μ0)
    Xl = X0 - 3.0
    N = round(Int, (Xb - Xl) / h)
    h = (Xb - Xl) / N
    dl, dd, du, c = zeros(N - 1), zeros(N), zeros(N - 1), zeros(N)
    q = zeros(N)
    q[clamp(floor(Int, (X0 - Xl) / h) + 1, 1, N)] = 1.0
    logf = fill(LOGFLOOR, nsteps + 1)
    logS = zeros(nsteps + 1)
    for n in 1:nsteps
        c_out = _fp_coeffs!(dl, dd, du, μ0 + ν * n * Δs, Xl, h)
        _ie_solve!(q, dl, dd, du, Δs, c)
        f, S = c_out * q[N], sum(q)
        logf[n + 1] = (f > 0 ? max(log(f), LOGFLOOR) : LOGFLOOR)
        logS[n + 1] = (S > 0 ? max(log(S), LOGFLOOR) : LOGFLOOR)
        (logS[n + 1] ≤ LOGFLOOR) && (logf[n + 2:end] .= LOGFLOOR; logS[n + 2:end] .= LOGFLOOR; break)
    end
    return logf, logS
end

# mean and sd of a density given by its log on a uniform grid s_n = (n - 1) Δs (trapezoid)
function _moments_from_log(logf, Δs)
    s = (0:(length(logf) - 1)) .* Δs
    w = exp.(logf .- maximum(logf))
    w[1] /= 2; w[end] /= 2
    Z = sum(w)
    m = sum(s .* w) / Z
    return (m, sqrt(sum((s .- m) .^ 2 .* w) / Z))
end

"""
    ramp_passage_column(ν, z_grid; nsub = 4, h = 0.02)

Standardised passage law of the linear ramp at rate ν on `z_grid`: (; logg, logS, logF, mean, sd), mean and sd in
noise time units from the start of the ramp. Implicit Euler in time (positivity preserving), at steps sd·Δz/nsub and
sd·Δz/(2 nsub), combined by Richardson extrapolation of the logs (second order).
"""
function ramp_passage_column(ν::Real, z_grid; nsub::Int = 4, h::Real = 0.02)
    # pass 1: coarse run to locate the law
    T = -ramp_start(ν) / ν + 10 * ν ^ (-1 / 3) + 10.0
    local m1, sd1
    while true
        Δs0 = T / 4000
        lf, lS = _ramp_run(ν, Δs0, 4000; h = h)
        if lS[end] < log(1e-10)
            m1, sd1 = _moments_from_log(lf, Δs0)
            break
        end
        T *= 2
    end
    # pass 2: fine runs on a common grid, up to the end of the z grid
    Δz = step(z_grid)
    Δs = sd1 * Δz / nsub
    nsteps = ceil(Int, (m1 + sd1 * (last(z_grid) + 2)) / Δs)
    lf1, lS1 = _ramp_run(ν, Δs, nsteps; h = h)
    lf2, lS2 = _ramp_run(ν, Δs / 2, 2 * nsteps; h = h)
    rich(a, b) = (a > LOGFLOOR + 50 && b > LOGFLOOR + 50 ? 2 * b - a : b)
    lf = [rich(lf1[n], lf2[2n - 1]) for n in eachindex(lf1)]
    lS = [min(rich(lS1[n], lS2[2n - 1]), 0.0) for n in eachindex(lS1)]
    m, sd = _moments_from_log(lf, Δs)
    # resample on the standardised grid
    nz = length(z_grid)
    logg, logS, logF = fill(LOGFLOOR, nz), zeros(nz), fill(LOGFLOOR, nz)
    for (iz, z) in enumerate(z_grid)
        s = m + sd * z
        s ≤ 0 && continue
        x = s / Δs + 1
        i = min(floor(Int, x), length(lf) - 1)
        t = x - i
        logg[iz] = max((1 - t) * lf[i] + t * lf[i + 1] + log(sd), LOGFLOOR)
        logS[iz] = max((1 - t) * lS[i] + t * lS[i + 1], LOGFLOOR)
        logF[iz] = log1mexp(min(logS[iz], -1e-300))
    end
    # left tail of the CDF from the density, where 1 - S is not resolved
    acc = -Inf
    for iz in 1:nz
        iz > 1 && (acc = logaddexp(acc, logaddexp(logg[iz - 1], logg[iz]) + log(Δz / 2)))
        logF[iz] < -25 && (logF[iz] = max(acc, LOGFLOOR))
    end
    return (; logg, logS, logF, mean = m, sd)
end

"""
    ramp_passage_table(logν_grid; Δz = 0.04, z_range = (-40, 12), nsub = 4, h = 0.02)

Passage table of the linear ramp for each log ν of `logν_grid` (uniform). The grid extends far to the left because
the ramp law has an early (Gumbel-like, exponential) tail and a double-exponentially light late tail.
"""
function ramp_passage_table(logν_grid::AbstractVector; Δz::Real = 0.04, z_range = (-40.0, 12.0), nsub::Int = 4, h::Real = 0.02)
    z_grid = range(z_range[1], z_range[2]; step = Δz)
    nz, nν = length(z_grid), length(logν_grid)
    logg, logS, logF = zeros(nz, nν), zeros(nz, nν), zeros(nz, nν)
    m_vec, sd_vec = zeros(nν), zeros(nν)
    Threads.@threads for i in 1:nν
        col = ramp_passage_column(exp(logν_grid[i]), z_grid; nsub = nsub, h = h)
        logg[:, i] .= col.logg; logS[:, i] .= col.logS; logF[:, i] .= col.logF
        m_vec[i], sd_vec[i] = col.mean, col.sd
    end
    return PassageTable(collect(Float64, logν_grid), z_grid, logg, logS, logF, m_vec, sd_vec, -Inf, Inf)
end

#################################################################################################################
# readers
#################################################################################################################

# linear interpolation of a tabulated column in z (constant extrapolation, values are log-probabilities)
@inline function _interp_z(col::AbstractVector, z_grid, z)
    x = (z - first(z_grid)) / step(z_grid) + 1
    nz = length(z_grid)
    x ≤ 1 && return col[1]
    x ≥ nz && return col[nz]
    i = floor(Int, x)
    t = x - i
    return (1 - t) * col[i] + t * col[i + 1]
end

# log survival with exponential extrapolation beyond the right end of the grid
@inline function _logS_z(col::AbstractVector, z_grid, z)
    nz = length(z_grid)
    if z > last(z_grid)
        slope = (col[nz] - col[nz - 1]) / step(z_grid)
        return max(col[nz] + slope * (z - last(z_grid)), LOGFLOOR)
    end
    return _interp_z(col, z_grid, z)
end

"""
    FoldPassageTime(tab, μ; scale = 1)

Distribution of the commitment (first-passage) time of the noisy fold dX = (μ + X²) ds + dW on a plateau
margin μ (noise units), read from the passage table `tab` (cubic interpolation in μ, linear in time).
`scale` is the noise time unit t_S expressed in the user's time unit: T = scale · s.
Support (0, ∞); its bijector is log, as for the other distributions on the positive half-line.
"""
struct FoldPassageTime{T <: Real} <: ContinuousUnivariateDistribution
    tab::PassageTable
    μ::T
    scale::T
end
FoldPassageTime(tab::PassageTable, μ::Real; scale::Real = 1.0) = FoldPassageTime(tab, promote(μ, scale)...)

Base.minimum(::FoldPassageTime) = 0.0
Base.maximum(::FoldPassageTime) = Inf
Distributions.insupport(::FoldPassageTime, x::Real) = x > 0
Bijectors.bijector(::FoldPassageTime) = Bijectors.elementwise(log)

# cubic (Keys) interpolation in μ of a per-margin value v(iμ)
function _μinterp(v, tab::PassageTable, μ)
    iμ, tμ, _ = _keys_index(tab.μ_grid, μ)
    w = _keys_w(tμ)
    nμ = length(tab.μ_grid)
    return sum(w[a] * v(clamp(iμ + a - 1, 1, nμ)) for a in 1:4)
end
# passage-time mean and sd in noise units (interpolated on the log scale)
_fold_mean(d::FoldPassageTime) = exp(_μinterp(i -> log(d.tab.mean[i]), d.tab, d.μ))
_fold_sd(d::FoldPassageTime) = exp(_μinterp(i -> log(d.tab.sd[i]), d.tab, d.μ))
_fold_z(d::FoldPassageTime, x) = (x / d.scale - _fold_mean(d)) / _fold_sd(d)

Distributions.mean(d::FoldPassageTime) = d.scale * _fold_mean(d)
Distributions.std(d::FoldPassageTime) = d.scale * _fold_sd(d)
Distributions.var(d::FoldPassageTime) = std(d) ^ 2

function Distributions.logpdf(d::FoldPassageTime, x::Real)
    (x > 0) || return oftype(float(x), -Inf)
    (; tab) = d
    z = _fold_z(d, x)
    return _μinterp(i -> _interp_z(view(tab.logg, :, i), tab.z_grid, z), tab, d.μ) - log(_fold_sd(d) * d.scale)
end
function Distributions.logccdf(d::FoldPassageTime, x::Real)
    (x > 0) || return zero(float(x))
    z = _fold_z(d, x)
    return min(_μinterp(i -> _logS_z(view(d.tab.logS, :, i), d.tab.z_grid, z), d.tab, d.μ), 0.0)
end
function Distributions.logcdf(d::FoldPassageTime, x::Real)
    (x > 0) || return oftype(float(x), -Inf)
    z = _fold_z(d, x)
    return min(_μinterp(i -> _interp_z(view(d.tab.logF, :, i), d.tab.z_grid, z), d.tab, d.μ), 0.0)
end
Distributions.pdf(d::FoldPassageTime, x::Real) = exp(logpdf(d, x))
Distributions.cdf(d::FoldPassageTime, x::Real) = exp(logcdf(d, x))
Distributions.ccdf(d::FoldPassageTime, x::Real) = exp(logccdf(d, x))

# inverse CDF by bisection in the standardised time
function Distributions.quantile(d::FoldPassageTime, p::Real)
    m, sd = _fold_mean(d), _fold_sd(d)
    lo, hi = max(first(d.tab.z_grid), -m / sd), last(d.tab.z_grid)
    for _ in 1:60
        z = 0.5 * (lo + hi)
        (cdf(d, d.scale * (m + sd * z)) < p) ? (lo = z) : (hi = z)
    end
    return d.scale * (m + sd * 0.5 * (lo + hi))
end
Base.rand(rng::AbstractRNG, d::FoldPassageTime) = quantile(d, rand(rng))

"cubic (Keys) interpolation of a table field (default the log density) in μ: the whole z column at an arbitrary μ"
function passage_column(tab::PassageTable, μ::Real; field::Symbol = :logg)
    M = getfield(tab, field)
    iμ, tμ, _ = _keys_index(tab.μ_grid, μ)
    w = _keys_w(tμ)
    nμ = length(tab.μ_grid)
    col = zeros(size(M, 1))
    for a in 1:4
        ia = clamp(iμ + a - 1, 1, nμ)
        col .+= w[a] .* view(M, :, ia)
    end
    return col
end

"inverse-CDF sampler of the standardised passage time z at shape μ: returns a function rng ↦ z"
function passage_sampler(tab::PassageTable, μ::Real)
    lg = passage_column(tab, μ)
    cdf = cumsum(exp.(lg))
    cdf ./= cdf[end]
    z_grid = tab.z_grid
    return rng -> begin
        u = rand(rng)
        i = searchsortedfirst(cdf, u)
        z_grid[clamp(i, 1, length(z_grid))] - step(z_grid) * rand(rng)
    end
end

"""
    passage_logg(tab, shape, z)

log g(z) of the standardised passage law, bicubic (Keys) in (shape, z), generic in the number type (smooth gradients
and Hessians in shape and z). Bilinear on the central cell when the stencil touches the floor (the left edge s < 0 is a
cliff to LOGFLOOR that a cubic would overshoot). log g = LOGFLOOR left of the grid, linear extrapolation of the last
slope right of it.
"""
function passage_logg(tab::PassageTable, shape, z)
    (; μ_grid, z_grid, logg) = tab
    nz, nμ = size(logg)
    z < first(z_grid) && return oftype(float(z), LOGFLOOR)
    iμ, tμ, _ = _keys_index(μ_grid, shape)
    if z > last(z_grid)
        Δz = step(z_grid)
        v = zero(float(z)) + zero(tμ)
        for (a, w) in zip(1:4, _keys_w(tμ))
            ia = clamp(iμ + a - 1, 1, nμ)
            v += w * (logg[nz, ia] + (logg[nz, ia] - logg[nz - 1, ia]) / Δz * (z - last(z_grid)))
        end
        return max(v, LOGFLOOR)
    end
    iz, tz, _ = _keys_index(z_grid, z)
    low = false
    @inbounds for a in 1:4, b in 1:4
        logg[clamp(iz + b - 1, 1, nz), clamp(iμ + a - 1, 1, nμ)] < -300 && (low = true)
    end
    if low
        i0, i1 = clamp(iz + 1, 1, nz), clamp(iz + 2, 1, nz)
        j0, j1 = clamp(iμ + 1, 1, nμ), clamp(iμ + 2, 1, nμ)
        return (1 - tz) * (1 - tμ) * logg[i0, j0] + tz * (1 - tμ) * logg[i1, j0] + (1 - tz) * tμ * logg[i0, j1] + tz * tμ * logg[i1, j1]
    end
    wμ, wz = _keys_w(tμ), _keys_w(tz)
    v = zero(tz) + zero(tμ)
    @inbounds for a in 1:4, b in 1:4
        v += wμ[a] * wz[b] * logg[clamp(iz + b - 1, 1, nz), clamp(iμ + a - 1, 1, nμ)]
    end
    return v
end

#################################################################################################################
# storage as plain arrays (independent of the struct definition)
#################################################################################################################

_table_dict(tab::PassageTable) = Dict{Symbol, Any}(:μ_grid => tab.μ_grid, :z_first => first(tab.z_grid), :z_step => step(tab.z_grid),
    :z_length => length(tab.z_grid), :logg => tab.logg, :logS => tab.logS, :logF => tab.logF, :mean => tab.mean, :sd => tab.sd,
    :X0 => tab.X0, :Xc => tab.Xc)
_table_from_dict(d) = PassageTable(d[:μ_grid], range(d[:z_first]; step = d[:z_step], length = d[:z_length]), d[:logg], d[:logS], d[:logF],
    d[:mean], d[:sd], d[:X0], d[:Xc])

save_passage_table(path::AbstractString, tab::PassageTable) = serialize(path, _table_dict(tab))
load_passage_table(path::AbstractString) = _table_from_dict(deserialize(path))

end # module PassageTime
