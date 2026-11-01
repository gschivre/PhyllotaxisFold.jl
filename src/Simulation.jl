# Simulation
#
# Simulated stems: from parameters (`simulate_stems(η; ...)`; the delay-law version is in CommitmentTiming.jl), from a
# fitted model (`simulate_stems(fit)`: stems like the data, for posterior predictive checks), and the divergence angles
# of a stem (`divergence_angles`).

"""
    simulate_stems(...)

Simulated stems (angles, true orders, twist rates). Methods: `simulate_stems(η; ...)` (orders from the banded permanent
with log-propensities η), `simulate_stems(f::DelayLaw; ...)` (orders from independent commitment delays f), both with
the same angle and twist options, and `simulate_stems(fit)` (stems like the data, from a fitted model).
"""
function simulate_stems end

# the angles of one stem of order σ (twist coordinate x): κ ~ LogNormal(logμκ, σκ), the golden spiral with probability
# α (else the Lucas one), uniform ϕ and orientation, twist ω ~ N(torsion_mean, torsion_sd²) degrees per unit of x, and
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
    ω = torsion_mean + torsion_sd * randn(rng)
    bad = [rand(rng) < outlier for _ in 1:n]
    # (the von Mises draw lies within π of its centre: wrapped to [0, 2π), like measured azimuths)
    y = [bad[p] ? 2π * rand(rng) : mod2pi(rand(rng, VonMises(mod2pi(ϕ + o * (σ[p] - 1) * ψ + deg2rad(ω) * x[p]), κ))) for p in 1:n]
    return (; y, ω, κ, outliers = findall(bad), ψ, o, ϕ)
end

"""
    simulate_stems(η; plants = 100, organs = 50, x = nothing, logμκ = 1.5, σκ = 0.0, α = 0.95, outlier = 0.0,
        torsion_mean = 0.0, torsion_sd = 0.0, rng)

Simulate from the propensity model: the order of each stem is an exact draw from the banded permanent with
log-propensities η (`η_1..η_k`, symmetric, or the full 2k + 1 vector, recognised by its middle entry `η_0 = 0`),
`κ_s ~ LogNormal(logμκ, σκ)` per stem, and
the angles are `y_p ~ vM(ϕ + o (σ[p] − 1) ψ + ω_s x_p, κ_s)`, replaced with probability `outlier` by a uniform angle.
`organs`: one count, or one per stem; `x`: the twist coordinate of each stem (default: the positions 0, 1, …, or
`torsion_coordinate(internodes)` for the stem length in mm); twist `ω_s ~ Normal(torsion_mean, torsion_sd)` degrees per
unit of x. The precision of the angles: a von Mises of concentration κ has a circular sd of about `1 / √κ` radians
(κ = 6: 23°). Returns (; angles, orders, torsion, κ, outliers, ψ, o, ϕ) (angles in [0, 2π), radians; outliers: the positions whose angle is
uniform; ψ, o, ϕ: the spiral of each stem, its canonical angle, orientation and offset, so that site j sits at
`ϕ + o (j − 1) ψ` before the twist).
"""
function simulate_stems(
        η::AbstractVector{<:Real}; plants::Int = 100, organs = 50, x = nothing, logμκ::Real = 1.5, σκ::Real = 0.0,
        α::Real = 0.95, outlier::Real = 0.0, torsion_mean::Real = 0.0, torsion_sd::Real = 0.0, rng::AbstractRNG = Random.default_rng()
    )
    # a cost η_d (d ≠ 0) is negative, never 0: an η_1..η_k of odd length is not taken for the full vector
    ηf = (isodd(length(η)) && iszero(η[(end + 1) ÷ 2]) ? collect(Float64, η) : vcat(reverse(η), 0.0, η))
    k = length(ηf) ÷ 2
    ns = organs isa Integer ? fill(organs, plants) : collect(organs)
    xs = (x === nothing ? [torsion_coordinate(n) for n in ns] : x)
    (length(ns) == plants && length(xs) == plants && all(length.(xs) .== ns)) ||
        throw(ArgumentError("organs and x: one entry per stem, x with one coordinate per organ"))
    ws = PermanentWorkspace(Float64, k, maximum(ns); logspace = true)
    toeplitz_weights!(ws, ηf, maximum(ns))
    out = map(1:plants) do s
        σ = sample_permutation!(rng, ws, ns[s])
        (; σ, _simulate_angles(rng, σ, xs[s]; logμκ, σκ, α, outlier, torsion_mean, torsion_sd)...)
    end
    return (;
        angles = [z.y for z in out], orders = [z.σ for z in out], torsion = [z.ω for z in out], κ = [z.κ for z in out],
        outliers = [z.outliers for z in out], ψ = [z.ψ for z in out], o = [z.o for z in out], ϕ = [z.ϕ for z in out],
    )
end

"""
    simulate_stems(fit::PhyllotaxisFit; laplace = false, rng)

Stems like the data of the fit, drawn from the fitted model: as many stems, with the same numbers of organs and the same
twist coordinates, and the fitted costs, precision law, spiral probability, outlier probability and twist law. With
`laplace = true`, the parameters are first drawn from the Laplace approximation of the posterior (one draw per call),
so that replicates carry the uncertainty of the fit. Returns the result of `simulate_stems(η; ...)` with the parameters
used, `θ` (`constrained_parameters`).

A posterior predictive check compares a statistic of the data with its values on replicates, e.g. the divergence
angles (`divergence_angles`) of the corrected stems against those of `[simulate_stems(fit; laplace = true) for _ in 1:100]`.
"""
function simulate_stems(fit::PhyllotaxisFit; laplace::Bool = false, rng::AbstractRNG = Random.default_rng())
    m = fit.model
    u = fit.u
    if laplace
        _negH_posdef(fit) || throw(ArgumentError("the Hessian of the fit is not negative definite: no Laplace draws"))
        u = u .+ cholesky(Symmetric(_vcov_u(fit))).L * randn(rng, length(u))
    end
    θ = constrained_parameters(m, u)
    mω, sω = (_has_twist(m) ? (θ.m_ω, θ.s_ω) : (0.0, 0.0))
    sim = simulate_stems(
        θ.η; plants = length(m.data), organs = length.(m.data), x = m.torsion_x, θ.logμκ, θ.σκ, θ.α,
        outlier = θ.outlier, torsion_mean = mω, torsion_sd = sω, rng
    )
    return (; sim..., θ)
end

"""
    divergence_angles(y; o = 1, degrees = true)

The divergence angles of a stem from its absolute angles y (radians, bottom to top): the angle from each organ to the
next, `mod(o (y[i + 1] − y[i]), 2π)`, n − 1 of them. o = +1 measures counter-clockwise; for a stem whose spiral turns the
other way, o = −1 makes the canonical divergence read 137.5° as well (`simulate_stems` returns the orientation of each
stem, `o`, and `decode_orders(fit; full = true)` that of each data stem). In degrees by default.
"""
function divergence_angles(y::AbstractVector{<:Real}; o::Integer = 1, degrees::Bool = true)
    d = [mod(o * (y[i + 1] - y[i]), 2π) for i in 1:(length(y) - 1)]
    return degrees ? rad2deg.(d) : d
end
