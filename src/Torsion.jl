"""
    Torsion

Stem torsion, shared by the models. The organ at stem position p is rotated by ω_s x_p in the absolute frame, with one
rate ω_s per stem (degrees per unit of x) and x_p the torsion coordinate of the stem:

- the stem length below p in mm, `torsion_coordinate(ℓ)` from the n − 1 internodes (preferred: the twist accumulates
  along the length; Landrein et al. 2013: an internode of length I on a stem of radius r twisted by α rotates the organs
  above it by I tan α / r, so ω_s = tan α / r for a constant α / r);
- p − 1, `torsion_coordinate(n)`, when the length is not measured (degrees per stem position).

The rates are random effects, like κ: ω_s ~ N(m_ω, s_ω²), with (m_ω, s_ω) fitted (hyperprior `TorsionPrior`) and each
ω_s integrated out of its stem's likelihood. The per-stem likelihood is sharp (≈ 0.1°/mm) and multimodal in ω (a twist
per position is confounded with the divergence angle; alternative spirals with a few permutations give secondary modes),
so the integral uses *adaptive* quadrature: Gauss–Hermite nodes placed on each mode of the stem's own likelihood, found
by a scan at preliminary parameters (`stem_nodes`), and kept fixed during the fit (an importance-sampling rule with a
Gaussian-mixture proposal, `TorsionNodes`). The fitted objective is then a smooth function of the global parameters and
(m_ω, log s_ω), maximised by Newton (`newton_map`; the derivatives are in `AngleLikelihood`).
"""
module Torsion

using Statistics
using Distributions: Normal, logpdf
using LogExpFunctions, FastGaussQuadrature
using Optim, NLSolversBase
using ..Common: _halft

# default options of the Newton MAP fits
const MAP_OPTIONS = Optim.Options(g_tol = 1.0e-5, iterations = 100)

export StemTorsion, TorsionPrior, TorsionNodes, torsion_coordinate, torsion_corrected, torsion_step, torsion_hyperprior,
    scan_rate, stem_nodes, node_logterms, node_posterior, newton_map, corrected_angles, update_torsion!,
    torsion_nodes, fit_map_torsion!, golden_section_max

"""
    corrected_angles(model, s; τ)

Angles of plant s with its twist removed (rate τ, the fitted one by default). Methods for
`PhyllotaxisModelAD`.
"""
function corrected_angles end

"""
    update_torsion!(model, θ_or_u; scan = -25:0.5:25)

Set each plant's twist rate to the value maximising its own log-likelihood at the global parameters (scan in degrees per
mean step of the coordinate, then golden section). Methods: `PhyllotaxisModelAD`
(unconstrained u).
"""
function update_torsion! end

"""
    torsion_nodes(model, θ_or_u; kwargs...)

Quadrature nodes of every stem's twist rate (`stem_nodes` applied to each stem's log-likelihood at the given global
parameters). Methods: `PhyllotaxisModelAD` (θ or u).
"""
function torsion_nodes end

"""
    fit_map_torsion!(model; prior = TorsionPrior(), u0, recenter = 1, nodes_kw = (;), options)

MAP of the global parameters and of the twist population (m_ω, log s_ω), with each stem's rate integrated out:
1. fit without torsion; 2. nodes at that fit, rates set to each stem's best mode; 3. fit with these rates fixed;
4. nodes at that fit; 5. Newton on the integrated objective; then `recenter` times: nodes at the new fit and refit.
Returns (; fit, m_ω, s_ω, τ, τ_mean, τ_sd, loglik, nodes) (loglik: Σ of the integrated stem log-likelihoods, priors
excluded): fit = (; θ, u, H, result) with u = [global u; m_ω; log s_ω] and H the
Hessian of the log posterior in u (−inv(H): Laplace covariance); per stem, τ the most probable mode (used by
`corrected_angles`), τ_mean and τ_sd the posterior mean and sd of the rate (aliases included). Methods:
`PhyllotaxisModelAD` (or its `TransformedLogDensityProblem`).

The model's stored twist rates (`model.torsion.τ`, the default rate of `stem_loglik`, `corrected_angles` and of the
decoding functions given plain parameters u) are overwritten: by each stem's most probable mode at the end.
"""
function fit_map_torsion! end

"torsion coordinates (one vector per stem) and rates (degrees per unit of the coordinate, one per stem, start at 0)"
struct StemTorsion
    x::Vector{Vector{Float64}}
    τ::Vector{Float64}
end
StemTorsion(x::AbstractVector{<:AbstractVector}) = StemTorsion([collect(Float64, xs) for xs in x], zeros(length(x)))

"""
    TorsionPrior(mt = 0.0, st = 2.0, νt = 4.0, τt = 1.0)

Hyperprior of the twist population ω_s ~ N(m_ω, s_ω²), in degrees per unit of the coordinate (°/mm with internodes):
m_ω ~ N(mt, st²), s_ω ~ half-t(νt, τt). An informative prior on the mean twist is a small st.
"""
Base.@kwdef struct TorsionPrior
    mt::Float64 = 0.0
    st::Float64 = 2.0
    νt::Float64 = 4.0
    τt::Float64 = 1.0
end

"log hyperprior of (m_ω, log s_ω), Jacobian of the log included"
torsion_hyperprior(p::TorsionPrior, m, logs) = logpdf(Normal(p.mt, p.st), m) + logpdf(_halft(p.νt, p.τt), exp(logs)) + logs

"x_p = p − 1 (degrees per stem position) for a stem of n organs, or the stem length below p from its n − 1 internodes"
torsion_coordinate(n::Integer) = collect(0.0:(n - 1))
torsion_coordinate(ℓ::AbstractVector{<:Real}) = vcat(0.0, cumsum(Float64.(ℓ)))

"mean step of a coordinate (1 per position, or the mean internode): the scale of a rate"
torsion_step(x::AbstractVector) = x[end] / (length(x) - 1)

"angles with the twist of rate τ (degrees per unit of x) removed (generic in τ: a dual rate gives dual angles)"
torsion_corrected(y::AbstractVector, x::AbstractVector, τ::Real) = ((τ isa AbstractFloat && iszero(τ)) ? y : y .- deg2rad(τ) .* x)

"""
    scan_rate(f, step; scan = -25:0.5:25, tol = 0.01)

Rate (per unit of x) maximising f(rate): a scan over `scan` followed by golden-section refinement to `tol`; `scan` and
`tol` are in degrees per mean step `step` of the coordinate, so the same grid suits positions and millimetres.
"""
function scan_rate(f, step::Real; scan = -25.0:0.5:25.0, tol::Real = 0.01)
    g(t) = f(t / step)
    v = [g(t) for t in scan]
    i = argmax(v)
    t, _ = golden_section_max(g, scan[max(i - 1, 1)], scan[min(i + 1, length(scan))]; tol)
    return t / step
end

"""
    golden_section_max(f, a, b; tol = 1e-3, score = identity)

Maximum of a unimodal f on [a, b] by golden-section search, to an interval narrower than `tol`. `score(f(x))` is the
number compared (for an f that returns a richer object). Returns (x, f(x)) of the best point evaluated.
"""
function golden_section_max(f, a::Real, b::Real; tol::Real = 1.0e-3, score = identity)
    c = (sqrt(5) - 1) / 2
    x1, x2 = b - c * (b - a), a + c * (b - a)
    f1, f2 = f(x1), f(x2)
    while b - a > tol
        if score(f1) > score(f2)
            b, x2, f2 = x2, x1, f1
            x1 = b - c * (b - a)
            f1 = f(x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + c * (b - a)
            f2 = f(x2)
        end
    end
    return score(f1) > score(f2) ? (x1, f1) : (x2, f2)
end


"""
    TorsionNodes(ω, logw, modes)

Fixed quadrature rule for each stem's rate: ∫ L_s(ω) p(ω) dω ≈ Σ_j exp(logw[s][j]) L_s(ω[s][j]) p(ω[s][j]), with
logw = log(Gauss–Hermite weight × mixture weight) − log q_s(ω) for the Gaussian-mixture proposal q_s placed on the modes
of L_s (`modes[s]`: (centre, sd, log-likelihood) of each). Exact in expectation for any p (multiple importance sampling,
balance heuristic); accurate when L_s is close to Gaussian around each mode and p varies slowly on the scale of the sd.
"""
struct TorsionNodes
    ω::Vector{Vector{Float64}}
    logw::Vector{Vector{Float64}}
    modes::Vector{Vector{NTuple{3, Float64}}}
end

"""
    stem_nodes(f, step; scan = -25:0.5:25, Δ = 8.0, maxmodes = 4, nnodes = 9, inflate = 1.25)

Nodes of one stem from its log-likelihood f(ω): scan (degrees per mean step `step`), local maxima within Δ log-units of
the best (at most `maxmodes`), each refined by golden section, with sd from the curvature (× `inflate`); `nnodes`
Gauss–Hermite nodes per mode. Returns (ω, logw, modes) for `TorsionNodes`.
"""
function stem_nodes(f, step::Real; scan = -25.0:0.5:25.0, Δ::Real = 8.0, maxmodes::Int = 4, nnodes::Int = 9, inflate::Real = 1.25)
    g(t) = f(t / step)
    v = [g(t) for t in scan]
    cand = [i for i in eachindex(v) if (i == 1 || v[i] ≥ v[i - 1]) && (i == length(v) || v[i] ≥ v[i + 1])]
    sort!(cand; by = i -> -v[i])
    modes = NTuple{3, Float64}[]
    for i in cand[1:min(end, 2 * maxmodes)]
        a, b = scan[max(i - 1, 1)], scan[min(i + 1, length(scan))]
        t = scan_rate(g, 1.0; scan = range(a, b; length = 5), tol = 0.002) # golden section on [a, b]
        gt = g(t)
        any(m -> abs(m[1] - t) < 0.25, modes) && continue
        δ = 0.05
        c2 = (g(t + δ) - 2gt + g(t - δ)) / δ^2
        sd = inflate * min(1 / sqrt(max(-c2, 1.0e-3)), 2.0)
        push!(modes, (t, sd, gt))
    end
    best = maximum(m -> m[3], modes)
    filter!(m -> m[3] ≥ best - Δ, modes)
    sort!(modes; by = m -> -m[3])
    modes = modes[1:min(end, maxmodes)]
    # mixture weights ∝ Laplace mass of each mode; nodes in rate units (per unit of x)
    lπ = [m[3] - best + log(m[2]) for m in modes]
    lπ .-= logsumexp(lπ)
    z, w = gausshermite(nnodes; normalize = true)
    comps = [(m[1] / step, m[2] / step) for m in modes]
    logq(ω) = logsumexp([lπ[r] + logpdf(Normal(c, h), ω) for (r, (c, h)) in enumerate(comps)])
    ω, logw = Float64[], Float64[]
    for (r, (c, h)) in enumerate(comps), j in eachindex(z)
        x = c + h * z[j]
        push!(ω, x)
        push!(logw, lπ[r] + log(w[j]) - logq(x))
    end
    return ω, logw, [(c, h, m[3]) for ((c, h), m) in zip(comps, modes)]
end

"log terms a_j = logw_j + ℓ_j + log N(ω_j | m, s) of one stem's integral; the stem's integrated log-likelihood is logsumexp(a)"
node_logterms(ω, logw, ℓ, m, s) = [logw[j] + ℓ[j] + logpdf(Normal(m, s), ω[j]) for j in eachindex(ω)]

"posterior of one stem's rate from its node terms a (Float64): (mean, sd, most probable mode centre)"
function node_posterior(ω, a, modes)
    pw = exp.(a .- logsumexp(a))
    μ = sum(pw .* ω)
    sd = sqrt(max(sum(pw .* (ω .- μ) .^ 2), 0.0))
    nn = length(ω) ÷ length(modes)
    mass = [sum(pw[((r - 1) * nn + 1):(r * nn)]) for r in eachindex(modes)]
    return μ, sd, modes[argmax(mass)][1]
end

"""
    newton_map(vgh, v0; options)

Maximise f by Newton trust region, `vgh(v, order)` returning (f, ∇f or nothing, ∇²f or nothing) up to `order`. A
non-finite evaluation is a rejected step. Returns (; v, H, result) with H the Hessian at the last Newton step.
"""
function newton_map(vgh, v0::AbstractVector; options = MAP_OPTIONS)
    H_last = zeros(length(v0), length(v0))
    fgh! = (F, G, H, v) -> begin
        order = (H !== nothing ? 2 : G !== nothing ? 1 : 0)
        l, g, h = vgh(v, order)
        ok = isfinite(l) && (order < 1 || all(isfinite, g)) && (order < 2 || all(isfinite, h))
        if order ≥ 2
            H .= (ok ? .-(h .+ h') ./ 2 : 0.0)
            ok && (H_last .= (h .+ h') ./ 2)
        end
        (G !== nothing) && (G .= (ok ? .-g : 0.0))
        return ok ? -l : Inf
    end
    res = optimize(NLSolversBase.only_fgh!(fgh!), collect(Float64, v0), NewtonTrustRegion(), options)
    return (; v = Optim.minimizer(res), H = H_last, result = res)
end

end # module Torsion
