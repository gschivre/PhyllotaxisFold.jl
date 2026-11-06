# Fitting
#
# The MAP fit of the propensity model and its result: `fit_map` (the fitting schedule, with or without a twist),
# `newton_map` (Newton trust region on a value / gradient / Hessian function from `AngleLikelihood.jl`), the placement
# of the twist quadrature nodes between the fits (`Torsion.jl`), the result `PhyllotaxisFit`, and its accessors,
# diagnostics and display. The objective and its derivatives are in `AngleLikelihood.jl`; the model in
# `PropensityModel.jl`.

#################################################################################################################
# the MAP: Newton and the fitting schedule
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
last Hessian evaluation, re-evaluated if it was not at the maximiser). Throws an error if f or its derivatives are
not finite at the point reached (at `v0`, with no finite point to move to).
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
        # (a non-finite point is given a zero gradient above, which the optimiser would report as converged)
        (isfinite(l) && all(isfinite, g) && all(isfinite, h)) ||
            error("newton_map: the objective or its derivatives are not finite at the point reached, v = $(v)")
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
# fit_map
#################################################################################################################

"""
    fit_map(model; u0, g_tol = 1e-5, iterations = 100, recenter = 1, nodes_kw = (;), options)

MAP of the propensity model by Newton trust region in the unconstrained space, with the analytic gradient and the
Hessian from one dual pass over it. Dispatches on the model: with or without outlier angles, with or without a twist
(`PhyllotaxisModelAD(...; torsion)`). Returns a `PhyllotaxisFit`.

Without twist: one Newton fit from `u0` (the prior mean by default). With a twist, each stem's rate is integrated out
by adaptive quadrature (nodes on the modes of the stem's own likelihood): 1. fit without twist; 2. nodes at that fit,
rates set to each stem's best mode; 3. fit with these rates fixed; 4. nodes at that fit; 5. Newton on the integrated
objective in `[u; m_ω; log s_ω]`; then `recenter` times: nodes at the new fit and refit. `nodes_kw` are passed to the
node placement (e.g. `(; nnodes = 9)`, nodes per mode; `scan`). `u0` has the model parameters only, or the twist
hyperparameters too (a previous fit's `u`), which then start step 5; by default step 5 starts from the spread of the
best modes. The nodes of the fit are stored in the model (its log density is then the objective of the fit).

Convergence: each Newton fit stops when no component of the gradient of the log posterior (in the unconstrained
space) exceeds `g_tol`, or after `iterations` steps (the fit is then reported as not converged). The default `g_tol`
leaves the parameters off the maximum by about `g_tol` × their posterior variance, far below their uncertainty on
usual data; lower it for data that determine the parameters very precisely. `options` (an `Optim.Options`) replaces
both, for full control of the optimiser.
"""
function fit_map(
        model::PhyllotaxisModelAD; u0 = nothing, g_tol::Real = MAP_G_TOL, iterations::Integer = MAP_ITERATIONS,
        options = map_options(; g_tol, iterations), recenter::Int = 1, nodes_kw = (;)
    )
    G, D = nmodel(model), LogDensityProblems.dimension(model)
    u = collect(Float64, (u0 === nothing ? prior_mean_parameters(model)[1:G] : u0))
    length(u) ∈ (G, D) || throw(DimensionMismatch("u0: $(G) model parameters, or $(D) with the twist hyperparameters"))
    return _fit(model, u, options, recenter, nodes_kw)
end
fit_map(prob::TransformedLogDensityProblem; kw...) = fit_map(prob.model; kw...)

_fit(m::PhyllotaxisModelAD{T, N, O, Nothing}, u0, options, recenter, nodes_kw) where {T, N, O} = _fit_no_twist(m, u0, options)
_fit(m::PhyllotaxisModelAD{T, N, O, TorsionPrior}, u0, options, recenter, nodes_kw) where {T, N, O} =
    _fit_twist(m, u0, options, recenter, nodes_kw)

#################################################################################################################
# the fit: accessors, diagnostics, show
#################################################################################################################

const _AnyFit = PhyllotaxisFit{<:PhyllotaxisModelAD}

function _parnames(m::PhyllotaxisModelAD)
    return vcat(
        ["η_$d" for d in 1:(m.k)], ["log μκ", "σκ", "α"], (_has_outlier(m) ? ["π_o"] : String[]),
        (_has_twist(m) ? ["m_ω", "s_ω"] : String[])
    )
end

"the MAP as the flat constrained vector θ (the order of `coefnames`)"
coef(fit::_AnyFit) = _constrained(fit.model, fit.u)
coefnames(fit::_AnyFit) = _parnames(fit.model)
loglikelihood(fit::_AnyFit) = fit.loglik
dof(fit::_AnyFit) = length(fit.u)
"the number of organs"
nobs(fit::_AnyFit) = sum(length, fit.model.data)

_negH_posdef(fit) = isposdef(Symmetric(-fit.H))
"Laplace covariance of the unconstrained parameters, −H⁻¹ (NaN if −H is not positive definite)"
_vcov_u(fit) = (_negH_posdef(fit) ? Matrix(inv(Symmetric(-fit.H))) : fill(NaN, size(fit.H)))

"""
    vcov(fit)

Laplace covariance of the constrained parameters θ (`coef`): `J (−H)⁻¹ Jᵀ`, J the Jacobian of θ in u at the MAP (delta
method). NaN if the Hessian is not negative definite (the fit is not at a maximum).
"""
function vcov(fit::_AnyFit)
    J = ForwardDiff.jacobian(u -> _constrained(fit.model, u), fit.u)
    return J * _vcov_u(fit) * J'
end
stderror(fit::_AnyFit) = sqrt.(max.(diag(vcov(fit)), 0.0))

# Laplace intervals: quantiles of θ(u) for u ~ N(û, −H⁻¹) (exact under the bijection, unlike θ̂ ± z sd)
function _laplace_intervals(fit, level; ndraws = 4000)
    V = _vcov_u(fit)
    all(isfinite, V) || return fill(NaN, length(fit.u), 2)
    L = cholesky(Symmetric(V)).L
    rng = Xoshiro(0)
    draws = reduce(hcat, [_constrained(fit.model, fit.u .+ L * randn(rng, length(fit.u))) for _ in 1:ndraws])
    a = (1 - level) / 2
    return reduce(vcat, [quantile(r, [a, 1 - a])' for r in eachrow(draws)])
end

"""
    coeftable(fit; level = 0.95)

The parameters at the MAP: estimate, Laplace standard deviation (`stderror`), Laplace interval at `level` (the normal
approximation in the unconstrained space, mapped back), and the ratio of the posterior sd to the prior sd: near 1, the
data say little about the parameter beyond its prior.
"""
function coeftable(fit::_AnyFit; level::Real = 0.95)
    ci = _laplace_intervals(fit, level)
    sd = stderror(fit)
    p = round(Int, 100 * level)
    return CoefTable(
        [coef(fit), sd, ci[:, 1], ci[:, 2], sd ./ _prior_sds(fit.model)],
        ["MAP", "Std. dev.", "Lower $(p)%", "Upper $(p)%", "sd / prior sd"], coefnames(fit)
    )
end

"""
    fit_diagnostics(fit; prior_ratio = 0.9, mode_mass = 0.9)

Checks of a `fit_map` result, as a NamedTuple:

- `converged`, `iterations`, `gradnorm` (largest |∂ log posterior / ∂u| at the MAP);
- `hessian_posdef` (−H positive definite: the MAP is a strict maximum), `hessian_cond` (its condition number: large
  values flag a direction the data barely determine);
- `logposterior`, `loglik`, `log_evidence` (Laplace approximation of the log marginal likelihood,
  `logposterior + D/2 log 2π − ½ log det(−H)`, to compare models fitted to the same stems, e.g. with and without
  outliers or another band k);
- `prior_dominated`: the parameters whose posterior sd exceeds `prior_ratio` × their prior sd (what the data do not
  inform, so what the prior sets);
- `low_stems`: stems whose log-likelihood per organ lies more than 3 robust sd (MAD) below the median (stems the model
  describes poorly: a measurement problem, another phyllotaxis);
- with a twist, `twist_multimodal`: the stems whose heaviest twist mode has a posterior mass below `mode_mass` (aliases
  of the twist, integrated over by the fit but worth a look), and `twist_at_edge`: stems with a mode at the edge of the
  rate scan (widen `nodes_kw = (; scan)`).
"""
function fit_diagnostics(fit::_AnyFit; prior_ratio::Real = 0.9, mode_mass::Real = 0.9)
    m = fit.model
    pd = _negH_posdef(fit)
    ev = eigvals(Symmetric(-fit.H))
    D = length(fit.u)
    log_evidence = pd ? fit.logposterior + D / 2 * log(2π) - sum(log, ev) / 2 : NaN
    ratio = stderror(fit) ./ _prior_sds(m)
    ll = _stem_loglik_per_organ(fit)
    med = median(ll)
    mad = 1.4826 * median(abs.(ll .- med))
    out = (;
        fit.converged, fit.iterations, gradnorm = maximum(abs, fit.gradient), hessian_posdef = pd,
        hessian_cond = (pd ? maximum(ev) / minimum(ev) : Inf), fit.logposterior, fit.loglik, log_evidence,
        prior_dominated = coefnames(fit)[ratio .> prior_ratio], low_stems = findall(ll .< med - 3 * mad),
    )
    fit.twist === nothing && return out
    edge = 25.0 - 1.0 # the default scan reaches ±25 degrees per mean step
    at_edge = findall(s -> any(md -> abs(md[1]) * _step(m, s) > edge, fit.twist.nodes.modes[s]), eachindex(m.data))
    return (; out..., twist_multimodal = findall(fit.twist.mode_mass .< mode_mass), twist_at_edge = at_edge)
end
_step(m, s) = (x = m.torsion_x[s]; x[end] / (length(x) - 1))
_stem_loglik_per_organ(fit) = stem_loglik(fit) ./ length.(fit.model.data)

function Base.show(io::IO, fit::_AnyFit)
    m = fit.model
    print(io, "PhyllotaxisFit (MAP): ")
    return show(io, m)
end
function Base.show(io::IO, ::MIME"text/plain", fit::_AnyFit)
    m = fit.model
    d = fit_diagnostics(fit)
    println(io, "PhyllotaxisFit (MAP) of the ", m)
    @printf(io, "  %s after %d iterations, max |gradient| %.1e\n", d.converged ? "converged" : "NOT CONVERGED", d.iterations, d.gradnorm)
    if d.hessian_posdef
        @printf(io, "  -Hessian positive definite, condition number %.1e\n", d.hessian_cond)
    else
        println(io, "  -Hessian NOT positive definite: not a strict maximum, no Laplace uncertainty")
    end
    @printf(io, "  log posterior %.2f, log-likelihood %.2f, Laplace log evidence %.2f (%d parameters)\n", d.logposterior, d.loglik, d.log_evidence, dof(fit))
    show(io, MIME"text/plain"(), coeftable(fit))
    println(io)
    isempty(d.prior_dominated) || println(io, "  set mostly by the prior (sd / prior sd > 0.9): ", join(d.prior_dominated, ", "))
    isempty(d.low_stems) || println(io, "  stems with a low log-likelihood per organ: ", d.low_stems)
    if fit.twist !== nothing
        isempty(d.twist_multimodal) || println(io, "  stems with a multimodal twist (heaviest mode < 90 %): ", d.twist_multimodal)
        isempty(d.twist_at_edge) || println(io, "  stems with a twist mode at the edge of the scan: ", d.twist_at_edge)
    end
    return nothing
end
