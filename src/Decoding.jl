# Decoding
#
# What a fit (or parameters given as a vector) says about each stem, the twist rate integrated out over the quadrature
# nodes of the fit: the log-likelihood of each stem (`stem_loglik`, `angle_loglikelihood`), the angles with the twist
# removed (`corrected_angles`), and the organ orders (`decode_orders`: joint MAP; `order_posterior`: exact posterior
# marginals and samples; `spiral_posterior`).

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
