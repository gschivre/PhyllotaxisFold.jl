# Breaks
#
# `spiral_breaks(fit)`: breaks of the spiral along each stem (manual: "Breaks of the spiral"). Each part between two
# breaks is scored as a stem of its own: one forward pass from organ a gives every part [a, b] (`prefix_logperms!`), and
# an exact change-point recursion sums over the segmentations, with a binomial prior per gap.

"""
    SpiralBreaks

The result of `spiral_breaks`: `stems`, one NamedTuple per stem, and the break rate (`break_rate`: its posterior,
summarised, or the fixed value). Per stem:

- `logZ`: log Z(j), j = 0..n − 1, the log-likelihood summed over every placement of j breaks (`logZ[1]` is the stem's
  log-likelihood without breaks, `stem_loglik`);
- `logBF`: log Bayes factor of "at least one break" against "none", at the break rate (integrated over its posterior);
- `p_breaks`: the posterior probability of each number of breaks, 0..n − 1;
- `p_cut`: P(a break just before organ i), i = 1..n (`p_cut[1] = 0`);
- `p_part`: P(organ i lies in a part of each value), n × the number of values of what breaks (orientation −1/+1, or
  the canonical angles, or both; nothing for `:precision`), with `part_values` naming the columns.
"""
struct SpiralBreaks
    kind::Tuple{Vararg{Symbol}}
    stems::Vector{Any}
    break_rate::Any
    part_values::Vector{Any}
    min_length::Int
end

const _BREAK_KINDS = (:phase, :orientation, :angle, :precision)

_break_kind(k::Symbol) = _break_kind((k,))
function _break_kind(k::Tuple)
    isempty(k) && throw(ArgumentError("kind: at least one of $(_BREAK_KINDS)"))
    for x in k
        x === :twist && throw(ArgumentError("kind = :twist is not available yet (planned)"))
        x in _BREAK_KINDS || throw(ArgumentError("kind: $(x) is not one of $(_BREAK_KINDS) (or :each)"))
    end
    return Tuple(x for x in _BREAK_KINDS if x in k)
end

"""
    spiral_breaks(fit; kind = :orientation, break_rate = Beta(1, 99), min_length = 1)

Breaks of the spiral along each stem of a fit (`SpiralBreaks`): the posterior number of breaks of each stem, where they
are, the log Bayes factor of a break, and the posterior of the break rate across the stems.

- `kind`: what changes at a break, besides the phase (each part has its own, uniform): `:orientation` (the default),
  `:angle` (another of the prior's `angles`), `:precision` (a new κ from the fitted law), `:phase` (nothing else), a
  tuple of them (changing together), or `:each` (every single kind, a NamedTuple of results).
- `break_rate`: the prior probability π_b that a gap between two organs is a break: a number, or a `Beta` prior shared
  by all the stems (the default, mean 1 %), integrated out.
- `min_length`: the fewest organs in a part.

The population parameters stay at the fit, the parts of a stem share its twist rate, and the order law restarts at
each break. Cost: about n/2 stem likelihoods per stem of n organs.
"""
function spiral_breaks(fit::PhyllotaxisFit; kind = :orientation, break_rate = Beta(1.0, 99.0), min_length::Int = 1)
    if kind === :each
        kinds = (:phase, :orientation, (length(fit.model.grid.angles) > 1 ? (:angle,) : ())..., :precision)
        return NamedTuple{kinds}(Tuple(spiral_breaks(fit; kind = k, break_rate, min_length) for k in kinds))
    end
    kd = _break_kind(kind)
    m = fit.model
    (:angle in kd && length(m.grid.angles) < 2) && throw(ArgumentError("kind = :angle needs at least two canonical angles"))
    min_length ≥ 1 || throw(ArgumentError("min_length ≥ 1"))
    (break_rate isa Real && !(0 < break_rate < 1)) && throw(ArgumentError("break_rate: a probability in (0, 1)"))
    nodes = _nodes(fit)
    u = fit.u
    p = stem_params(m, map(Float64, u))
    # pass 1: the part tables of every stem give log Z(j); the break rate's posterior follows
    Z = _breaks_pass(m, p, u, nodes, kd, min_length, nothing)
    rate, πnodes, πw = _break_rate_posterior(break_rate, Z, length.(m.data))
    # pass 2: the posteriors of the cuts and of the parts, at the rate's nodes
    post = _breaks_pass(m, p, u, nodes, kd, min_length, (πnodes, πw))
    stems = map(eachindex(Z)) do s
        n = length(m.data[s])
        lz = Z[s]
        pj = _breaks_posterior(lz, n, πnodes, πw)
        lb = _logBF_breaks(lz, n, πnodes, πw)
        (; logZ = lz, logBF = lb, p_breaks = pj, post[s]...)
    end
    return SpiralBreaks(kd, stems, rate, _part_values(m, kd), min_length)
end

# the values of a part (the columns of `p_part`)
function _part_values(m, kd)
    os = (:orientation in kd ? (-1, 1) : (nothing,))
    js = (:angle in kd ? Tuple(eachindex(m.grid.angles)) : (nothing,))
    return Any[(o = o, angle = (j === nothing ? nothing : m.grid.angles[j])) for j in js for o in os if !(o === nothing && j === nothing)]
end

# log Σ_j π^j (1 − π)^(n − 1 − j) Z(j)
_binom_mix(lz, n, π) = logsumexp(j * log(π) + (n - 1 - j) * log1p(-π) + lz[j + 1] for j in 0:(n - 1))

# posterior of the number of breaks at each rate node, averaged over the rate's posterior
function _breaks_posterior(lz, n, πn, πw)
    out = zeros(n)
    for (π, w) in zip(πn, πw)
        lp = [j * log(π) + (n - 1 - j) * log1p(-π) + lz[j + 1] for j in 0:(n - 1)]
        out .+= w .* exp.(lp .- logsumexp(lp))
    end
    return out
end
# log BF of ≥ 1 break against none (the prior given ≥ 1 break: the binomial renormalised), averaged over the rate's
# posterior
function _logBF_breaks(lz, n, πn, πw)
    return logsumexp(log(w) + logsumexp(j * log(π) + (n - 1 - j) * log1p(-π) + lz[j + 1] for j in 1:(n - 1)) -
        log1mexp((n - 1) * log1p(-π)) - lz[1] for (π, w) in zip(πn, πw))
end

# the break rate: a fixed number, or the posterior of a Beta prior given every stem (on a fine logit grid), summarised
# by 15 nodes of equal posterior mass for the per-stem posteriors
function _break_rate_posterior(r::Real, Z, ns)
    return (; fixed = Float64(r)), [Float64(r)], [1.0]
end
function _break_rate_posterior(prior::Beta, Z, ns)
    xs = range(-16.0, 4.0; length = 2001)
    πs = logistic.(xs)
    lp = [logpdf(prior, π) + log(π) + log1p(-π) + sum(_binom_mix(Z[s], ns[s], π) for s in eachindex(Z)) for π in πs]
    w = exp.(lp .- logsumexp(lp))
    cw = cumsum(w)
    q(a) = πs[min(searchsortedfirst(cw, a), length(πs))]
    nodes = [q((i - 0.5) / 15) for i in 1:15]
    summary = (; prior, mean = sum(w .* πs), median = q(0.5), ci95 = (q(0.025), q(0.975)), grid = collect(πs), posterior = w)
    return summary, nodes, fill(1 / 15, 15)
end

# One task per (stem, twist node), longest stems first: the number of twist nodes varies a lot between stems. Pass 1
# (rates = nothing): log Z(j) of each stem; pass 2: the posteriors of the cuts and of the parts.
function _breaks_pass(m, p, u, nodes, kd, min_length, rates)
    S = length(m.data)
    rules = [_twist_rule(m, u, s, nodes) for s in 1:S]
    items = [(s, j) for s in 1:S for j in eachindex(rules[s][1])]
    sort!(items; by = it -> -length(m.data[it[1]]))
    out = Dict{Tuple{Int, Int}, Any}()
    lk = ReentrantLock()
    pool = _pool(m, Float64)
    @sync for (s, j) in items
        Threads.@spawn begin
            v = _with_workspace(pool, m, s) do _, fws
                _node_breaks(m, p, s, rules[s][1][j], rules[s][2][j], kd, min_length, fws, rates)
            end
            lock(() -> (out[(s, j)] = v), lk)
        end
    end
    return map(1:S) do s
        acc = reduce(vcat, [out[(s, j)] for j in eachindex(rules[s][1])])
        _combine_breaks(acc, length(m.data[s]), kd, length(m.grid.angles), rates)
    end
end

# the per-(twist node, outer value) results of one stem, combined
function _combine_breaks(acc, n, kd, L, rates)
    rates === nothing && return [logsumexp(a[j] for a in acc) for j in 1:n]
    # weights: by evidence within a rate node; by the rate's posterior across them (it already holds this stem)
    T = (:orientation in kd ? 2 : 1) * (:angle in kd ? L : 1)
    p_cut, p_part = zeros(n), zeros(n, T)
    for (r, wr) in enumerate(rates[2])
        le = [a.logev[r] for a in acc]
        wts = exp.(le .- logsumexp(le))
        for (i, a) in enumerate(acc)
            p_cut .+= wr * wts[i] .* a.p_cut[r]
            p_part .+= wr * wts[i] .* a.p_part[r]
        end
    end
    return (; p_cut, p_part = ((:orientation in kd || :angle in kd) ? p_part : zeros(n, 0)))
end

# stem s at the twist rate ω (log weight lpω): one result per outer value (pass 1: log Z(j); pass 2: evidence and
# posteriors at each rate node)
function _node_breaks(m, p, s::Int, ω, lpω, kd, min_length, fws::StemWorkspace, rates)
    grid = m.grid
    (; k, ngrid_ϕ, ngrid_κ, κ_nodes, κ_logweights, angles) = grid
    n = length(m.data[s])
    K = 2k + 1
    L = length(angles)
    # the value of a part (what breaks: a state of the recursion), and the outer values (what is shared)
    os_t, js_t = (:orientation in kd ? 2 : 1), (:angle in kd ? L : 1)
    T = os_t * js_t
    lps = _angle_logprobs(p.α)
    tidx(io, j) = (:orientation in kd ? io : 1) + os_t * ((:angle in kd ? j : 1) - 1)
    logp_t = zeros(T)
    for io in 1:os_t, j in 1:js_t
        logp_t[tidx(io, j)] = (:orientation in kd ? -log(2) : 0.0) + (:angle in kd ? lps[j] : 0.0)
    end
    logtrans = fill(-Inf, T, T)
    for a in 1:T, b in 1:T
        T == 1 && (logtrans[a, b] = 0.0; continue)
        a != b && (logtrans[a, b] = logp_t[b] - log1mexp(logp_t[a]))
    end
    nouter_o = (:orientation in kd ? 1 : 2)
    nouter_j = (:angle in kd ? 1 : L)
    nouter_q = (:precision in kd ? 1 : ngrid_κ)
    Nout = nouter_o * nouter_j * nouter_q
    # log weights of the outer values, and the index of a combination (o index, angle j, κ node q)
    outer_of(io, j, q) = 1 + (:orientation in kd ? 0 : io - 1) + nouter_o * ((:angle in kd ? 0 : j - 1) + nouter_j * (:precision in kd ? 0 : q - 1))
    logw_out = zeros(Nout)
    for io in 1:nouter_o, j in 1:nouter_j, q in 1:nouter_q
        logw_out[outer_of(io, j, q)] = (:orientation in kd ? 0.0 : -log(2)) + (:angle in kd ? 0.0 : lps[j]) +
            (:precision in kd ? 0.0 : κ_logweights[q])
    end
    # the η normaliser of a part depends on its length only
    toeplitz_weights!(fws.perm, p.η, n)
    Nη = prefix_logperms!(zeros(n), fws.perm, n)
    κs = [exp(p.logμκ + p.σκ * z) for z in κ_nodes]
    ots = [_outlier_terms(p.outlier, κ) for κ in κs]
    cst = [κ - logbesseli(0, κ) - log(2π) for κ in κs] # per organ: the emission's constant
    seg = PermanentWorkspace(Float64, m.graph, n)
    pre = zeros(n)
    S = fill(-Inf, n, n, T, Nout) # log-likelihood of part [a, b] with value t, at the outer values
    y = _corrected!(fws, m, s, ω)
    _fill_cosines!(fws, grid, y, p.α)
    for q in 1:ngrid_κ, c in 1:(2 * L * ngrid_ϕ)
        cmb = _combo(grid, c, p.α)
        io = (cmb.o == -1 ? 1 : 2)
        t, oi = tidx(io, cmb.j), outer_of(io, cmb.j, q)
        # a part's phase: the stem's offset grid, shifted (ϕ' = ϕ + o (a − 1) ψ)
        lw_part = -log(ngrid_ϕ) + (:precision in kd ? κ_logweights[q] : 0.0)
        W = _fill_weights!(fws.perm.w, p.η, fws.C, c, κs[q], n, k, ots[q])
        for a in 1:n
            nn = n - a + 1
            @inbounds for col in 1:K, r in 1:nn
                seg.w[r, col] = W[a + r - 1, col]
            end
            prefix_logperms!(pre, seg, nn)
            @inbounds for b in (a + min_length - 1):n
                len = b - a + 1
                S[a, b, t, oi] = logaddexp(S[a, b, t, oi], lw_part + pre[len] + len * cst[q] - Nη[len])
            end
        end
    end
    return map(1:Nout) do oi
        Sv, base = view(S, :, :, :, oi), lpω + logw_out[oi]
        rates === nothing ? base .+ _dp_counts(Sv, logp_t, logtrans, n, T) : _dp_posteriors(Sv, logp_t, logtrans, n, T, rates, base)
    end
end

# log Z(j) for j = 0..n − 1: the change-point recursion over the parts, F[b, t, j] = log Σ over the segmentations of
# 1..b with j breaks whose last part, of value t, ends at b
function _dp_counts(S, logp_t, logtrans, n, T)
    F = fill(-Inf, n, T, n)
    @inbounds for b in 1:n, t in 1:T
        F[b, t, 1] = logp_t[t] + S[1, b, t]
        for a in 2:b
            Sab = S[a, b, t]
            Sab == -Inf && continue
            for j in 2:a, t0 in 1:T
                lt = logtrans[t0, t]
                lt == -Inf && continue
                F[b, t, j] = logaddexp(F[b, t, j], F[a - 1, t0, j - 1] + lt + Sab)
            end
        end
    end
    return [logsumexp(F[n, t, j] for t in 1:T) for j in 1:n]
end

# at each rate π of the nodes: the log evidence (with the binomial factors) and the posteriors of the cuts and of the
# parts, by forward–backward over the parts
function _dp_posteriors(S, logp_t, logtrans, n, T, rates, base)
    πn, πw = rates
    logev, p_cuts, p_parts = Float64[], Vector{Float64}[], Matrix{Float64}[]
    A, Bk = zeros(n, T), zeros(n, T)
    for (π, w) in zip(πn, πw)
        lπ, l1π = log(π), log1p(-π)
        @inbounds for b in 1:n, t in 1:T
            v = logp_t[t] + S[1, b, t] + (b - 1) * l1π
            for a in 2:b
                Sab = S[a, b, t]
                Sab == -Inf && continue
                v = logaddexp(v, _into(A, a - 1, t, logtrans, T) + lπ + Sab + (b - a) * l1π)
            end
            A[b, t] = v
        end
        @inbounds for a in n:-1:1, t in 1:T
            v = S[a, n, t] + (n - a) * l1π
            for b in a:(n - 1)
                Sab = S[a, b, t]
                Sab == -Inf && continue
                v = logaddexp(v, Sab + (b - a) * l1π + lπ + _outof(Bk, b + 1, t, logtrans, T))
            end
            Bk[a, t] = v
        end
        lZ = logsumexp(A[n, t] for t in 1:T)
        pc = zeros(n)
        for τ in 2:n
            pc[τ] = exp(logsumexp(_into(A, τ - 1, t, logtrans, T) + lπ + Bk[τ, t] for t in 1:T) - lZ)
        end
        # parts: P(part [a, b] of value t), spread over its organs by differences
        D = zeros(n + 1, T)
        @inbounds for a in 1:n, b in a:n, t in 1:T
            Sab = S[a, b, t]
            Sab == -Inf && continue
            left = (a == 1 ? logp_t[t] : _into(A, a - 1, t, logtrans, T) + lπ)
            right = (b == n ? 0.0 : lπ + _outof(Bk, b + 1, t, logtrans, T))
            q = exp(left + Sab + (b - a) * l1π + right - lZ)
            D[a, t] += q
            D[b + 1, t] -= q
        end
        pp = cumsum(D[1:n, :]; dims = 1)
        push!(logev, base + lZ)
        push!(p_cuts, pc)
        push!(p_parts, pp)
    end
    return (; logev, p_cut = p_cuts, p_part = p_parts)
end
# log Σ_{t0} A[b, t0] p(t | t0): the parts before, into a part of value t
@inline _into(A, b, t, logtrans, T) = logsumexp(A[b, t0] + logtrans[t0, t] for t0 in 1:T)
# log Σ_{t1} p(t1 | t) Bk[a, t1]: out of a part of value t, into the parts after
@inline _outof(Bk, a, t, logtrans, T) = logsumexp(logtrans[t, t1] + Bk[a, t1] for t1 in 1:T)

function Base.show(io::IO, ::MIME"text/plain", r::SpiralBreaks)
    nb = [sum((0:(length(x.p_breaks) - 1)) .* x.p_breaks) for x in r.stems]
    println(io, "SpiralBreaks: kind $(join(r.kind, " + ")), $(length(r.stems)) stems")
    br = r.break_rate
    if haskey(br, :fixed)
        println(io, "  break rate π_b = $(br.fixed) per gap (fixed)")
    else
        @printf(io, "  break rate π_b: posterior mean %.4f, 95%% interval (%.4f, %.4f) per gap\n", br.mean, br.ci95...)
    end
    @printf(io, "  stems with P(≥ 1 break) > 0.5: %d; expected breaks per stem: %.2f\n",
        count(x -> 1 - x.p_breaks[1] > 0.5, r.stems), mean(nb))
    return nothing
end
Base.show(io::IO, r::SpiralBreaks) = print(io, "SpiralBreaks($(join(r.kind, " + ")), $(length(r.stems)) stems)")
