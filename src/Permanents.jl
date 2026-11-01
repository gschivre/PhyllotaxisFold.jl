"""
    Permanents

Banded permanents by dynamic programming over a sliding window of used columns. A permutation σ with |σ[i] − i| ≤ k is a
path through a graph whose states are the C(2k, k) masks of used columns in the window {i − k, …, i + k − 1}; each edge
assigns row i to column i + d. On this graph:

- `forward!`: log of the summed weight of all paths = log perm(W) (log-space, or scaled linear space)
- `backward!`: backward messages and the edge marginals P(σ[i] = i + d)
- `sample_permutation!`: exact sample (backward filtering, forward sampling)
- `map_permutation`: most likely permutation (Viterbi)

Edge log-weights live in `ws.w[i, d + k + 1]` (in both modes): any position-dependent weight table can be used,
`toeplitz_weights!` fills the stationary case `η_d`.

Convention: σ[i] is the canonical index of the organ observed at position i, displacement d = σ[i] − i.
"""
module Permanents

using Random: Random, AbstractRNG
using LogExpFunctions: logaddexp, logsumexp
using ForwardDiff: ForwardDiff

# logaddexp that returns the other argument exactly when one is -Inf (keeps ForwardDiff partials free of NaN)
@inline _isneginf(x) = isinf(x) && x < 0 # (x == -Inf is false for a dual number with nonzero partials)
@inline _logaddexp(a, b) = (_isneginf(a) ? b : (_isneginf(b) ? a : logaddexp(a, b)))

# Float64 value of possibly nested dual numbers
_fullvalue(x) = (x isa ForwardDiff.Dual ? _fullvalue(ForwardDiff.value(x)) : x)

# nothing exported; the interface is public (Julia ≥ 1.11), used as `PhyllotaxisFold.Permanents.forward!` etc.
@static if VERSION >= v"1.11"
    eval(
        Meta.parse(
            "public PermutationGraph, PermanentWorkspace, toeplitz_weights!, forward!, forward_linear!, " *
                "backward!, sample_permutation!, map_permutation, logprob_permutation, cycle_lengths"
        )
    )
end


struct PermutationGraph
    k::Int
    M::Int # number of states (binomial(2k, k))
    start_s::Int
    end_s::Int
    edge_ptr::Vector{Int} # CSR: edges of state s are edge_ptr[s]:(edge_ptr[s + 1] − 1)
    edge_next::Vector{Int}
    edge_d::Vector{Int}
    edge_col::Vector{Int} # column of the weight tables, d + k + 1
    # the same edges by destination (CSR): the edges into state t are in_ptr[t]:(in_ptr[t + 1] − 1)
    in_ptr::Vector{Int}
    in_src::Vector{Int}
    in_col::Vector{Int}
end

"""
    PermutationGraph(k)

States: masks of 2k bits with exactly k set (bit b ↔ column i − k + b, set = already used). Start = end = (1 << k) − 1.
From a mask whose bit 0 is free, row i must take column i − k (d = −k); otherwise it takes any free bit b (d = b − k) or
the column entering the window (d = +k). The window then slides by one column (>> 1).
"""
function PermutationGraph(k::Int)
    states = Int[]
    for mask in 0:((1 << (2 * k)) - 1)
        count_ones(mask) == k && push!(states, mask)
    end
    M = length(states)
    mask_to_idx = Dict(mask => idx for (idx, mask) in enumerate(states))
    start_s = mask_to_idx[(1 << k) - 1]

    edge_ptr = zeros(Int, M + 1)
    edge_next = Int[]
    edge_d = Int[]
    for (s, m) in enumerate(states)
        edge_ptr[s] = length(edge_next) + 1
        if (m & 1) == 0
            push!(edge_next, mask_to_idx[(m | 1) >> 1])
            push!(edge_d, -k)
        else
            for b in 1:(2 * k - 1)
                if (m & (1 << b)) == 0
                    push!(edge_next, mask_to_idx[(m | (1 << b)) >> 1])
                    push!(edge_d, b - k)
                end
            end
            push!(edge_next, mask_to_idx[(m >> 1) | (1 << (2 * k - 1))])
            push!(edge_d, k)
        end
    end
    edge_ptr[M + 1] = length(edge_next) + 1
    edge_col = edge_d .+ (k + 1)
    ins = [Tuple{Int, Int}[] for _ in 1:M]
    for s in 1:M, e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
        push!(ins[edge_next[e]], (s, edge_col[e]))
    end
    in_ptr = cumsum(vcat(1, length.(ins)))
    in_src, in_col = [x[1] for v in ins for x in v], [x[2] for v in ins for x in v]
    return PermutationGraph(k, M, start_s, start_s, edge_ptr, edge_next, edge_d, edge_col, in_ptr, in_src, in_col)
end

"""
    PermanentWorkspace(T, k, n_max; logspace = false)

Graph and buffers for permanents of size ≤ `n_max` with element type T (Float64, or a ForwardDiff dual type: build one
workspace per number type, they are cheap). The caller always writes log-weights in `ws.w`.
- `logspace = false` (default): scaled linear space (HMM scaling trick), faster; each call falls back to
  log-space by itself when the scaled pass cannot be trusted (see `forward!`), so the result is the same in both modes.
- `logspace = true`: messages are always logs, accumulated with logaddexp.
"""
struct PermanentWorkspace{T}
    g::PermutationGraph
    n_max::Int
    logspace::Bool # always log-space, or scaled linear space with automatic fallback
    F::Matrix{T} # forward messages, M × (n_max + 1) (logs when the last forward! ran in log-space)
    B::Matrix{T} # backward messages
    w::Matrix{T} # edge log-weights, n_max × (2k + 1)
    ew::Matrix{T} # linear weights, each row divided by its largest entry (scaled mode)
    lrow::Vector{T} # log of the largest weight of each row (scaled mode)
    rho::Matrix{T} # edge marginals P(σ[i] = i + d), n_max × (2k + 1)
    inlog::Base.RefValue{Bool} # whether the last forward! ran in log-space (read by backward! and sampling)
    wstale::Base.RefValue{Bool} # the last pass ran on linear weights only (`forward_linear!`): w does not hold them
    nfallback::Base.RefValue{Int} # number of scaled calls redone in log-space (diagnostic)
end
PermanentWorkspace(::Type{T}, k::Int, n_max::Int; logspace::Bool = false) where {T} =
    PermanentWorkspace(T, PermutationGraph(k), n_max; logspace)
# with a graph built once and shared (it is never written)
function PermanentWorkspace(::Type{T}, g::PermutationGraph, n_max::Int; logspace::Bool = false) where {T}
    K = 2 * g.k + 1
    return PermanentWorkspace{T}(
        g, n_max, logspace, zeros(T, g.M, n_max + 1), zeros(T, g.M, n_max + 1), zeros(T, n_max, K),
        zeros(T, n_max, K), zeros(T, n_max), zeros(T, n_max, K), Ref(logspace), Ref(false), Ref(0)
    )
end
PermanentWorkspace(k::Int, n_max::Int; kwargs...) = PermanentWorkspace(Float64, k, n_max; kwargs...)

"""
    toeplitz_weights!(ws, η, n)

Stationary log-weights `w[i, d] = η_d` for every row; η has length 2k + 1, `η[d + k + 1] = η_d` (`η_0 = 0` by gauge).
"""
function toeplitz_weights!(ws::PermanentWorkspace, η::AbstractVector, n::Int)
    for d_idx in axes(ws.w, 2)
        for i in 1:n
            ws.w[i, d_idx] = η[d_idx]
        end
    end
    return ws.w
end

# Budget of the scaled pass. The mass dropped by underflow at one step is < floatmin ≈ e^-708 relative to that step's
# total; afterwards it can gain at most a factor 1 / c_j per step on the surviving paths (rescaled weights are ≤ 1),
# so its relative error on the result is < e^(-708 + D), D = Σ_j max(0, −log c_j). D ≤ 600 keeps it below e^-108.
# A single c_j below e^-345 ≈ 1e-150 would also overflow inv(c) or its dual partials (−c'/c²).
const SCALED_STEP_MIN = 1.0e-150
const SCALED_DEFICIT_MAX = 600.0

"""
    forward!(ws, n)

Forward pass on the log-weights stored in ws.w, returns the log-permanent.
- log-space: F[s, i] = log of the summed weight of all partial paths reaching state s at step i.
- scaled linear space: row i of the weights is exponentiated relative to its largest entry `m_i`, the messages are
  renormalised at each step (`Σ_s F[s, i] = 1`) and `log perm = Σ_i (m_i + log c_i)`. The row maxima absorb a row where
  every candidate is unlikely (an outlier angle at large κ); the call is redone in log-space when a step normaliser
  falls below `SCALED_STEP_MIN` or the accumulated deficit exceeds `SCALED_DEFICIT_MAX` (see above), so the result
  never depends on underflow. `ws.inlog[]` records which pass produced F.
"""
function forward!(ws::PermanentWorkspace{T}, n::Int) where {T}
    ws.wstale[] = false
    ws.inlog[] = ws.logspace
    ws.logspace && return _forward_log!(ws, n)
    ok, l = _forward_scaled!(ws, n)
    ok && return l
    ws.inlog[] = true
    ws.nfallback[] += 1
    return _forward_log!(ws, n)
end

"""
    forward_linear!(ws, n)

`forward!` on linear weights: `ws.ew[i, d + k + 1]` holds the weight itself (not its log) of each admissible edge
(entries outside the columns 1..n are ignored). For weights that cannot underflow, e.g. a mixture with a uniform
component, it saves the log and exp of each entry that `forward!` needs. The scaled pass runs on them directly; in
log-space, or when the scaled pass cannot be trusted, `ws.w` is filled with their logs and the log-space pass runs.
`backward!` and sampling then work as after `forward!`.
"""
function forward_linear!(ws::PermanentWorkspace{T}, n::Int) where {T}
    ws.wstale[] = true
    logZ = _scale_rows_linear!(ws, n)
    _isneginf(logZ) && (ws.inlog[] = ws.logspace; return logZ)
    if !ws.logspace
        ws.inlog[] = false
        ok, l = _scaled_dp!(ws, n, logZ)
        ok && return l
        ws.nfallback[] += 1
    end
    _logweights_from_scaled!(ws, n)
    ws.inlog[] = true
    return _forward_log!(ws, n)
end
# w from the row-scaled weights ew and the row maxima (after `forward_linear!`)
function _logweights_from_scaled!(ws::PermanentWorkspace, n::Int)
    (; w, ew, lrow) = ws
    k = ws.g.k
    @inbounds for i in 1:n, j in (max(-k, 1 - i) + k + 1):(min(k, n - i) + k + 1)
        w[i, j] = log(ew[i, j]) + lrow[i]
    end
    ws.wstale[] = false
    return ws
end

function _forward_log!(ws::PermanentWorkspace{T}, n::Int) where {T}
    (; g, F, w) = ws
    (; k, M, edge_ptr, edge_next, edge_d) = g
    @inbounds begin
        fill!(view(F, :, 1:(n + 1)), T(-Inf))
        F[g.start_s, 1] = zero(T)
        for i in 1:n
            for s in 1:M
                val = F[s, i]
                _isneginf(val) && continue
                for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
                    d = edge_d[e]
                    (1 ≤ i + d ≤ n) || continue
                    nxt = edge_next[e]
                    F[nxt, i + 1] = _logaddexp(F[nxt, i + 1], val + w[i, d + k + 1])
                end
            end
        end
        return F[g.end_s, n + 1]
    end
end

# scaled pass: (true, log-permanent), or (false, 0) when it cannot be trusted (the caller then runs the log-space pass;
# a tuple rather than `nothing`, which Julia 1.10 boxes). The weights of the columns outside 1..n are set to 0 in ew, so
# the inner loops (here and in the scaled backward pass) need no bounds test: an edge to a column below 1 never leaves a
# reachable state (those columns start used), and a path through a column beyond n carries weight 0.
function _forward_scaled!(ws::PermanentWorkspace{T}, n::Int) where {T}
    logZ = _scale_rows_log!(ws, n)
    _isneginf(logZ) && return (true, logZ) # a row has no admissible column: perm = 0 exactly
    return _scaled_dp!(ws, n, logZ)
end

# ew[i, :] = exp(w[i, :] − m_i) with m_i the largest log-weight of row i among its admissible columns, 0 elsewhere;
# returns Σ m_i (or −Inf for an empty row)
function _scale_rows_log!(ws::PermanentWorkspace{T}, n::Int) where {T}
    (; g, w, ew, lrow) = ws
    k = g.k
    K = 2 * k + 1
    logZ = zero(T)
    @inbounds for i in 1:n
        lo, hi = max(-k, 1 - i) + k + 1, min(k, n - i) + k + 1
        mi = T(-Inf)
        for j in lo:hi
            mi = max(mi, w[i, j])
        end
        _isneginf(mi) && return T(-Inf)
        for j in 1:(lo - 1)
            ew[i, j] = zero(T)
        end
        for j in lo:hi
            ew[i, j] = exp(w[i, j] - mi)
        end
        for j in (hi + 1):K
            ew[i, j] = zero(T)
        end
        lrow[i] = mi
        logZ += mi
    end
    return logZ
end

# the same from linear weights in ew: each row divided by its largest admissible entry
function _scale_rows_linear!(ws::PermanentWorkspace{T}, n::Int) where {T}
    (; g, ew, lrow) = ws
    k = g.k
    K = 2 * k + 1
    logZ = zero(T)
    @inbounds for i in 1:n
        lo, hi = max(-k, 1 - i) + k + 1, min(k, n - i) + k + 1
        mi = zero(T)
        for j in lo:hi
            mi = max(mi, ew[i, j])
        end
        mi > 0 || return T(-Inf)
        im = inv(mi)
        for j in 1:(lo - 1)
            ew[i, j] = zero(T)
        end
        for j in lo:hi
            ew[i, j] *= im
        end
        for j in (hi + 1):K
            ew[i, j] = zero(T)
        end
        lrow[i] = log(mi)
        logZ += lrow[i]
    end
    return logZ
end

# the scaled dynamic programme on the row-scaled weights ew, from logZ = Σ_i m_i: (ok, log-permanent)
function _scaled_dp!(ws::PermanentWorkspace{T}, n::Int, logZ) where {T}
    (; g, F, ew) = ws
    (; M, in_ptr, in_src, in_col) = g
    @inbounds begin
        fill!(view(F, :, 1), zero(T))
        F[g.start_s, 1] = one(T)
        deficit = 0.0
        for i in 1:n
            # pull: each state sums over its incoming edges (one store per state, no scattered writes)
            c = zero(T)
            for t in 1:M
                acc = zero(T)
                for e in in_ptr[t]:(in_ptr[t + 1] - 1)
                    acc += F[in_src[e], i] * ew[i, in_col[e]]
                end
                F[t, i + 1] = acc
                c += acc
            end
            (c > SCALED_STEP_MIN) || return (false, zero(T))
            lc = log(c)
            deficit -= min(0.0, _fullvalue_f(lc))
            (deficit ≤ SCALED_DEFICIT_MAX) || return (false, zero(T))
            logZ += lc
            ic = inv(c)
            for s in 1:M
                F[s, i + 1] *= ic
            end
        end
        # one state is reachable after the last step (all columns used): F_end = 1 up to rounding
        return (true, logZ + log(F[g.end_s, n + 1]))
    end
end

# plain Float64 value of a real or dual number, for the underflow bookkeeping
_fullvalue_f(x::Float64) = x
_fullvalue_f(x::Real) = Float64(_fullvalue(x))

"""
    backward!(ws, n)

Backward pass (requires `forward!` on the same weights, and follows the pass it used), fills ws.rho with the edge
marginals P(σ[i] = i + d) (probabilities in both modes, each row sums to one) and returns view(ws.rho, 1:n, :).
"""
function backward!(ws::PermanentWorkspace{T}, n::Int) where {T}
    if !ws.inlog[]
        _backward_scaled!(ws, n) && return view(ws.rho, 1:n, :)
        # a backward normaliser underflowed: redo both passes in log-space (from the linear weights if that is all
        # the forward pass had)
        ws.inlog[] = true
        ws.nfallback[] += 1
        ws.wstale[] && _logweights_from_scaled!(ws, n)
        _forward_log!(ws, n)
    end
    _backward_log!(ws, n)
    return view(ws.rho, 1:n, :)
end

function _backward_log!(ws::PermanentWorkspace{T}, n::Int) where {T}
    (; g, F, B, w, rho) = ws
    (; k, M, edge_ptr, edge_next, edge_d) = g
    K = 2 * k + 1
    @inbounds begin
        fill!(view(B, :, n + 1), T(-Inf))
        B[g.end_s, n + 1] = zero(T)
        for i in n:-1:1
            for d_idx in 1:K
                rho[i, d_idx] = T(-Inf)
            end
            for s in 1:M
                val_F = F[s, i]
                acc = T(-Inf)
                for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
                    d = edge_d[e]
                    (1 ≤ i + d ≤ n) || continue
                    val_B = B[edge_next[e], i + 1] + w[i, d + k + 1]
                    acc = _logaddexp(acc, val_B)
                    rho[i, d + k + 1] = _logaddexp(rho[i, d + k + 1], val_F + val_B)
                end
                B[s, i] = acc
            end
            # each position holds exactly one site, so row i of the edge marginals sums to one
            r = logsumexp(view(rho, i, :))
            for d_idx in 1:K
                rho[i, d_idx] = exp(rho[i, d_idx] - r)
            end
        end
    end
    return true
end

# scaled backward pass on ws.ew; returns false when a normaliser underflows
function _backward_scaled!(ws::PermanentWorkspace{T}, n::Int) where {T}
    (; g, F, B, ew, rho) = ws
    (; k, M, edge_ptr, edge_next, edge_col) = g
    K = 2 * k + 1
    @inbounds begin
        fill!(view(B, :, n + 1), zero(T))
        B[g.end_s, n + 1] = one(T)
        for i in n:-1:1
            for d_idx in 1:K
                rho[i, d_idx] = zero(T)
            end
            for s in 1:M
                val_F = F[s, i]
                acc = zero(T)
                for e in edge_ptr[s]:(edge_ptr[s + 1] - 1) # ew is 0 outside the columns 1..n (see the forward pass)
                    j = edge_col[e]
                    val_B = B[edge_next[e], i + 1] * ew[i, j]
                    acc += val_B
                    rho[i, j] += val_F * val_B
                end
                B[s, i] = acc
            end
            r = sum(view(rho, i, :))
            c = sum(view(B, :, i))
            (r > SCALED_STEP_MIN && c > SCALED_STEP_MIN) || return false
            for d_idx in 1:K
                rho[i, d_idx] /= r
            end
            for s in 1:M
                B[s, i] /= c
            end
        end
    end
    return true
end

"""
    sample_permutation!(rng, ws, n)

Exact sample of σ with `P(σ) ∝ ∏_i exp(w[i, σ[i] − i])` (log-weights in ws.w): forward and backward passes, then from the
start state choose each edge with probability ∝ w B(next state).
"""
function sample_permutation!(rng::AbstractRNG, ws::PermanentWorkspace, n::Int)
    (; g, B, w) = ws
    (; k, edge_ptr, edge_next, edge_d) = g
    forward!(ws, n)
    backward!(ws, n)
    inlog = ws.inlog[]
    σ = zeros(Int, n)
    s = g.start_s
    for i in 1:n
        # transition weights w B (relative to their maximum in log-space)
        lmax = -Inf
        if inlog
            for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
                d = edge_d[e]
                (1 ≤ i + d ≤ n) && (lmax = max(lmax, w[i, d + k + 1] + B[edge_next[e], i + 1]))
            end
        end
        wsum = 0.0
        for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
            d = edge_d[e]
            (1 ≤ i + d ≤ n) && (wsum += _transition_weight(ws, inlog, lmax, i, e, d))
        end
        r = rand(rng) * wsum
        cumw = 0.0
        for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
            d = edge_d[e]
            (1 ≤ i + d ≤ n) || continue
            cumw += _transition_weight(ws, inlog, lmax, i, e, d)
            if cumw ≥ r
                σ[i] = i + d
                s = edge_next[e]
                break
            end
        end
    end
    return σ
end

@inline function _transition_weight(ws::PermanentWorkspace, inlog::Bool, lmax, i::Int, e::Int, d::Int)
    (; g, B, w, ew) = ws
    nxt, j = g.edge_next[e], d + g.k + 1
    return inlog ? exp(w[i, j] + B[nxt, i + 1] - lmax) : ew[i, j] * B[nxt, i + 1]
end
sample_permutation!(ws::PermanentWorkspace, n::Int) = sample_permutation!(Random.default_rng(), ws, n)

"""
    map_permutation(g, L, n)

Most likely permutation for a matrix of log-weights L (n × (2k + 1)): max-product (Viterbi) on the graph, then
backtracking from the end state.
"""
function map_permutation(g::PermutationGraph, L::AbstractMatrix, n::Int)
    (; k, M, edge_ptr, edge_next, edge_d) = g
    V = fill(-Inf, M, n + 1)
    V[g.start_s, 1] = 0.0
    back_d = zeros(Int, M, n)
    back_s = zeros(Int, M, n)
    for i in 1:n, s in 1:M
        (V[s, i] == -Inf) && continue
        for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
            d = edge_d[e]
            (1 ≤ i + d ≤ n) || continue
            val = V[s, i] + L[i, d + k + 1]
            if val > V[edge_next[e], i + 1]
                V[edge_next[e], i + 1] = val
                back_d[edge_next[e], i] = d
                back_s[edge_next[e], i] = s
            end
        end
    end
    σ = zeros(Int, n)
    s = g.end_s
    for i in n:-1:1
        σ[i] = i + back_d[s, i]
        s = back_s[s, i]
    end
    return σ
end

"log P(σ | η) under the stationary (Toeplitz) permanent model"
function logprob_permutation(ws::PermanentWorkspace, η::AbstractVector, σ::AbstractVector{Int})
    n = length(σ)
    k = ws.g.k
    toeplitz_weights!(ws, η, n)
    lp = -forward!(ws, n)
    for p in 1:n
        d = σ[p] - p
        (abs(d) ≤ k) || return -Inf
        lp += η[d + k + 1]
    end
    return lp
end

"cycle lengths of a permutation, fixed points included (length 1)"
function cycle_lengths(σ::AbstractVector{Int})
    visited = falses(length(σ))
    out = Int[]
    for i in eachindex(σ)
        visited[i] && continue
        len, j = 0, i
        while !visited[j]
            visited[j] = true
            j = σ[j]
            len += 1
        end
        push!(out, len)
    end
    return out
end

end # module Permanents
