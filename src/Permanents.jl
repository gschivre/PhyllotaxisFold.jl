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
`toeplitz_weights!` fills the stationary case η_d.

Convention: σ[i] is the canonical index of the organ observed at position i, displacement d = σ[i] − i.
"""
module Permanents

using Random, LogExpFunctions
using ..Common: _isneginf, _logaddexp, _fullvalue

export PermutationGraph, PermanentWorkspace, toeplitz_weights!, forward!, backward!, sample_permutation!, map_permutation,
    logprob_permutation, cycle_lengths

struct PermutationGraph
    k::Int
    M::Int # number of states (binomial(2k, k))
    start_s::Int
    end_s::Int
    edge_ptr::Vector{Int} # CSR: edges of state s are edge_ptr[s]:(edge_ptr[s + 1] − 1)
    edge_next::Vector{Int}
    edge_d::Vector{Int}
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
    return PermutationGraph(k, M, start_s, start_s, edge_ptr, edge_next, edge_d)
end

"""
    PermanentWorkspace(T, k, n_max; logspace = false)

Graph and buffers for permanents of size ≤ n_max with element type T (Float64, or a ForwardDiff dual type: build one
workspace per number type, they are cheap). The caller always writes log-weights in `ws.w`.
- `logspace = false` (default): scaled linear space (HMM scaling trick), about twice as fast; each call falls back to
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
    rho::Matrix{T} # edge marginals P(σ[i] = i + d), n_max × (2k + 1)
    inlog::Base.RefValue{Bool} # whether the last forward! ran in log-space (read by backward! and sampling)
    nfallback::Base.RefValue{Int} # number of scaled calls redone in log-space (diagnostic)
end
function PermanentWorkspace(::Type{T}, k::Int, n_max::Int; logspace::Bool = false) where {T}
    g = PermutationGraph(k)
    K = 2 * k + 1
    return PermanentWorkspace{T}(
        g, n_max, logspace, zeros(T, g.M, n_max + 1), zeros(T, g.M, n_max + 1), zeros(T, n_max, K),
        zeros(T, n_max, K), zeros(T, n_max, K), Ref(logspace), Ref(0)
    )
end
PermanentWorkspace(k::Int, n_max::Int; kwargs...) = PermanentWorkspace(Float64, k, n_max; kwargs...)

"""
    toeplitz_weights!(ws, η, n)

Stationary log-weights w[i, d] = η_d for every row; η has length 2k + 1, η[d + k + 1] = η_d (η_0 = 0 by gauge).
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
- scaled linear space: row i of the weights is exponentiated relative to its largest entry m_i, the messages are
  renormalised at each step (Σ_s F[s, i] = 1) and log perm = Σ_i (m_i + log c_i). The row maxima absorb a row where
  every candidate is unlikely (an outlier angle at large κ); the call is redone in log-space when a step normaliser
  falls below `SCALED_STEP_MIN` or the accumulated deficit exceeds `SCALED_DEFICIT_MAX` (see above), so the result
  never depends on underflow. `ws.inlog[]` records which pass produced F.
"""
function forward!(ws::PermanentWorkspace{T}, n::Int) where {T}
    ws.inlog[] = ws.logspace
    ws.logspace && return _forward_log!(ws, n)
    l = _forward_scaled!(ws, n)
    l === nothing || return l
    ws.inlog[] = true
    ws.nfallback[] += 1
    return _forward_log!(ws, n)
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

# scaled pass; returns nothing when it cannot be trusted (the caller then runs the log-space pass)
function _forward_scaled!(ws::PermanentWorkspace{T}, n::Int) where {T}
    (; g, F, w, ew) = ws
    (; k, M, edge_ptr, edge_next, edge_d) = g
    @inbounds begin
        logZ = zero(T)
        for i in 1:n
            # largest log-weight among the columns that exist in row i
            lo, hi = max(-k, 1 - i), min(k, n - i)
            mi = T(-Inf)
            for d in lo:hi
                mi = max(mi, w[i, d + k + 1])
            end
            _isneginf(mi) && return T(-Inf) # row i has no admissible column: perm = 0 exactly
            for d in lo:hi
                ew[i, d + k + 1] = exp(w[i, d + k + 1] - mi)
            end
            logZ += mi
        end
        fill!(view(F, :, 1:(n + 1)), zero(T))
        F[g.start_s, 1] = one(T)
        deficit = 0.0
        for i in 1:n
            for s in 1:M
                val = F[s, i]
                val == 0 && continue
                for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
                    d = edge_d[e]
                    (1 ≤ i + d ≤ n) || continue
                    F[edge_next[e], i + 1] += val * ew[i, d + k + 1]
                end
            end
            c = zero(T)
            for s in 1:M
                c += F[s, i + 1]
            end
            (c > SCALED_STEP_MIN) || return nothing
            lc = log(c)
            deficit -= min(0.0, _fullvalue_f(lc))
            (deficit ≤ SCALED_DEFICIT_MAX) || return nothing
            logZ += lc
            ic = inv(c)
            for s in 1:M
                F[s, i + 1] *= ic
            end
        end
        # one state is reachable after the last step (all columns used): F_end = 1 up to rounding
        return logZ + log(F[g.end_s, n + 1])
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
        # a backward normaliser underflowed: redo both passes in log-space
        ws.inlog[] = true
        ws.nfallback[] += 1
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
    (; k, M, edge_ptr, edge_next, edge_d) = g
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
                for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
                    d = edge_d[e]
                    (1 ≤ i + d ≤ n) || continue
                    val_B = B[edge_next[e], i + 1] * ew[i, d + k + 1]
                    acc += val_B
                    rho[i, d + k + 1] += val_F * val_B
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

Exact sample of σ with P(σ) ∝ ∏_i exp(w[i, σ[i] − i]) (log-weights in ws.w): forward and backward passes, then from the
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
