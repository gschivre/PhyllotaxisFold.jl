# Figures of docs/method.md (the method illustrations). Every figure is computed from simulated stems only; the
# validation figures on simulated data are made by docs/make_validation_figures.jl.
# Run from the docs environment (GLMakie):
#   julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make_figures.jl")'
# or, in a REPL with that environment active: include("docs/make_figures.jl")

using PhyllotaxisFold, Random, Statistics, Distributions, LogExpFunctions, Serialization
using SpecialFunctions: loggamma
using GLMakie

const FIG = joinpath(@__DIR__, "figures")
mkpath(FIG)
const PF = PhyllotaxisFold

const C_USED, C_FREE, C_BAND, C_PATH = :gray25, :white, (:steelblue, 0.15), :darkorange
const C_LATE, C_EARLY = :firebrick, :royalblue

# the states of the graph in the order of PermutationGraph (masks of 2k bits with k set)
graph_masks(k) = [m for m in 0:((1 << (2k)) - 1) if count_ones(m) == k]

# the state sequence (indices) and the window masks visited by a banded permutation σ
function state_path(g::PermutationGraph, σ)
    k = g.k
    masks = graph_masks(k)
    idx = Dict(m => i for (i, m) in enumerate(masks))
    m = (1 << k) - 1
    path = [idx[m]]
    for (i, s) in enumerate(σ)
        d = s - i
        @assert abs(d) ≤ k
        b = d + k # bit of column i + d in the window {i − k, …, i + k}, bit 2k = the entering column
        @assert (m & (1 << b)) == 0
        m = (m | (1 << b)) >> 1
        push!(path, idx[m])
    end
    return path
end

# draw the window of a mask as 2k small squares at (x0 + b·w, y0)
function draw_mask!(ax, mask, k, x0, y0; w = 0.18, used = C_USED, free = C_FREE)
    for b in 0:(2k - 1)
        c = ((mask >> b) & 1 == 1) ? used : free
        poly!(ax, Rect2f(x0 + b * w, y0 - w / 2, w * 0.9, w * 0.9); color = c, strokecolor = :black, strokewidth = 0.8)
    end
end

#################################################################################################################
#################################################################################################################
# m0. the fold: speed of the switch variable for three margins
#################################################################################################################
let fig = Figure(size = (900, 330))
    X = range(-2.5, 2.5; length = 400)
    for (j, (μ, ttl)) in enumerate(((-1.0, "μ < 0: two states (stable low, threshold)"), (0.0, "μ = 0: the fold"), (0.6, "μ > 0: no state left, a slow \"ghost\"")))
        ax = Axis(fig[1, j]; title = ttl, xlabel = "X (state of the switch)", ylabel = j == 1 ? "speed dX/ds = μ + X²" : "", titlesize = 13)
        lines!(ax, X, μ .+ X .^ 2; color = :black, linewidth = 2)
        hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
        if μ < 0
            scatter!(ax, [-sqrt(-μ)], [0.0]; color = :blue, markersize = 14)
            scatter!(ax, [sqrt(-μ)], [0.0]; color = :white, strokecolor = :red, strokewidth = 2, markersize = 14)
        elseif μ == 0
            scatter!(ax, [0.0], [0.0]; color = :purple, markersize = 14)
        else
            band!(ax, [-0.6, 0.6], [-1.2, -1.2], [5.0, 5.0]; color = (:orange, 0.2))
            text!(ax, 0.0, 3.5; text = "slow zone", align = (:center, :center))
        end
        ylims!(ax, -1.2, 5)
    end
    save(joinpath(FIG, "m0_fold.png"), fig)
end

# m1. a permutation is a path: the assignment grid, the window state and the trellis
#################################################################################################################
let k = 2, n = 8, σ = [2, 1, 3, 5, 6, 4, 7, 8], irow = 5
    g = PermutationGraph(k)
    masks = graph_masks(k)
    path = state_path(g, σ)
    fig = Figure(size = (1500, 560))

    # (a) assignment grid
    ax = Axis(fig[1, 1]; title = "(a) An order is an assignment of stem positions to sites", xlabel = "canonical site j (spiral order)",
        ylabel = "stem position p (bottom → top)", xticks = 1:n, yticks = 1:n, aspect = DataAspect(), titlesize = 15)
    for p in 1:n, j in 1:n
        abs(j - p) ≤ k && poly!(ax, Rect2f(j - 0.5, p - 0.5, 1, 1); color = C_BAND, strokecolor = (:gray, 0.4), strokewidth = 0.5)
    end
    # columns used by rows 1..irow−1, and the window seen by row irow
    used = Set(σ[1:(irow - 1)])
    poly!(ax, Rect2f(irow - k - 0.5, irow - 0.5, 2k, 1); color = :transparent, strokecolor = C_PATH, strokewidth = 3)
    for p in 1:n
        scatter!(ax, [σ[p]], [p]; color = (p < irow ? C_USED : p == irow ? C_PATH : (:gray, 0.5)), markersize = 20)
    end
    for j in used
        lines!(ax, [j, j], [irow - 0.45, n + 0.5]; color = (:gray, 0.6), linestyle = :dash)
    end
    text!(ax, irow - k - 0.45, irow + 0.55; text = "window of row $irow", color = C_PATH, fontsize = 13)
    text!(ax, 0.6, n + 0.9; text = "σ = " * join(σ, " ") * "    (shaded: |σ(p) − p| ≤ k = $k)", fontsize = 13)
    xlims!(ax, 0.4, n + 0.6); ylims!(ax, 0.4, n + 1.4)

    # (b) the window states along σ
    ax2 = Axis(fig[1, 2]; title = "(b) The only memory needed: which window columns are used", titlesize = 15,
        xlabel = "", ylabel = "before row p", yreversed = true, yticks = 1:(n + 1))
    hidexdecorations!(ax2); hidespines!(ax2, :t, :r, :b)
    for (step, s) in enumerate(path)
        draw_mask!(ax2, masks[s], k, 0.0, step; w = 0.5)
        for b in 0:(2k - 1)
            col = step - k + b # column index of bit b before row `step`
            text!(ax2, b * 0.5 + 0.22, step - 0.42; text = (1 ≤ col ≤ n ? string(col) : "·"), fontsize = 10, align = (:center, :center), color = :gray40)
        end
        if step ≤ n
            d = σ[step] - step
            text!(ax2, 2.3, step + 0.5; text = "row $step → site $(σ[step])  (d = $(d ≥ 0 ? "+" : "")$d)", fontsize = 12,
                align = (:left, :center), color = (step == irow ? C_PATH : :black))
        end
    end
    text!(ax2, 0.0, 0.25; text = "filled = used, numbers = sites in the window", fontsize = 11, color = :gray30)
    xlims!(ax2, -0.2, 6.2); ylims!(ax2, n + 1.6, -0.1)

    # (c) the trellis: every banded permutation is one path from start to end
    ax3 = Axis(fig[1, 3]; title = "(c) All $(round(Int, exp(PF.Permanents.forward!(let ws = PermanentWorkspace(k, n); toeplitz_weights!(ws, zeros(2k + 1), n); ws end, n)))) banded orders = all paths of this graph", titlesize = 15,
        xlabel = "step p (row being assigned)", xticks = 1:(n + 1), yticks = (1:g.M, fill("", g.M)))
    for i in 1:n, s in 1:g.M, e in g.edge_ptr[s]:(g.edge_ptr[s + 1] - 1)
        d = g.edge_d[e]
        (1 ≤ i + d ≤ n) || continue
        lines!(ax3, [i, i + 1], [s, g.edge_next[e]]; color = (:gray, 0.35), linewidth = 1)
    end
    # reachable states only (forward from start and backward from end, with unit weights)
    ws = PermanentWorkspace(k, n); toeplitz_weights!(ws, zeros(2k + 1), n); forward!(ws, n); backward!(ws, n)
    for i in 1:(n + 1), s in 1:g.M
        reach = isfinite(ws.F[s, i]) && ws.F[s, i] > -1e300 && (i == n + 1 ? s == g.end_s : isfinite(ws.B[s, i]) && ws.B[s, i] > -1e300)
        scatter!(ax3, [i], [s]; color = (reach ? :black : (:gray, 0.25)), markersize = 7)
    end
    lines!(ax3, 1:(n + 1), path; color = C_PATH, linewidth = 4)
    scatter!(ax3, 1:(n + 1), path; color = C_PATH, markersize = 12)
    for (s, m) in enumerate(masks)
        draw_mask!(ax3, m, k, -0.05, s; w = 0.17)
    end
    text!(ax3, 1, g.start_s + 0.35; text = "start", fontsize = 12, align = (:center, :bottom))
    text!(ax3, n + 1, g.end_s + 0.35; text = "end", fontsize = 12, align = (:center, :bottom))
    xlims!(ax3, -0.15, n + 1.5)
    colsize!(fig.layout, 1, Relative(0.3)); colsize!(fig.layout, 2, Relative(0.27))
    save(joinpath(FIG, "m1_dp_paths.png"), fig)
end

#################################################################################################################
# m2. why it matters: number of orders vs cost of the DP
#################################################################################################################
let ns = 2:60
    fig = Figure(size = (1100, 420))
    ax = Axis(fig[1, 1]; xlabel = "number of organs n", ylabel = "log₁₀(count)", title = "(a) Orders to sum over vs work done by the DP",
        titlesize = 15)
    lines!(ax, ns, [loggamma(n + 1) / log(10) for n in ns]; color = :black, linewidth = 3, label = "all orders (n!)")
    cols = Makie.wong_colors()
    for (c, k) in enumerate(1:4)
        cnt = [begin
            ws = PermanentWorkspace(k, n); toeplitz_weights!(ws, zeros(2k + 1), n); forward!(ws, n) / log(10)
        end for n in ns]
        g = PermutationGraph(k)
        ops = [log10(n * length(g.edge_d)) for n in ns]
        lines!(ax, ns, cnt; color = cols[c], linewidth = 2.5, label = "orders with |d| ≤ $k")
        lines!(ax, ns, ops; color = cols[c], linewidth = 2, linestyle = :dash, label = "DP operations, k = $k")
    end
    axislegend(ax; position = :lt, framevisible = false, labelsize = 11, nbanks = 1)
    ax2 = Axis(fig[1, 2]; xlabel = "maximum displacement k", ylabel = "", title = "(b) Size of the DP for one stem of n = 50",
        xticks = 1:6, titlesize = 15, yscale = log10)
    ks = 1:6
    states = [binomial(2k, k) for k in ks]
    edges = [length(PermutationGraph(k).edge_d) for k in ks]
    scatterlines!(ax2, ks, states; label = "states C(2k, k)", linewidth = 2.5, markersize = 12)
    scatterlines!(ax2, ks, 50 .* edges; label = "edge operations for n = 50", linewidth = 2.5, markersize = 12)
    axislegend(ax2; position = :lt, framevisible = false)
    save(joinpath(FIG, "m2_dp_scaling.png"), fig)
end

#################################################################################################################
# m3. what the forward–backward pass returns: marginal posterior of the displacements, prior vs given the angles
#################################################################################################################
let rng = Xoshiro(11), n = 30, k = 4, κ = 2.0
    law = DelayLaw(0.45; πL = 0.08, mL = 1.2, πE = 0.08, mE = 1.0) # fold commitment with two tails (Level 0)
    σ = sortperm([i + rand(rng, law) for i in 1:n])
    ψ, o, ϕ = Golden_angle, 1, 0.7
    y = [mod2pi(rand(rng, VonMises(ϕ + o * (σ[p] - 1) * ψ, κ))) for p in 1:n]
    η = [-3.1, -2.5, -1.9, -0.8, 0.0, -0.8, -1.9, -2.5, -3.1] # propensity-model costs of the order of hda9 WT
    ws = PermanentWorkspace(k, n)
    toeplitz_weights!(ws, η, n); forward!(ws, n); ρ0 = copy(backward!(ws, n))
    for d in -k:k, i in max(1, 1 - d):min(n, n - d)
        ws.w[i, d + k + 1] = η[d + k + 1] + κ * (cos(y[i] - ϕ - o * (i + d - 1) * ψ) - 1)
    end
    forward!(ws, n); ρ1 = copy(backward!(ws, n))
    σmap = map_permutation(ws.g, ws.w[1:n, :], n)
    fig = Figure(size = (1300, 470))
    for (c, (ρ, ttl)) in enumerate(((ρ0, "(a) Before seeing the angles: P(σ(p) = p + d | η)"), (ρ1, "(b) Given the angles: P(σ(p) = p + d | y, η, ϕ, o, ψ, κ)")))
        ax = Axis(fig[1, c]; title = ttl, xlabel = "stem position p", ylabel = "displacement d", yticks = -k:k, titlesize = 15)
        hm = heatmap!(ax, 1:n, -k:k, ρ; colormap = :Blues, colorrange = (0, 1))
        scatter!(ax, 1:n, σ .- (1:n); color = :transparent, strokecolor = C_PATH, strokewidth = 2, markersize = 14, label = "true d")
        c == 2 && scatter!(ax, 1:n, σmap .- (1:n); color = :black, marker = :xcross, markersize = 9, label = "Viterbi (MAP order)")
        c == 2 && Colorbar(fig[1, 3], hm; label = "probability")
        axislegend(ax; position = :rb, framevisible = true, backgroundcolor = (:white, 0.8))
    end
    save(joinpath(FIG, "m3_dp_marginals.png"), fig)
end


#################################################################################################################
#################################################################################################################
# m9. a swap seen in the timeline, the internodes and the angles (hand-made Level-0 stem, plateau-like late organ)
#################################################################################################################
let n = 12
    τ = collect(1.0:n) .+ [0.05, -0.1, 0.12, 0.0, -0.08, 1.35, -0.12, 0.1, -0.05, 0.08, 0.0, -0.1]
    τ[6] = 7.15 # site 6 commits late, after site 7
    σ = sortperm(τ) # site at each stem position
    t = τ[σ]
    ℓ = diff(t) .* 5.5
    ψ = Golden_angle
    y = [mod2pi((σ[p] - 1) * ψ) for p in 1:n]
    div = [mod(rad2deg(y[p + 1] - y[p]), 360) for p in 1:(n - 1)]
    fig = Figure(size = (1400, 430))
    ax = Axis(fig[1, 1]; xlabel = "time (plastochrons)", ylabel = "site j (canonical order)", yticks = 1:n, titlesize = 15,
        title = "(a) Level 0: ready at j (○), commits at τ_j (★)")
    for j in 1:n
        lines!(ax, [j, τ[j]], [j, j]; color = :gray)
    end
    scatter!(ax, 1:n, 1:n; color = :white, strokecolor = :black, strokewidth = 1.5, markersize = 12)
    scatter!(ax, τ, 1:n; color = [j ∈ (6, 7) ? C_PATH : :seagreen for j in 1:n], marker = :star5, markersize = 18)
    text!(ax, 1, n + 0.3; text = "stem order σ = " * join(σ, " "), fontsize = 13, align = (:left, :bottom))
    ylims!(ax, 0.3, n + 1.2)
    ax2 = Axis(fig[1, 2]; xlabel = "internode between stem positions p and p + 1", ylabel = "internode (mm)", xticks = 1:(n - 1),
        title = "(b) Internodes: ≈ 2 plastochrons, near zero, just under 1", titlesize = 15)
    barplot!(ax2, 1:(n - 1), ℓ; color = [p ∈ (5, 6, 7) ? C_PATH : :seagreen for p in 1:(n - 1)])
    hlines!(ax2, [5.5]; color = :gray, linestyle = :dash)
    text!(ax2, 0.6, 5.7; text = "one plastochron", fontsize = 12, color = :gray30)
    ax3 = Axis(fig[1, 3]; xlabel = "divergence between positions p and p + 1", ylabel = "divergence angle (°)", xticks = 1:(n - 1),
        title = "(c) Angles: 2ψ, −ψ, 2ψ instead of ψ, ψ, ψ", titlesize = 15, yticks = 0:45:360)
    scatterlines!(ax3, 1:(n - 1), div; color = [p ∈ (5, 6, 7) ? C_PATH : :seagreen for p in 1:(n - 1)], markersize = 14, linewidth = 1)
    hlines!(ax3, [rad2deg(ψ)]; color = :gray, linestyle = :dash)
    ylims!(ax3, 0, 360)
    save(joinpath(FIG, "m9_swap_signature.png"), fig)
end

