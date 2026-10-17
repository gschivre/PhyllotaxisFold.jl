# Figures of the manual (docs/src, the method illustrations). Every figure is computed from simulated stems only; the
# validation figures on simulated data are made by docs/make_validation_figures.jl.
# Run from the docs environment (GLMakie):
#   julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make_figures.jl")'
# or, in a REPL with that environment active: include("docs/make_figures.jl")

using PhyllotaxisFold, Random, Statistics, Distributions, LogExpFunctions, Serialization
using SpecialFunctions: loggamma
using GLMakie

const FIG = joinpath(@__DIR__, "src", "figures")
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

#################################################################################################################
#################################################################################################################
# e1–e5. the running example: 40 simulated stems of 35 organs (WT-like delay law, twist in mm), one of them followed
# through the whole manual. The fit (≈ 5 min on 18 threads) is cached in docs/example.jls.
#################################################################################################################
const EX_CACHE = joinpath(@__DIR__, "example.jls")
const EX_LAW = DelayLaw(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)
const EX_TPRIOR = TorsionPrior(m0 = 0.0, sm = 2.0, νs = 4.0, τs = 1.0) # °/mm

function example_data()
    rng = Xoshiro(2024)
    S, n = 40, 35
    x = map(1:S) do _
        ℓ = [exp(log(9.0 - 6.0 * (p - 1) / (n - 2)) + 0.35 * randn(rng)) for p in 1:(n - 1)] # internodes 9 → 3 mm
        [0.0; cumsum(ℓ)]
    end
    sim = simulate_stems(EX_LAW; plants = S, organs = n, x, logμκ = 1.85, σκ = 0.15, torsion_mean = 1.0, torsion_sd = 0.7, rng)
    return x, sim, PhyllotaxisModelAD(sim.angles; torsion_x = x)
end

function example_fit()
    x, sim, m = example_data()
    isfile(EX_CACHE) && return (; deserialize(EX_CACHE)..., m)
    r = fit_map_torsion!(m; prior = EX_TPRIOR)
    dec = decode_orders(m, r; full = true)
    post = order_posterior(m, r; nsamples = 200, rng = Xoshiro(7))
    ex = (; x, sim, θ = r.fit.θ, r.m_ω, r.s_ω, r.τ, dec, post)
    serialize(EX_CACHE, ex)
    return (; ex..., m)
end

# the example stem: a few displaced organs, of both kinds (late and early), and a clear twist
function example_stem(ex)
    score(s) = (σ = ex.sim.orders[s]; nd = count(σ .!= eachindex(σ));
        (4 ≤ nd ≤ 9) * (any(σ .- eachindex(σ) .≤ -2) || any(σ .- eachindex(σ) .≥ 2)) * abs(ex.sim.torsion[s]))
    return argmax(score, eachindex(ex.sim.orders))
end

circdeg(a) = rad2deg(mod(a + π, 2π) - π) # angle in (−180°, 180°]

function fig_examples(ex = example_fit())
    s = example_stem(ex)
    σ, y, x, e = ex.sim.orders[s], ex.sim.angles[s], ex.x[s], ex.sim.delays[s]
    n, k = length(σ), (length(ex.θ.η) - 1) ÷ 2
    d = σ .- (1:n)
    off = d .!= 0
    dec = ex.dec[s]
    ψ, o, ϕ = dec.ψ, dec.o, dec.ϕ
    ω = ex.sim.torsion[s]

    # e1. the stem, its divergences and the timeline behind them
    let fig = Figure(size = (1500, 470))
        ax = Axis(fig[1, 1]; title = "(a) The organs: azimuth against height", xlabel = "stem length below the organ (mm)",
            ylabel = "azimuth (°)", yticks = 0:90:360, titlesize = 15)
        scatter!(ax, x, mod.(rad2deg.(ϕ .+ o .* (0:(n - 1)) .* ψ .+ deg2rad(dec.τ) .* x), 360); color = :white,
            strokecolor = :gray50, strokewidth = 1.2, markersize = 13, label = "site of the spiral at this height")
        scatter!(ax, x, mod.(rad2deg.(y), 360); color = [off[p] ? C_PATH : :black for p in 1:n], markersize = 8, label = "organ")
        axislegend(ax; position = :rt, framevisible = true, backgroundcolor = (:white, 0.85), labelsize = 11)
        ylims!(ax, 0, 400)
        ax2 = Axis(fig[1, 2]; title = "(b) Divergence between consecutive organs", xlabel = "height p (between p and p + 1)",
            ylabel = "divergence (°)", yticks = 0:45:360, titlesize = 15)
        dv = [mod(o * rad2deg(y[p + 1] - y[p]), 360) for p in 1:(n - 1)]
        hlines!(ax2, [rad2deg(ψ)]; color = :gray, linestyle = :dash)
        scatterlines!(ax2, 1:(n - 1), dv; color = [(off[p] || off[p + 1]) ? C_PATH : :black for p in 1:(n - 1)], markersize = 9,
            linewidth = 0.8)
        ylims!(ax2, 0, 360)
        ax3 = Axis(fig[1, 3]; title = "(c) Hidden: ready at j (○), commits at j + eⱼ (★)", xlabel = "time (plastochrons)",
            ylabel = "site j", titlesize = 15)
        for j in 1:n
            lines!(ax3, [j, j + e[j]], [j, j]; color = :gray)
        end
        late = Set(σ[p] for p in 1:n if d[p] != 0)
        scatter!(ax3, 1:n, 1:n; color = :white, strokecolor = :black, strokewidth = 1, markersize = 8)
        scatter!(ax3, (1:n) .+ e, 1:n; color = [j in late ? C_PATH : :seagreen for j in 1:n], marker = :star5, markersize = 12)
        save(joinpath(FIG, "e1_example_stem.png"), fig)
    end

    # e2. the twist: errors against the true sites, before and after removing the stem's rate
    let fig = Figure(size = (1100, 400))
        r = [circdeg(y[p] - ϕ - o * (σ[p] - 1) * ψ) for p in 1:n]
        r .-= circdeg(atan(mean(sin.(deg2rad.(r .- ω .* x))), mean(cos.(deg2rad.(r .- ω .* x))))) # centre
        r = circdeg.(deg2rad.(r .- ω .* x)) .+ ω .* x # unwrapped around the twist line
        ax = Axis(fig[1, 1]; title = "(a) Error of each angle with respect to its true site", xlabel = "stem length below the organ (mm)",
            ylabel = "error (°)", titlesize = 15)
        scatter!(ax, x, r; color = :black, markersize = 8)
        lines!(ax, [0, x[end]], [0, ω * x[end]]; color = C_PATH, linewidth = 2, label = "twist: $(round(ω; digits = 2)) °/mm")
        axislegend(ax; position = :lt, framevisible = false)
        ax2 = Axis(fig[1, 2]; title = "(b) The same once the twist is removed", xlabel = "stem length below the organ (mm)",
            ylabel = "error (°)", titlesize = 15)
        scatter!(ax2, x, circdeg.(deg2rad.(r .- ω .* x)); color = :black, markersize = 8)
        hlines!(ax2, [0.0]; color = :gray, linestyle = :dash)
        linkyaxes!(ax, ax2)
        save(joinpath(FIG, "e2_example_twist.png"), fig)
    end

    # e3. the posterior of the order, before and given the angles
    let fig = Figure(size = (1300, 470))
        ws = PermanentWorkspace(k, n)
        toeplitz_weights!(ws, ex.θ.η, n); forward!(ws, n); ρ0 = copy(backward!(ws, n))
        ρ1 = ex.post[s].ρ
        for (c, (ρ, ttl)) in enumerate(((ρ0, "(a) Before seeing the angles"), (ρ1, "(b) Given the angles")))
            ax = Axis(fig[1, c]; title = ttl, xlabel = "height p", ylabel = "displacement d = σ(p) − p", yticks = -k:k, titlesize = 15)
            hm = heatmap!(ax, 1:n, -k:k, ρ; colormap = :Blues, colorrange = (0, 1))
            scatter!(ax, 1:n, d; color = :transparent, strokecolor = C_PATH, strokewidth = 2, markersize = 14, label = "true d")
            c == 2 && scatter!(ax, 1:n, dec.σ .- (1:n); color = :white, strokecolor = :black, strokewidth = 0.5, marker = :xcross,
                markersize = 10, label = "most probable order")
            c == 2 && Colorbar(fig[1, 3], hm; label = "probability")
            axislegend(ax; position = :rb, framevisible = true, backgroundcolor = (:white, 0.8))
        end
        save(joinpath(FIG, "e3_example_posterior.png"), fig)
    end

    # e4. the propensity model on the 40 stems
    let fig = Figure(size = (1400, 420))
        ax = Axis(fig[1, 1]; title = "(a) Fitted costs η_d", xlabel = "displacement d", ylabel = "η_d", xticks = -k:k, titlesize = 15)
        barplot!(ax, -k:k, ex.θ.η; color = :steelblue, strokewidth = 0, gap = 0.25)
        fd, _ = estimate_displacement_frequencies(ex.m, [ex.θ]; N = 1, n)
        tr = [mean(count(abs.(σs .- (1:n)) .== a) / n for σs in ex.sim.orders) for a in 0:k]
        ax2 = Axis(fig[1, 2]; title = "(b) Frequency of |d|: model vs truth", xlabel = "|d|", ylabel = "fraction of organs",
            yscale = log10, xticks = 0:k, titlesize = 15)
        scatterlines!(ax2, 0:k, max.(fd, 1e-5); label = "fitted model", color = :steelblue, markersize = 12)
        scatterlines!(ax2, 0:k, max.(tr, 1e-5); label = "true orders", color = C_PATH, markersize = 12)
        axislegend(ax2; position = :rt, framevisible = false)
        acc = sort([mean(dd.σ .== σs) for (dd, σs) in zip(ex.dec, ex.sim.orders)])
        ax3 = Axis(fig[1, 3]; title = "(c) Organs on their correct site, per stem", xlabel = "stems (sorted)",
            ylabel = "fraction correct", titlesize = 15)
        barplot!(ax3, 1:length(acc), acc; color = :steelblue, strokewidth = 0, gap = 0.2)
        ylims!(ax3, 0, 1.02)
        save(joinpath(FIG, "e4_example_propensity.png"), fig)
    end

    # e5. re-weighting the posterior orders of the example stem
    let fig = Figure(size = (1500, 440))
        smp = order_samples(ex.post, ex.θ.η)
        ss = smp[s]
        ordr = sortperm(ss.w; rev = true)
        ax = Axis(fig[1, 1]; title = "(a) Distinct posterior orders of the stem", xlabel = "order (by frequency)",
            ylabel = "frequency among 200 samples", titlesize = 15, xticks = 1:length(ordr))
        barplot!(ax, 1:length(ordr), ss.w[ordr]; color = [ss.orders[j] == σ ? C_PATH : :steelblue for j in ordr], strokewidth = 0)
        lw = [loglik_order_exact(ss.orders[j], EX_LAW) - ss.logA[j] for j in ordr]
        ax2 = Axis(fig[1, 2]; title = "(b) log P_f(σ) − log P_A(σ), true delay law", xlabel = "order (by frequency)",
            ylabel = "log weight", titlesize = 15, xticks = 1:length(ordr))
        barplot!(ax2, 1:length(ordr), lw; color = [ss.orders[j] == σ ? C_PATH : :steelblue for j in ordr], strokewidth = 0)
        sgrid = range(0.2, 0.8; length = 25)
        withs(sd) = DelayLaw(sd; πL = EX_LAW.πL, mL = EX_LAW.mL, πE = EX_LAW.πE, mE = EX_LAW.mE)
        tot = [relative_loglik(smp, withs(sd)) for sd in sgrid]
        one = [stem_terms([ss], withs(sd))[1] for sd in sgrid]
        ax3 = Axis(fig[1, 3]; title = "(c) Δℓ against the core sd (tails at their true values)", xlabel = "core sd s (plastochrons)",
            ylabel = "Δℓ, 40 stems", titlesize = 15)
        lines!(ax3, sgrid, tot; color = :black, linewidth = 3, label = "sum over the 40 stems")
        vlines!(ax3, [EX_LAW.s]; color = :gray, linestyle = :dash)
        ax3b = Axis(fig[1, 3]; yaxisposition = :right, ylabel = "Δℓ_s, example stem", ylabelcolor = C_PATH, yticklabelcolor = C_PATH)
        hidexdecorations!(ax3b); hidespines!(ax3b)
        lines!(ax3b, sgrid, one; color = C_PATH, linewidth = 1.5)
        axislegend(ax3; position = :lb, framevisible = false)
        save(joinpath(FIG, "e5_example_reweighting.png"), fig)
    end
    return s
end

if abspath(PROGRAM_FILE) == @__FILE__
    fig_examples()
end
