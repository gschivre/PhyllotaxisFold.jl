# Figures of docs/method.md. Every figure is computed from the passage tables and simulated stems only.
# Run from the docs environment (GLMakie):
#   julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make_figures.jl")'
# or, in a REPL with that environment active: include("docs/make_figures.jl")

using PhyllotaxisFold, Random, Statistics, Distributions, LogExpFunctions, Serialization
using SpecialFunctions: loggamma
using GLMakie

const FIG = joinpath(@__DIR__, "figures")
mkpath(FIG)
const PF = PhyllotaxisFold
const TAB = load_tables(joinpath(pkgdir(PhyllotaxisFold), "tables", "tables_k4.jls"))
const RTAB = load_tables(joinpath(pkgdir(PhyllotaxisFold), "tables", "tables_ramp_k4.jls"))

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
let rng = Xoshiro(11), n = 30, k = 4, μ = 2.0, γ = 1.2, κ = 2.0
    (; tab, ptab) = TAB
    smp = passage_sampler(tab, μ)
    σ = sortperm([i * γ + smp(rng) for i in 1:n])
    ψ, o, ϕ = Golden_angle, 1, 0.7
    y = [mod2pi(rand(rng, VonMises(ϕ + o * (σ[p] - 1) * ψ, κ))) for p in 1:n]
    η = zeros(2k + 1); propensity!(η, nothing, ptab, μ, log(γ))
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
# m4. the passage law: plateau (late tail) and ramp (early tail)
#################################################################################################################
let tab = TAB.tab, rt = RTAB.tab
    fig = Figure(size = (1300, 450))
    reds = cgrad(:Reds, 6; categorical = true)
    blues = cgrad(:Blues, 5; categorical = true)
    for (c, ttl) in enumerate(("(a) Standardised commitment-time density g(z)", "(b) Same on a log scale: the tails decide the permutations"))
        ax = Axis(fig[1, c]; xlabel = "standardised commitment time z = (t − mean) / sd", ylabel = (c == 1 ? "g(z)" : "log₁₀ g(z)"),
            title = ttl, titlesize = 15)
        tr(v) = (c == 1 ? exp.(v) : v ./ log(10))
        for (i, μ) in enumerate((0.0, 2.0, 5.0, 10.0))
            lines!(ax, tab.z_grid, tr(passage_column(tab, μ)); color = reds[i + 2], linewidth = 2.5, label = "plateau μ = $(μ)")
        end
        for (i, lν) in enumerate((-2.0, 0.0, 2.0))
            lines!(ax, rt.z_grid, tr(passage_column(rt, lν)); color = blues[i + 2], linewidth = 2.5, linestyle = :dash, label = "ramp log ν = $(lν)")
        end
        zz = range(-6, 7; length = 400)
        lines!(ax, zz, tr(logpdf.(Normal(), zz)); color = :black, linestyle = :dot, linewidth = 2, label = "Gaussian")
        xlims!(ax, -5, 6.5)
        c == 2 && ylims!(ax, -7, 0)
        c == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = 12)
    end
    save(joinpath(FIG, "m4_passage_shapes.png"), fig)
end

# nearest table node
_node(pt, shape, γ) = (argmin(abs.(pt.μ_grid .- shape)), argmin(abs.(pt.logγ_grid .- log(γ))))

#################################################################################################################
# m5. from the passage law to the displacement law and to η (plateau vs ramp, same γ)
#################################################################################################################
let k = 4, γ = 1.5
    cases = (("plateau μ = 1", TAB.ptab, 1.0), ("plateau μ = 10", TAB.ptab, 10.0), ("ramp log ν = −1", RTAB.ptab, -1.0))
    fig = Figure(size = (1350, 640))
    ds = -k:k
    cols = [d < 0 ? C_LATE : d > 0 ? C_EARLY : :gray50 for d in ds]
    for (c, (lab, pt, shape)) in enumerate(cases)
        iμ, iγ = _node(pt, shape, γ)
        pd = pt.pd[:, iμ, iγ] ./ sum(pt.pd[:, iμ, iγ])
        ax = Axis(fig[1, c]; title = "$lab, γ = $(round(exp(pt.logγ_grid[iγ]); digits = 2)): P(d) (exact Level 0)", ylabel = (c == 1 ? "log₁₀ P(d)" : ""),
            xticks = ds, titlesize = 14)
        barplot!(ax, ds, log10.(max.(pd, 1e-14)); fillto = -14, color = cols)
        ylims!(ax, -14, 0.5)
        ax2 = Axis(fig[2, c]; title = "matched log-propensity η_d", xlabel = "displacement d = σ(p) − p", ylabel = (c == 1 ? "η_d" : ""),
            xticks = ds, titlesize = 14)
        barplot!(ax2, ds, pt.η[:, iμ, iγ]; color = cols)
        ylims!(ax2, -16, 0.5)
    end
    Label(fig[0, :], "red: d < 0, the organ came late (it sits above its canonical place); blue: d > 0, it came early"; fontsize = 13)
    save(joinpath(FIG, "m5_displacement_eta.png"), fig)
end

#################################################################################################################
# m6. what γ and the shape do to η
#################################################################################################################
let pt = TAB.ptab, rp = RTAB.ptab, k = 4
    fig = Figure(size = (1300, 470))
    ax = Axis(fig[1, 1]; xlabel = "γ = readiness spacing / passage sd", ylabel = "½ (η₂ + η₋₂)", xscale = log10,
        title = "(a) γ sets the depth: mean log-propensity of |d| = 2", titlesize = 15, xticks = [0.5, 1, 2, 4, 8])
    γs = exp.(pt.logγ_grid)
    for (μ, col) in ((0.0, :orange), (5.0, :firebrick), (10.0, :darkred))
        iμ = argmin(abs.(pt.μ_grid .- μ))
        lines!(ax, γs, 0.5 .* (pt.η[k + 3, iμ, :] .+ pt.η[k - 1, iμ, :]); color = col, linewidth = 2.5, label = "plateau μ = $μ")
    end
    for (lν, col) in ((-2.0, :skyblue), (2.0, :navy))
        iμ = argmin(abs.(rp.μ_grid .- lν))
        lines!(ax, exp.(rp.logγ_grid), 0.5 .* (rp.η[k + 3, iμ, :] .+ rp.η[k - 1, iμ, :]); color = col, linewidth = 2.5, linestyle = :dash, label = "ramp log ν = $lν")
    end
    ylims!(ax, -25, 0.5)
    axislegend(ax; position = :lb, framevisible = false)
    ax2 = Axis(fig[1, 2]; xlabel = "shape: plateau μ (solid) / ramp log ν (dashed)", ylabel = "η₋₂ − η₊₂",
        title = "(b) The shape sets the asymmetry (> 0: late organs favoured)", titlesize = 15)
    for (γ, col) in ((1.0, :gray70), (1.5, :gray45), (2.2, :black))
        iγ = argmin(abs.(pt.logγ_grid .- log(γ)))
        lines!(ax2, pt.μ_grid, pt.η[k - 1, :, iγ] .- pt.η[k + 3, :, iγ]; color = col, linewidth = 2.5, label = "γ = $γ")
        lines!(ax2, rp.μ_grid, rp.η[k - 1, :, iγ] .- rp.η[k + 3, :, iγ]; color = col, linewidth = 2.5, linestyle = :dash)
    end
    hlines!(ax2, [0.0]; color = :gray, linestyle = :dot)
    axislegend(ax2; position = :rt, framevisible = false)
    save(joinpath(FIG, "m6_eta_gamma_shape.png"), fig)
end

#################################################################################################################
# m7. the distribution of η implied by the fold prior vs the softplus prior of model A
#################################################################################################################
function fold_eta_draws(pt, mshape, sshape, mγ, sγ; N = 4000, rng = Xoshiro(3))
    lo, hi = extrema(pt.μ_grid); glo, ghi = extrema(pt.logγ_grid)
    out = zeros(2pt.k + 1, N); par = zeros(2, N)
    for i in 1:N
        s = rand(rng, truncated(Normal(mshape, sshape), lo, hi)); lg = rand(rng, truncated(Normal(mγ, sγ), glo, ghi))
        propensity!(view(out, :, i), nothing, pt, s, lg); par[:, i] .= (s, lg)
    end
    return out, par
end
function softplus_eta_draws(prior::PhyllotaxisPrior, symmetric; N = 4000, rng = Xoshiro(4))
    k = length(prior.μΔ)
    dpos = LogNormalSoftPlus([prior.μΔ...], [prior.σΔ...]); dneg = ReverseLogNormalSoftPlus(reverse([prior.μΔ...]), reverse([prior.σΔ...])) # gaps from the outside in
    out = zeros(2k + 1, N)
    for i in 1:N
        ep = rand(rng, dpos)
        en = symmetric ? reverse(ep) : rand(rng, dneg)
        out[:, i] .= vcat(en, 0.0, ep)
        gauge!(view(out, :, i), k)
    end
    return out
end
let k = 4, N = 4000
    fp, parp = fold_eta_draws(TAB.ptab, 2.0, 2.0, log(2.0), 0.5)
    fr, parr = fold_eta_draws(RTAB.ptab, 0.0, 1.5, log(2.0), 0.5)
    as = softplus_eta_draws(PhyllotaxisPrior(), true)
    aa = softplus_eta_draws(PhyllotaxisPrior(), false)
    fig = Figure(size = (1450, 480))
    ax = Axis(fig[1, 1]; xlabel = "displacement d", ylabel = "η_d (median, 5–95 %)", xticks = -k:k, titlesize = 15,
        title = "(a) Prior marginals of η_d")
    for (j, (D, lab, col)) in enumerate(((as, "model A prior (softplus, symmetric)", :gray40), (fp, "fold prior, plateau", :firebrick), (fr, "fold prior, ramp", :royalblue)))
        x = collect(-k:k) .+ (j - 2) * 0.22
        q = [quantile(D[i, :], p) for i in 1:(2k + 1), p in (0.05, 0.5, 0.95)]
        rangebars!(ax, x, q[:, 1], q[:, 3]; color = col, linewidth = 2.5, whiskerwidth = 6)
        scatter!(ax, x, q[:, 2]; color = col, markersize = 10, label = lab)
    end
    ylims!(ax, -30, 1)
    axislegend(ax; position = :cb, framevisible = false, labelsize = 12)
    ax2 = Axis(fig[1, 2]; xlabel = "η₊₂ (early by 2)", ylabel = "η₋₂ (late by 2)", titlesize = 15, aspect = 1,
        title = "(b) Asymmetry: A is symmetric, the fold is not")
    scatter!(ax2, aa[k + 3, :], aa[k - 1, :]; color = (:gray, 0.15), markersize = 4, label = "A prior, asymmetric variant")
    scatter!(ax2, as[k + 3, :], as[k - 1, :]; color = (:black, 0.5), markersize = 4, label = "A prior, symmetric")
    scatter!(ax2, fp[k + 3, :], fp[k - 1, :]; color = (:firebrick, 0.3), markersize = 4, label = "fold, plateau")
    scatter!(ax2, fr[k + 3, :], fr[k - 1, :]; color = (:royalblue, 0.3), markersize = 4, label = "fold, ramp")
    lines!(ax2, [-25, 0], [-25, 0]; color = :gray, linestyle = :dot)
    xlims!(ax2, -25, 0.5); ylims!(ax2, -25, 0.5)
    axislegend(ax2; position = :lt, framevisible = false, labelsize = 11)
    ax3 = Axis(fig[1, 3]; xlabel = "η_{±1}", ylabel = "½ (η₊₃ + η₋₃)", titlesize = 15, aspect = 1,
        title = "(c) Coupling: the fold puts η on a 2-D surface")
    scatter!(ax3, aa[k + 2, :], 0.5 .* (aa[k + 4, :] .+ aa[k - 2, :]); color = (:gray, 0.15), markersize = 4)
    scatter!(ax3, as[k + 2, :], 0.5 .* (as[k + 4, :] .+ as[k - 2, :]); color = (:black, 0.5), markersize = 4)
    sc = scatter!(ax3, fp[k + 2, :], 0.5 .* (fp[k + 4, :] .+ fp[k - 2, :]); color = parp[2, :], colormap = :viridis, markersize = 4)
    Colorbar(fig[1, 4], sc; label = "log γ (fold, plateau)")
    xlims!(ax3, -4, 0.2); ylims!(ax3, -30, 0.5)
    save(joinpath(FIG, "m7_implied_eta_prior.png"), fig)
end

# realised bulk displacement law of the permanent model with log-propensities η (n = 60, 2k trimmed at each end)
function realised_pd(η; n = 60)
    k = (length(η) - 1) ÷ 2
    ws = PermanentWorkspace(k, n); toeplitz_weights!(ws, η, n); forward!(ws, n)
    ρ = backward!(ws, n)
    p = vec(sum(ρ[(2k + 1):(n - 2k), :]; dims = 1))
    return p ./ sum(p)
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
# m10. the tempo changes along the stem: what β means (synthetic flowering curves)
#################################################################################################################

# P(organ at position p displaced) under row-wise propensities η(shape, log γ_p): exact edge marginals of the permanent
function displaced_by_row(ptab, shape, logγ)
    k, n = ptab.k, length(logγ)
    ws = PermanentWorkspace(Float64, k, n; logspace = true)
    η = zeros(2k + 1)
    for i in 1:n
        propensity!(η, nothing, ptab, shape, clamp(logγ[i], extrema(ptab.logγ_grid)...))
        ws.w[i, :] .= η
    end
    forward!(ws, n)
    rho = backward!(ws, n)
    return [1 - rho[i, k + 1] for i in 1:n]
end

let n = 55, γbar = 1.7, shape = 1.0
    BCOL = Dict(1.0 => :black, 0.17 => :mediumpurple, 0.0 => :gray50, -0.5 => :teal)
    BLAB = Dict(1.0 => "β = 1: constant noise", 0.17 => "β = 0.17: auxin ramp, α = 1", 0.0 => "β = 0: γ conserved",
        -0.5 => "β = −0.5: noisier where slow")
    # two synthetic flowering curves (piecewise-linear rate, p = (a, log b, log d, μ_τ, log σ_τ)), the median plant
    curves = (("steady", [0.02, log(3.0), log(1.0), log(17.0), log(0.1)], :gray35),
        ("accelerating", [0.25, log(1.6), log(1.0), log(15.0), log(0.1)], C_PATH))
    TPs = Dict(name => plastochron_by_rank(p, n) for (name, p, _) in curves)
    fig = Figure(size = (1600, 900))

    # (a) the plastochron along the stem
    ax = Axis(fig[1, 1]; title = "(a) The tempo changes along the stem", xlabel = "flower rank (bottom → top)",
        ylabel = "plastochron T_P (h)", titlesize = 15)
    for (name, _, c) in curves
        lines!(ax, 1:n, 24 .* TPs[name]; color = c, linewidth = 2.5, label = "$(name) flowering")
    end
    axislegend(ax; framevisible = false, position = :rt)

    # (b, c) one simulated stem with a shortening plastochron, the same delays, two exponents
    m = 24
    r = exp.(range(log(2.0), log(0.6); length = m)); r ./= mean(r)
    R = cumsum(r) # readiness, in mean plastochrons
    smp = PhyllotaxisFold.PassageTime.passage_sampler(TAB.tab, shape)
    rng = Xoshiro(2026)
    z = [smp(rng) for _ in 1:m]
    for (col, β) in ((2, 1.0), (3, 0.0))
        sd = r .^ (1 - β) ./ γbar # delay sd in mean plastochrons
        t = R .+ 2.0 .+ sd .* z   # commitment (mean delay 2 plastochrons, for display)
        σ = sortperm(t)
        moved = falses(m)
        for p in 1:m
            σ[p] != p && (moved[σ[p]] = true)
        end
        ax = Axis(fig[1, col]; title = (β == 1 ? "(b) β = 1: the delay spread stays the same" : "(c) β = 0: the delay spread follows the tempo"),
            xlabel = "time (mean plastochrons)", ylabel = "site j (canonical order)", yticks = 0:4:m, titlesize = 15,
            limits = (nothing, (0, m + 1)))
        for j in 1:m
            c = moved[j] ? C_PATH : :gray40
            mid = R[j] + 2.0
            lines!(ax, [mid - 1.96sd[j], mid + 1.96sd[j]], [j, j]; color = (c, 0.35), linewidth = 6) # 95 % of the delay
            scatter!(ax, [R[j]], [j]; color = :white, strokecolor = :gray30, strokewidth = 1.2, markersize = 9)
            scatter!(ax, [t[j]], [j]; color = c, marker = :star5, markersize = 14)
        end
        text!(ax, 0.02, 0.97; text = "$(count(moved)) of $(m) sites out of order", space = :relative, align = (:left, :top), fontsize = 13)
    end

    # (d) local precision along the stem for several β (accelerating tempo)
    TP = TPs["accelerating"]
    lr = log.(TP ./ mean(TP))
    ax = Axis(fig[2, 1]; title = "(d) Local precision γ_p = γ̄ r_p^β (accelerating tempo, γ̄ = 1.7)", xlabel = "flower rank",
        ylabel = "γ_p", yscale = log10, yticks = [0.5, 1, 2, 4], titlesize = 15)
    for β in (1.0, 0.17, 0.0, -0.5)
        lines!(ax, 1:n, γbar .* exp.(β .* lr); color = BCOL[β], linewidth = 2.5, label = BLAB[β])
    end
    axislegend(ax; framevisible = false, position = :rt, labelsize = 11)

    # (e) what the stem shows: probability that the organ at each position is displaced
    ax = Axis(fig[2, 2]; title = "(e) Predicted fraction of organs displaced", xlabel = "flower rank", ylabel = "P(displaced)",
        limits = (nothing, (0, 0.75)), titlesize = 15)
    for β in (1.0, 0.17, 0.0, -0.5)
        lines!(ax, 1:n, displaced_by_row(TAB.ptab, shape, log(γbar) .+ β .* lr); color = BCOL[β], linewidth = 2.5)
    end
    text!(ax, 0.03, 0.97; text = "the angles see β through this slope", space = :relative, align = (:left, :top), fontsize = 12)

    # (f) the scaling behind the ramp coupling: spread of the ramp delay against the ramp rate
    lν, sd = RTAB.tab.μ_grid, RTAB.tab.sd
    ax = Axis(fig[2, 3]; title = "(f) A steeper auxin ramp commits more precisely", xlabel = "ramp rate in noise units, log ν",
        ylabel = "log sd of the delay (noise time units)", titlesize = 15)
    scatter!(ax, lν, log.(sd); color = C_EARLY, markersize = 9, label = "Fokker–Planck table")
    i0 = findfirst(==(0.0), lν)
    lines!(ax, lν, log(sd[i0]) .- (5 / 6) .* lν; color = :black, linestyle = :dash, linewidth = 1.8, label = "slope −5/6 (weak-noise limit)")
    axislegend(ax; framevisible = false, position = :rt)
    text!(ax, 0.03, 0.05; text = "sd ∝ ν^−0.83 and ν ∝ T_P^−α  ⇒  β = 1 − 0.83 α", space = :relative, align = (:left, :bottom), fontsize = 12)

    save(joinpath(FIG, "m10_tempo_concept.png"), fig; px_per_unit = 1.5)
end
