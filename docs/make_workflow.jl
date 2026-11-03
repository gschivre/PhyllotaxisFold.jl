# The worked analysis of the manual (docs/src/workflow.md): a wild type and a mutant, simulated, analysed exactly as
# real data would be. The fits are cached in docs/workflow.jls; the figures go to
# docs/src/figures/w*.png. Run in the docs environment, with threads:
#   julia -t auto --project=docs docs/make_workflow.jl

using PhyllotaxisFold, Random, Statistics, Serialization
using Distributions: Poisson
using CairoMakie

const WFIG = joinpath(@__DIR__, "src", "figures")
const WCACHE = joinpath(@__DIR__, "workflow.jls")
const W_STEMS, W_ORGANS = 50, 40
const W_TPRIOR = TorsionPrior(mt = 0.0, st = 2.0, νt = 4.0, τt = 1.0) # °/mm
# the truth, hidden from the analysis: the mutant has more late organs, with the same core
const W_LAWS = (
    wt = GaussianExpTails(0.42; πL = 0.05, mL = 1.5, πE = 0.05, mE = 1.0),
    mutant = GaussianExpTails(0.42; πL = 0.2, mL = 1.5, πE = 0.05, mE = 1.0),
)
const W_COL = (wt = "#2a78d6", mutant = "#eb6834")
# the flowering of the plants (Plastochron): about 2.5 flowers a day at first, rising, then a decline from day ≈ 14 to
# the arrest. p = (a, log b, log d, μ_τ, log σ_τ), the same law for both groups
const W_FLOWERING = [0.1, log(2.5), log(0.6), log(14.0), log(0.2)]

wstage(msg) = isdefined(Main, :KaimonGate) && Main.KaimonGate.progress(msg)

# one group: simulate its stems (internodes 9 → 3 mm), then the analysis of the manual
function workflow_group(law, seed)
    rng = Xoshiro(seed)
    x = map(1:W_STEMS) do _
        ℓ = [exp(log(9.0 - 6.0 * (p - 1) / (W_ORGANS - 2)) + 0.35 * randn(rng)) for p in 1:(W_ORGANS - 1)]
        [0.0; cumsum(ℓ)]
    end
    sim = simulate_stems(
        law; plants = W_STEMS, organs = W_ORGANS, x, logμκ = 1.85, σκ = 0.15, torsion_mean = 1.0, torsion_sd = 0.7, rng
    )
    wstage("seed $(seed): fitting the propensity model")
    m = PhyllotaxisModelAD(sim.angles; torsion = W_TPRIOR, torsion_x = x)
    r = fit_map(m)
    wstage("seed $(seed): decoding")
    dec = decode_orders(r; full = true)
    post = order_posterior(r; nsamples = 200, rng = Xoshiro(seed + 1))
    smp = order_samples(post, r.θ.η)
    wstage("seed $(seed): delay laws")
    fits = (
        core = fit_delay_law(smp, GaussianExpTails(:core)),
        late = fit_delay_law(smp, GaussianExpTails(:late)),
        both = fit_delay_law(smp, GaussianExpTails(:both)),
    )
    inf = delay_law_inference(smp, fits.both)
    prof = overtaking_profile(smp)
    lg = delay_logdensity(fits.both.law)
    ess = map(smp) do s
        lw = [log(s.w[j]) + loglik_order_exact(s.orders[j], lg, DELAY_GRID, 1.0) - s.logA[j] for j in eachindex(s.orders)]
        w = exp.(lw .- maximum(lw))
        sum(w)^2 / sum(w .^ 2 ./ s.w ./ 200)
    end
    return (;
        law, x, sim = (; sim.angles, sim.orders, sim.torsion), r.θ, r.θ.m_ω, r.θ.s_ω, r.loglik,
        dec = [(; d.σ, d.o, d.ψ, d.ω) for d in dec], ρ = [p.ρ for p in post], ω_mean = [p.ω_mean for p in post],
        ω_sd = [p.ω_sd for p in post], smp, fits, inf, prof, ess, truth_ll = relative_loglik(smp, law), k = m.k,
    )
end

# daily cumulative counts of open flowers of each plant of a group, recorded until 3 days without a new flower; every
# fifth plant has a day without record (NaN)
function flowering_counts(seed)
    rng, p = Xoshiro(seed), W_FLOWERING
    return map(1:W_STEMS) do i
        τ = exp(p[4] + exp(p[5]) * randn(rng))
        c, acc, z, day = Float64[], 0.0, 0, 0
        while z < 3 && day < 80
            day += 1
            n = rand(rng, Poisson(max(flowering_cumulative(Float64(day), p, τ) - flowering_cumulative(day - 1.0, p, τ), 1.0e-12)))
            acc += n
            push!(c, acc)
            z = (n == 0 && acc > 0 ? z + 1 : 0)
        end
        i % 5 == 0 && (c[6] = NaN)
        c
    end
end

# delays in hours: the flowering of each group, its mean plastochron over the measured ranks, the core sd in hours
function workflow_hours(res)
    ranks = [1:W_ORGANS for _ in 1:W_STEMS]
    out = map((wt = (res.wt, 21), mutant = (res.mutant, 22))) do (g, seed)
        counts = flowering_counts(seed)
        fit = fit_flowering(counts)
        T = mean_plastochron(fit, ranks)
        (; counts, fit, T, σc_alone = in_hours(g.inf.estimate.σc, g.inf.se.σc, T))
    end
    sh = res.shared_inf
    shared = map(i -> in_hours(sh.estimate[i].σc, sh.se[i].σc, out[i].T), (wt = 1, mutant = 2))
    return (; out..., shared)
end

function workflow_data(; refit = false)
    !refit && isfile(WCACHE) && return deserialize(WCACHE)
    wt = workflow_group(W_LAWS.wt, 11)
    mutant = workflow_group(W_LAWS.mutant, 12)
    wstage("comparing the groups")
    cmp = compare_profiles(wt.prof, mutant.prof)
    sct = shared_core_test([wt.smp, mutant.smp], GaussianExpTails(:both))
    shared_inf = delay_law_inference([wt.smp, mutant.smp], sct.shared)
    res = (; wt, mutant, cmp, sct, shared_inf)
    serialize(WCACHE, res)
    return res
end

# the true overtaking profile of the simulated orders of a group (what a perfect reading of the stems would give)
true_profile(g; d = 1:4) = [
    sum(count(invperm(σ)[i + dd] < invperm(σ)[i] for i in 1:(length(σ) - dd)) for σ in g.sim.orders) /
        sum(length(σ) - dd for σ in g.sim.orders) for dd in d
]

# posterior mass of the organs displaced by k (the edge of the band), per stem
band_edge(g) = [mean(ρ[:, 1] .+ ρ[:, end]) for ρ in g.ρ]

function workflow_figures(res = workflow_data())
    mkpath(WFIG)
    groups = ((:wt, "wild type", res.wt), (:mutant, "mutant", res.mutant))

    # w1. the raw data: divergence angles of all stems, and one stem of each group
    let fig = Figure(size = (1400, 420))
        ax = Axis(
            fig[1, 1]; title = "(a) Divergence angles, all stems", xlabel = "divergence (°)", ylabel = "density",
            xticks = 0:45:360, titlesize = 15
        )
        for (key, lab, g) in groups
            dv = reduce(vcat, [mod.(d.o .* rad2deg.(diff(y)), 360) for (y, d) in zip(g.sim.angles, g.dec)])
            hist!(ax, dv; bins = 0:5:360, normalization = :pdf, color = (W_COL[key], 0.5), label = lab)
        end
        vlines!(ax, [137.5]; color = :gray, linestyle = :dash)
        axislegend(ax; position = :rt, framevisible = false)
        for (c, (key, lab, g)) in enumerate(groups)
            s = argmax(s -> count(g.sim.orders[s] .!= eachindex(g.sim.orders[s])), 1:5) # among the first five stems
            y, o = g.sim.angles[s], g.dec[s].o
            ax2 = Axis(
                fig[1, c + 1]; title = "($("bc"[c])) One $(lab) stem", xlabel = "height p (between p and p + 1)",
                ylabel = "divergence (°)", yticks = 0:45:360, titlesize = 15
            )
            hlines!(ax2, [137.5]; color = :gray, linestyle = :dash)
            scatterlines!(ax2, 1:(length(y) - 1), mod.(o .* rad2deg.(diff(y)), 360); color = W_COL[key], markersize = 8, linewidth = 0.8)
            ylims!(ax2, 0, 360)
        end
        save(joinpath(WFIG, "w1_data.png"), fig)
    end

    # w2. checks of step 1: the twist of each stem, the edge of the band, and the order of one mutant stem
    let fig = Figure(size = (1400, 430))
        ax = Axis(fig[1, 1]; title = "(a) Twist of each stem (± 2 sd)", xlabel = "stems (sorted)", ylabel = "twist (°/mm)", titlesize = 15)
        off = 0
        for (key, lab, g) in groups
            o = sortperm(g.ω_mean)
            xs = off .+ (1:length(o))
            errorbars!(ax, xs, g.ω_mean[o], 2 .* g.ω_sd[o]; color = (W_COL[key], 0.6))
            scatter!(ax, xs, g.ω_mean[o]; color = W_COL[key], markersize = 6, label = lab)
            off += length(o) + 5
        end
        axislegend(ax; position = :lt, framevisible = false)
        ax2 = Axis(
            fig[1, 2]; title = "(b) Organs at the edge of the band |d| = $(res.wt.k)", xlabel = "posterior fraction per stem",
            ylabel = "stems", titlesize = 15
        )
        for (key, lab, g) in groups
            hist!(ax2, band_edge(g); bins = 0:0.0025:0.06, color = (W_COL[key], 0.5), label = lab)
        end
        axislegend(ax2; position = :rt, framevisible = false)
        g = res.mutant
        s = argmax(s -> count(abs.(g.sim.orders[s] .- eachindex(g.sim.orders[s])) .≥ 2), eachindex(g.sim.orders))
        n, k = length(g.sim.orders[s]), g.k
        ax3 = Axis(
            fig[1, 3]; title = "(c) Posterior of the order, one mutant stem", xlabel = "height p", ylabel = "displacement d",
            yticks = -k:k, titlesize = 15
        )
        hm = heatmap!(ax3, 1:n, -k:k, g.ρ[s]; colormap = :Blues, colorrange = (0, 1))
        scatter!(ax3, 1:n, g.dec[s].σ .- (1:n); color = :black, marker = :xcross, markersize = 9, label = "most probable order")
        Colorbar(fig[1, 4], hm; label = "probability")
        axislegend(ax3; position = :rb, framevisible = true, backgroundcolor = (:white, 0.8))
        save(joinpath(WFIG, "w2_step1_checks.png"), fig)
    end

    # w2b. one decoded stem and the posterior of its order
    fig_decoded_stem(res)

    # w3. the overtaking profiles: read from the orders, from the fitted law, and the truth
    let fig = Figure(size = (1100, 430))
        ax = Axis(
            fig[1, 1]; title = "(a) Overtaking profile", xlabel = "distance d (sites)", ylabel = rich("q", subscript("d")), yscale = log10,
            xticks = 1:4, titlesize = 15
        )
        for (j, (key, lab, g)) in enumerate(groups)
            dx = (j - 1.5) * 0.12
            q, se = g.prof.q, g.prof.se
            errorbars!(ax, (1:4) .+ dx, q, min.(2se, 0.95q), 2se; color = W_COL[key])
            scatter!(ax, (1:4) .+ dx, q; color = W_COL[key], markersize = 11, label = "$(lab): read from the orders")
            lines!(ax, (1:4) .+ dx, g.inf.profile.q; color = W_COL[key], linestyle = :dash, label = "$(lab): fitted law")
            scatter!(ax, (1:4) .+ dx, overtaking(g.law); color = :transparent, strokecolor = :black, strokewidth = 1.2, marker = :diamond, markersize = 10)
        end
        scatter!(ax, [NaN], [NaN]; color = :transparent, strokecolor = :black, strokewidth = 1.2, marker = :diamond, label = "truth (the simulated law)")
        axislegend(ax; position = :lb, framevisible = false, labelsize = 11)
        ax2 = Axis(
            fig[1, 2]; title = "(b) Mutant − wild type", xlabel = "distance d (sites)", ylabel = rich("difference in q", subscript("d")),
            xticks = 1:3, titlesize = 15
        )
        c = res.cmp
        hlines!(ax2, [0.0]; color = :gray, linestyle = :dash)
        errorbars!(ax2, 1:3, c.diff, 2 .* c.se; color = :black)
        scatter!(ax2, 1:3, c.diff; color = :black, markersize = 11)
        scatter!(ax2, 1:3, overtaking(res.mutant.law; d = 1:3) .- overtaking(res.wt.law; d = 1:3); color = :transparent, strokecolor = :black, strokewidth = 1.2, marker = :diamond, markersize = 10)
        save(joinpath(WFIG, "w3_profiles.png"), fig)
    end

    # w4. the fitted delay laws: separate and with a shared core, against the truth
    let fig = Figure(size = (1100, 420))
        grid = DELAY_GRID
        sel = -3 .≤ grid .≤ 5
        for (j, (key, lab, g)) in enumerate(groups)
            ax = Axis(
                fig[1, j]; title = "($("ab"[j])) $(lab)", xlabel = "delay e (plastochrons)", ylabel = j == 1 ? "density (log scale)" : "",
                yscale = log10, titlesize = 15
            )
            dens(f) = max.(exp.(delay_logdensity(f)[sel]), 1.0e-6)
            lines!(ax, grid[sel], dens(g.law); color = :black, linestyle = :dash, label = "truth")
            lines!(ax, grid[sel], dens(g.fits.both.law); color = W_COL[key], linewidth = 2.5, label = "fitted alone")
            lines!(ax, grid[sel], dens(res.sct.shared.laws[j]); color = W_COL[key], linewidth = 1.5, linestyle = :dot, label = "shared core")
            ylims!(ax, 1.0e-4, 2)
            axislegend(ax; position = :rt, framevisible = false)
        end
        save(joinpath(WFIG, "w4_delay_laws.png"), fig)
    end

    # w5. delays in hours: the flower counts, and the plastochron by rank
    let fig = Figure(size = (1100, 420)), h = workflow_hours(res)
        fl = h.wt.fit
        ax = Axis(
            fig[1, 1]; title = "(a) Open flowers, six wild-type plants", xlabel = "day", ylabel = "cumulative open flowers",
            titlesize = 15
        )
        cols = Makie.wong_colors()
        for (j, c) in enumerate(h.wt.counts[1:6])
            post = tau_posterior(fl, c)
            τ = exp(sum(post.w .* log.(post.τ))) # the plant's own change point (posterior geometric mean)
            days = findall(!isnan, c)
            scatter!(ax, days, c[days]; color = cols[j], markersize = 7)
            lines!(ax, 0:0.25:length(c), [flowering_cumulative(t, fl.p, τ) for t in 0:0.25:length(c)]; color = cols[j])
        end
        ax2 = Axis(
            fig[1, 2]; title = "(b) Plastochron by flower rank", xlabel = "flower rank k", ylabel = "plastochron (hours)",
            titlesize = 15
        )
        K = 50 # beyond about 55, ranks exceed the expected total of a plant
        vspan!(ax2, [0.5], [W_ORGANS + 0.5]; color = (:gray, 0.15))
        for c in h.wt.counts[1:6]
            lines!(ax2, 1:K, 24 .* plant_plastochron(fl, c, K); color = (W_COL.wt, 0.35), linewidth = 1)
        end
        for (key, lab) in ((:wt, "wild type"), (:mutant, "mutant"))
            lines!(ax2, 1:K, 24 .* plastochron_by_rank(h[key].fit, K); color = W_COL[key], linewidth = 3, label = "$(lab), median plant")
        end
        lines!(ax2, 1:K, 24 .* plastochron_by_rank(W_FLOWERING, K); color = :black, linestyle = :dash, label = "truth, median plant")
        text!(ax2, 2, 0.5; text = "measured organs", fontsize = 12, color = :gray30)
        ylims!(ax2, 0, nothing)
        axislegend(ax2; position = :lt, framevisible = false)
        save(joinpath(WFIG, "w5_hours.png"), fig)
    end
    return
end

# the shuffled blocks of an order: the shortest segments a:b that σ maps onto themselves, other than fixed organs
function shuffled_blocks(σ)
    blocks, a, top = UnitRange{Int}[], 1, 0
    for i in eachindex(σ)
        top = max(top, σ[i])
        if top == i
            (i > a || σ[a] != a) && push!(blocks, a:i)
            a = i + 1
        end
    end
    return blocks
end

# the motif of the divergences around a block, in multiples of the canonical angle α: from the organ below the block to
# the organ above it (an isolated swap reads "2α −α 2α")
function block_motif(σ, b)
    i0, i1 = max(first(b) - 1, 1), min(last(b), length(σ) - 1)
    c = [σ[i + 1] - σ[i] for i in i0:i1]
    return join(alpha_label.(c), " ")
end
alpha_label(c) = (c == 1 ? "α" : c == -1 ? "−α" : "$(c < 0 ? "−" : "")$(abs(c))α")

# chains of touching blocks (each starts right above the previous one): they share the divergences at their junctions,
# so a chain gets one motif, as in Refahi et al. (2011, Fig. 3)
function block_chains(blocks)
    chains = Vector{Vector{UnitRange{Int}}}()
    for b in blocks
        !isempty(chains) && first(b) == last(last(chains[end])) + 1 ? push!(chains[end], b) : push!(chains, [b])
    end
    return chains
end

# a stem that reads well: blocks of at most three organs (one of three at least), and an organ whose displacement is
# uncertain; among those, the one with the most blocks
function readable_stem(g)
    k = g.k
    ok(s) = (bl = shuffled_blocks(g.dec[s].σ);
        !isempty(bl) && maximum(length, bl) == 3 && any(i -> 0.25 < 1 - g.ρ[s][i, k + 1] < 0.75, axes(g.ρ[s], 1)))
    return argmax(s -> ok(s) ? length(shuffled_blocks(g.dec[s].σ)) : -1, eachindex(g.dec))
end

# w2b. the decoding of one stem, drawn like Refahi et al. (2011, Fig. 3), and the posterior of its order
function fig_decoded_stem(res; s = nothing)
    g, k = res.mutant, res.mutant.k
    s = something(s, readable_stem(g))
    d, y, x, ρ = g.dec[s], g.sim.angles[s], g.x[s], g.ρ[s]
    n, σ, ψ = length(y), d.σ, rad2deg(d.ψ)
    yc = y .- deg2rad(d.ω) .* x # the twist removed
    pred = d.o .* (σ .- 1) .* d.ψ # the sites of the decoded order (the offset cancels in the divergences)
    chains = block_chains(shuffled_blocks(σ))
    fig = Figure(size = (1200, 700))
    cs = [3, -2, 1, -4, 4, -1, 2, -3] # the divergences of the short permutations, in multiples of α (increasing angle)
    ax = Axis(
        fig[1, 1]; title = "(a) The decoded order of one mutant stem", ylabel = "divergence angle",
        yticks = ([mod(c * ψ, 360) for c in cs], ["$(alpha_label(c)) = $(round(mod(c * ψ, 360); digits = 1))°" for c in cs]),
        titlesize = 15, xticklabelsvisible = false, ygridvisible = false
    )
    for ch in chains # shaded chain, a dotted line between its blocks
        vspan!(ax, [first(ch[1]) - 0.5], [last(ch[end]) + 0.5]; color = (W_COL.mutant, 0.12))
        length(ch) > 1 && vlines!(ax, [last(b) + 0.5 for b in ch[1:(end - 1)]]; color = (W_COL.mutant, 0.6), linestyle = :dot)
    end
    hlines!(ax, [mod(c * ψ, 360) for c in cs]; color = :gray70, linewidth = 0.8)
    xs = (1:(n - 1)) .+ 0.5 # divergence i lies between organs i and i + 1
    scatterlines!(ax, xs, divergence_angles(yc; o = d.o); color = :black, linestyle = :dot, marker = :rect,
        markersize = 6, label = "measured (twist removed)")
    lines!(ax, xs, divergence_angles(pred; o = d.o); color = W_COL.mutant, linewidth = 2, label = "decoded")
    row = 1 # a label one row lower when its chain starts close to the previous one
    for (j, ch) in enumerate(chains)
        a, b = first(ch[1]), last(ch[end])
        row = (j > 1 && a - last(chains[j - 1][end]) < 5 ? 3 - row : 1)
        text!(ax, (a + b) / 2, (row == 1 ? 372 : 352); text = block_motif(σ, a:b), align = (:center, :center), fontsize = 11)
    end
    ylims!(ax, 0, 385)
    Legend(fig[1, 2], ax; framevisible = false)

    ax2 = Axis(
        fig[2, 1]; title = "(b) Posterior probability that each organ is displaced", xlabel = "organ (height p)",
        ylabel = "probability", titlesize = 15, yticks = 0:0.25:1
    )
    pd(i, r) = sum(ρ[i, k + 1 + dd] for dd in -k:k if abs(dd) in r)
    p1, p2, p3 = [pd(i, 1:1) for i in 1:n], [pd(i, 2:2) for i in 1:n], [pd(i, 3:k) for i in 1:n]
    barplot!(ax2, 1:n, p1 .+ p2 .+ p3; color = (:steelblue, 0.45), label = "|d| = 1")
    barplot!(ax2, 1:n, p2 .+ p3; color = (:darkorange, 0.75), label = "|d| = 2")
    barplot!(ax2, 1:n, p3; color = :firebrick, label = "|d| ≥ 3")
    for i in 1:n # the displacement of the most probable order
        dd = σ[i] - i
        dd != 0 && text!(ax2, i, 1.06; text = (dd > 0 ? "+$dd" : "−$(-dd)"), align = (:center, :center), fontsize = 10)
    end
    truth = findall(g.sim.orders[s] .!= 1:n)
    scatter!(ax2, truth, fill(-0.07, length(truth)); marker = :utriangle, color = :black, markersize = 8,
        label = "truly displaced")
    ylims!(ax2, -0.12, 1.12)
    Legend(fig[2, 2], ax2; framevisible = false)
    linkxaxes!(ax, ax2)
    xlims!(ax2, 0.3, n + 0.7)
    rowsize!(fig.layout, 1, Relative(0.55))
    save(joinpath(WFIG, "w2b_decoded_stem.png"), fig)
    return s
end

if abspath(PROGRAM_FILE) == @__FILE__
    workflow_figures()
end
