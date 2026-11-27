# Figures of the pages "Canonical angles" and "Breaks of the spiral" (docs/src/figures/angles_*.png, breaks_*.png).
# Simulated data only. Run in the docs environment with threads, after docs/validation_breaks.jl:
#   include("docs/make_angles_breaks_figures.jl"); angles_breaks_figures()

using PhyllotaxisFold, Random, Statistics, Serialization, CairoMakie

include(joinpath(@__DIR__, "validation_breaks.jl"))

const AB_FIG = joinpath(@__DIR__, "src", "figures")
const AB_FILE = joinpath(@__DIR__, "angles_example.jls")
const C_GOLD, C_DIST, C_GREY = :steelblue, :darkorange, :gray55

divs(y, o) = [mod(o * rad2deg(y[p + 1] - y[p]), 360.0) for p in 1:(length(y) - 1)]

# the example of the canonical-angles page, as written there
function angles_example()
    isfile(AB_FILE) && return deserialize(AB_FILE)
    angles = (Golden_angle, Lucas_angle, π)
    sim = simulate_stems([-1.2, -2.5, -3.1, -4.0]; plants = 40, organs = 40, logμκ = 1.85, α = (0.6, 0.5), angles,
        rng = Xoshiro(3))
    prior = PhyllotaxisPrior(; angles, aα = (95, 999), bα = (5, 1))
    r = fit_map(PhyllotaxisModelAD(sim.angles, prior))
    post = spiral_posterior(r)
    # the evidence of a 180° stem: its log-likelihood under 180° alone against golden alone
    out = (; sim, θ = r.θ, p_angle = [p.p_angle for p in post], u = r.u)
    serialize(AB_FILE, out)
    return out
end

function fig_angles(ex)
    f = Figure(size = (1000, 360), fontsize = 14)
    ax = Axis(f[1, 1]; title = "(a) divergences, all stems of each angle", xlabel = "divergence (°)", ylabel = "density",
        xticks = [0, 99.5, 137.5, 180, 360])
    for (ψ, c, lab) in ((Golden_angle, C_GOLD, "golden stems"), (Float64(π), C_DIST, "stems at 180°"))
        idx = findall(==(ψ), ex.sim.ψ)
        d = reduce(vcat, [divs(ex.sim.angles[s], ex.sim.o[s]) for s in idx])
        hist!(ax, d; bins = 0:10:360, normalization = :pdf, color = (c, 0.5), strokecolor = c, strokewidth = 1, label = lab)
    end
    xlims!(ax, 0, 360)
    axislegend(ax; position = :rt, framevisible = false)
    ax2 = Axis(f[1, 2]; title = "(b) P(180°) of each stem", xticks = (1:3, ["golden", "Lucas", "180°"]),
        ylabel = "posterior probability of 180°", xlabel = "true canonical angle")
    rng = Xoshiro(1)
    for (i, ψ) in enumerate((Golden_angle, Lucas_angle, Float64(π)))
        idx = findall(==(ψ), ex.sim.ψ)
        v = [ex.p_angle[s][3] for s in idx]
        scatter!(ax2, i .+ 0.2 .* (rand(rng, length(v)) .- 0.5), v; color = (i == 3 ? C_DIST : C_GOLD), markersize = 9)
    end
    ylims!(ax2, -0.05, 1.05)
    save(joinpath(AB_FIG, "angles_f1_distichous.png"), f; px_per_unit = 2)
    return f
end

# a mutant stem with two breaks: its divergences, the posterior orientation of each organ and the cut posterior
function fig_breaks_stem(VB; s = nothing)
    r = VB["mutant"]
    s = something(s, findfirst(c -> length(c) == 2 && all(diff([1; c; VB_NORGANS]) .≥ 8), r.cuts))
    f = Figure(size = (1000, 560), fontsize = 14)
    o = r.o[s]
    d = divs(r.angles[s], o)
    ax = Axis(f[1, 1]; title = "(a) a simulated mutant stem with two reversals", ylabel = "divergence (°)",
        yticks = [0, 137.5, 222.5, 360])
    hlines!(ax, [137.5, 222.5]; color = (:gray, 0.4), linestyle = :dash)
    vlines!(ax, r.cuts[s] .- 0.5; color = :black, linestyle = :dot)
    scatterlines!(ax, 1:length(d), d; color = C_GREY, markercolor = :black, markersize = 6)
    ylims!(ax, 0, 360)
    hidexdecorations!(ax; grid = false)
    ax2 = Axis(f[2, 1]; title = "(b) what spiral_breaks reads", xlabel = "organ position", ylabel = "probability")
    b = r.breaks[s]
    pplus = b.p_part[:, (o == 1 ? 2 : 1)] # P(the organ's part has the stem's first orientation)
    barplot!(ax2, 1:VB_NORGANS, b.p_cut; color = (C_DIST, 0.8), label = "P(break before the organ)")
    lines!(ax2, 1:VB_NORGANS, pplus; color = C_GOLD, linewidth = 2.5, label = "P(first orientation)")
    vlines!(ax2, r.cuts[s] .- 0.5; color = :black, linestyle = :dot)
    ylims!(ax2, -0.02, 1.02)
    axislegend(ax2; position = :rc, framevisible = false)
    linkxaxes!(ax, ax2)
    save(joinpath(AB_FIG, "breaks_f1_stem.png"), f; px_per_unit = 2)
    return f, s
end

# the break rate of each group, and each stem's P(≥ 1 break) against its true number of breaks
function fig_breaks_groups(VB)
    f = Figure(size = (1000, 380), fontsize = 14)
    ax = Axis(f[1, 1]; title = "(a) posterior of the break rate", xlabel = "π_b (breaks per gap)",
        ylabel = "posterior density (scaled to 1)")
    for (g, c) in (("WT", C_GOLD), ("mutant", C_DIST))
        rt = VB[g].rate
        dens = rt.posterior[1:(end - 1)] ./ diff(rt.grid)
        sel = rt.grid[1:(end - 1)] .< 0.08
        lines!(ax, rt.grid[1:(end - 1)][sel], dens[sel] ./ maximum(dens[sel]); color = c, linewidth = 2.5, label = g)
    end
    vlines!(ax, [0.02]; color = :black, linestyle = :dot)
    xlims!(ax, 0, 0.06)
    axislegend(ax; position = :rt, framevisible = false)
    ax2 = Axis(f[1, 2]; title = "(b) each stem", xlabel = "true number of breaks", ylabel = "P(≥ 1 break)")
    rng = Xoshiro(2)
    for (g, c, dxg) in (("WT", C_GOLD, -0.12), ("mutant", C_DIST, 0.12))
        r = VB[g]
        nb = length.(r.cuts)
        pb = [1 - b.p_breaks[1] for b in r.breaks]
        scatter!(ax2, nb .+ dxg .+ 0.08 .* (rand(rng, length(nb)) .- 0.5), pb; color = (c, 0.8), markersize = 9, label = g)
    end
    hlines!(ax2, [0.5]; color = :gray, linestyle = :dash)
    ylims!(ax2, -0.03, 1.03)
    axislegend(ax2; position = :rb, framevisible = false)
    save(joinpath(AB_FIG, "breaks_f2_groups.png"), f; px_per_unit = 2)
    return f
end

function angles_breaks_figures()
    mkpath(AB_FIG)
    ex = angles_example()
    VB = run_breaks_validation()
    return (fig_angles(ex), fig_breaks_stem(VB), fig_breaks_groups(VB))
end
