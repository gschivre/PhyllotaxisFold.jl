# Figures of the manual page "Simulating stems" (docs/src/simulation.md): simulated stems only, no fit, a few seconds.
# Run from the docs environment (CairoMakie):
#   julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make_simulation_figures.jl")'

using PhyllotaxisFold, Random, Statistics
using CairoMakie

const FIG = joinpath(@__DIR__, "src", "figures")


# the stems of the page, with its code (docs/src/simulation.md, "Example")
function simulation_page_stems()
    rng = Xoshiro(2026)
    organs = rand(rng, 30:60, 70)                           # organs per stem
    internodes(n) = [9.0 - 6.0 * (p - 1) / (n - 2) for p in 1:(n - 1)]
    x = [torsion_coordinate(internodes(n)) for n in organs] # stem length (mm) below each organ
    sim = simulate_stems(
        [-1.2, -2.5, -3.1, -4.0];                           # the costs η_1 … η_4
        plants = 70, organs, x,
        α = 0.96,                                           # the spiral: golden in 96 % of the stems
        logμκ = 1.85, σκ = 0.37,                            # the precision
        torsion_mean = 1.0, torsion_sd = 0.5,               # the twist, °/mm
        outlier = 0.005,                                    # outliers
        rng,
    )
    return (; sim, x)
end

# s1: one stem, absolute and divergence angles (as simulated, twist removed, and the sites of its true order)
function fig_simulated_stem(sim, x; s = nothing)
    # a golden stem with a few displaced organs
    s = something(s, findfirst(i -> sim.ψ[i] == Golden_angle && 3 ≤ count(sim.orders[i] .!= eachindex(sim.orders[i])) ≤ 8, eachindex(sim.angles)))
    y, σ, o, ψ, ϕ, ω = sim.angles[s], sim.orders[s], sim.o[s], sim.ψ[s], sim.ϕ[s], sim.torsion[s]
    n = length(y)
    yc = y .- deg2rad(ω) .* x[s] # twist removed
    site = ϕ .+ o .* (σ .- 1) .* ψ # the site each organ belongs to (its true order)
    deg(v) = mod.(rad2deg.(v), 360)
    fig = Figure(size = (1000, 640))
    ttl = "simulated stem: twist $(round(ω; digits = 2)) °/mm, κ = $(round(sim.κ[s]; digits = 1)), ψ = 137.5°, " *
        "$(count(σ .!= 1:n)) displaced organs"
    ax1 = Axis(fig[1, 1]; ylabel = "absolute angle (°)", yticks = 0:100:300, title = ttl, titlesize = 14)
    scatterlines!(ax1, 1:n, deg(y); color = :gray55, linestyle = :dash, marker = :utriangle, markersize = 8, label = "as simulated (twisted)")
    scatterlines!(ax1, 1:n, deg(yc); color = :red, markersize = 7, label = "twist removed")
    scatterlines!(ax1, 1:n, deg(site); color = :green, markersize = 7, label = "site of the true order")
    # the displacement d = σ[p] − p of each displaced organ, above the panel (+1: the organ of the site above)
    disp = [(p, σ[p] - p) for p in 1:n if σ[p] != p]
    text!(
        ax1, first.(disp), fill(372.0, length(disp)); text = [(d > 0 ? "+" : "−") * string(abs(d)) for (_, d) in disp],
        align = (:center, :center), fontsize = 11, color = :green
    )
    ylims!(ax1, 0, 385)
    dv(v) = divergence_angles(v; o)
    ax2 = Axis(fig[2, 1]; xlabel = "organ index (from the base)", ylabel = "divergence angle (°)", yticks = 0:100:300)
    scatterlines!(ax2, 1.5:1:(n - 0.5), dv(y); color = :gray55, linestyle = :dash, marker = :utriangle, markersize = 8)
    scatterlines!(ax2, 1.5:1:(n - 0.5), dv(yc); color = :red, markersize = 7)
    scatterlines!(ax2, 1.5:1:(n - 0.5), dv(site); color = :green, markersize = 7)
    hlines!(ax2, [137.5]; color = :black, linestyle = :dash, label = "canonical angle")
    ylims!(ax2, 0, 360)
    linkxaxes!(ax1, ax2)
    Legend(fig[1:2, 2], ax1; framevisible = true)
    axislegend(ax2; position = :rt, framevisible = true)
    save(joinpath(FIG, "s1_simulated_stem.png"), fig; px_per_unit = 2)
    return fig
end

# s2: divergence angles of all the stems, as simulated and with each stem's twist removed
function fig_divergence_histogram(sim, x)
    divs(f) = reduce(vcat, [divergence_angles(f(s); o = sim.o[s]) for s in eachindex(sim.angles)])
    raw = divs(s -> sim.angles[s])
    cor = divs(s -> sim.angles[s] .- deg2rad(sim.torsion[s]) .* x[s])
    fig = Figure(size = (900, 360))
    ax = Axis(
        fig[1, 1]; xlabel = "divergence angle (°)", ylabel = "density", xticks = 0:45:360,
        title = "$(length(sim.angles)) simulated stems, $(length(raw)) divergence angles", titlesize = 14
    )
    hist!(ax, raw; bins = 0:5:360, normalization = :pdf, color = (:gray55, 0.35), strokecolor = :gray40, strokewidth = 1, label = "as simulated (twisted)")
    hist!(ax, cor; bins = 0:5:360, normalization = :pdf, color = (:red, 0.25), strokecolor = :red, strokewidth = 1, label = "twist removed")
    # the canonical angle ψ, and the divergences of a swap of two neighbours (2ψ, −ψ) and of a displacement by two (3ψ)
    vlines!(ax, [137.5, 275.0, 222.5, 52.5]; color = :black, linestyle = :dash, linewidth = 1)
    xlims!(ax, 0, 360)
    axislegend(ax; position = :rt)
    save(joinpath(FIG, "s2_divergence_histogram.png"), fig; px_per_unit = 2)
    return fig
end

function fig_simulation()
    (; sim, x) = simulation_page_stems()
    fig_simulated_stem(sim, x)
    fig_divergence_histogram(sim, x)
    return nothing
end

fig_simulation()
