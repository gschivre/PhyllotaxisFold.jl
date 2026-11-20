# Manual §6.2: sites that wander. The angle model puts independent noise on the absolute angle of each organ around its
# site; here the sites themselves follow a random walk, θ_j = θ_{j−1} + ψ + w_j with w_j ~ N(0, s²) (s degrees per
# site), as if each site were placed relative to the previous one. Simulated data only: the stems of §6.1 (60 stems of
# 45 organs, the light-tailed delay law, κ ≈ 6.4, twist 1.0 ± 0.7 °/mm), plus the walk, added through the true order
# (the organ of site j is moved by o w_j). Each data set is fitted and decoded as usual (fit_map with the twist, joint
# MAP), and the displacements decoded are compared with the true ones.
# Run in the docs environment with threads (results cached in site_walk.jls):
#   include("docs/validation_site_walk.jl"); SW = run_site_walk(); site_walk_table(SW)

using PhyllotaxisFold, Random, Statistics, Serialization

const SW_FILE = joinpath(@__DIR__, "site_walk.jls")
const SW_NSTEMS, SW_NORGANS = 60, 45
const SW_LAW = GaussianExpTails(0.42; πL = 0.05, mL = 1.5, πE = 0.054, mE = 1.04) # the light tails of §6.1
const SW_TPRIOR = TorsionPrior(mt = 0.0, st = 2.0, νt = 4.0, τt = 1.0) # °/mm

# the twist coordinates of §6.1: internodes from ≈ 9 mm at the base to ≈ 3 mm, log-normal noise
function sw_coordinates(rng)
    return map(1:SW_NSTEMS) do _
        ℓ = [exp(log(9.0 - 6.0 * (p - 1) / (SW_NORGANS - 2)) + 0.35 * randn(rng)) for p in 1:(SW_NORGANS - 1)]
        [0.0; cumsum(ℓ)]
    end
end

function displacement_fractions(orders)
    d = abs.(reduce(vcat, [σ .- (1:length(σ)) for σ in orders]))
    return (; any = mean(d .> 0), d1 = mean(d .== 1), d2plus = mean(d .≥ 2))
end

function run_walk(s; seed = 61)
    rng = Xoshiro(seed)
    x = sw_coordinates(rng)
    sim = simulate_stems(SW_LAW; plants = SW_NSTEMS, organs = SW_NORGANS, x, logμκ = 1.85, σκ = 0.15, α = 0.95,
        torsion_mean = 1.0, torsion_sd = 0.7, rng)
    angles = map(1:SW_NSTEMS) do i
        w = [0.0; cumsum(deg2rad(s) .* randn(rng, SW_NORGANS - 1))] # the walk of the sites, from site 1
        σ = sim.orders[i]
        [mod2pi(sim.angles[i][p] + sim.o[i] * w[σ[p]]) for p in 1:SW_NORGANS]
    end
    r = fit_map(PhyllotaxisModelAD(angles; torsion = SW_TPRIOR, torsion_x = x))
    dec = decode_orders(r) # the MAP order of each stem
    return (; s, θ = r.θ, truth = displacement_fractions(sim.orders), decoded = displacement_fractions(dec))
end

function run_site_walk(; steps = [0.0, 10.0, 15.0, 20.0])
    res = isfile(SW_FILE) ? deserialize(SW_FILE) : Dict{Any, Any}()
    for s in steps
        haskey(res, s) && continue
        res[s] = run_walk(s) # the same seed for every step: the same stems and orders, only the walk differs
        serialize(SW_FILE, res)
        isdefined(Main, :KaimonGate) && Main.KaimonGate.progress("walk $(s)°")
    end
    return res
end

function site_walk_table(res)
    pc(v) = round(100v; digits = 1)
    return [(s, κ = round(exp(r.θ.logμκ); digits = 1), η = round.(r.θ.η[(end ÷ 2 + 2):end]; digits = 2),
        displaced = (pc(r.decoded.any), pc(r.truth.any)), by_2plus = (pc(r.decoded.d2plus), pc(r.truth.d2plus)))
        for (s, r) in sort(collect(res); by = first)]
end
