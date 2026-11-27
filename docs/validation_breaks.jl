# Manual §6.x and the breaks page: spirals that reverse along the stem. Simulated data only: the stems of §6.1 (45
# organs, the light-tailed delay law, κ ≈ 6.4, twist 1.0 ± 0.7 °/mm), as two genotypes of 40 stems each:
#
# - "WT": one spiral per stem;
# - "mutant": each gap between two organs is a reversal of the spiral with probability 2 % (about 0.9 per stem); each
#   new part has the opposite orientation and its own phase, uniform (its sites ϕ_k + o_k (j − 1) ψ).
#
# Both are fitted as usual (fit_map with the twist), decoded, and scanned for orientation breaks (spiral_breaks, the
# default Beta(1, 99) prior on the break rate). The orders are drawn along the whole stem, so permutations may cross
# a break (the break model lets none cross: a check of that approximation).
# Run in the docs environment with threads (results cached in breaks.jls):
#   include("docs/validation_breaks.jl"); VB = run_breaks_validation(); breaks_table(VB)

using PhyllotaxisFold, Random, Statistics, Serialization

const VB_FILE = joinpath(@__DIR__, "breaks.jls")
const VB_NSTEMS, VB_NORGANS = 40, 45
const VB_LAW = GaussianExpTails(0.42; πL = 0.05, mL = 1.5, πE = 0.054, mE = 1.04)
const VB_TPRIOR = TorsionPrior(mt = 0.0, st = 2.0, νt = 4.0, τt = 1.0)

function vb_coordinates(rng)
    return map(1:VB_NSTEMS) do _
        ℓ = [exp(log(9.0 - 6.0 * (p - 1) / (VB_NORGANS - 2)) + 0.35 * randn(rng)) for p in 1:(VB_NORGANS - 1)]
        [0.0; cumsum(ℓ)]
    end
end

# stems with reversals at rate πb per gap: the cuts, the orientation of each site, and the angles
function vb_stems(πb; seed)
    rng = Xoshiro(seed)
    x = vb_coordinates(rng)
    sim = simulate_stems(
        VB_LAW; plants = VB_NSTEMS, organs = VB_NORGANS, x, logμκ = 1.85, σκ = 0.15, α = 0.95,
        torsion_mean = 1.0, torsion_sd = 0.7, rng
    )
    cuts = [[τ for τ in 2:VB_NORGANS if rand(rng) < πb] for _ in 1:VB_NSTEMS]
    angles = map(1:VB_NSTEMS) do s
        o, ψ, ϕ, σ = sim.o[s], sim.ψ[s], sim.ϕ[s], sim.orders[s]
        bounds = [1; cuts[s]; VB_NORGANS + 1]
        θ = zeros(VB_NORGANS)
        for kpart in 1:(length(bounds) - 1)
            ok = o * (isodd(kpart) ? 1 : -1)
            ϕk = (kpart == 1 ? ϕ : 2π * rand(rng))
            for j in bounds[kpart]:(bounds[kpart + 1] - 1)
                θ[j] = ϕk + ok * (j - 1) * ψ
            end
        end
        θold = [ϕ + o * (j - 1) * ψ for j in 1:VB_NORGANS]
        [mod2pi(sim.angles[s][p] + θ[σ[p]] - θold[σ[p]]) for p in 1:VB_NORGANS]
    end
    return (; angles, x, cuts, orders = sim.orders, o = sim.o)
end

function run_breaks_validation(; groups = (("WT", 0.0, 71), ("mutant", 0.02, 72)))
    res = isfile(VB_FILE) ? deserialize(VB_FILE) : Dict{String, Any}()
    for (g, πb, seed) in groups
        haskey(res, g) && continue
        st = vb_stems(πb; seed)
        r = fit_map(PhyllotaxisModelAD(st.angles; torsion = VB_TPRIOR, torsion_x = st.x))
        dec = decode_orders(r)
        br = spiral_breaks(r)
        res[g] = (;
            πb, cuts = st.cuts, orders = st.orders, o = st.o, angles = st.angles, x = st.x, θ = r.θ, dec,
            breaks = br.stems, rate = br.break_rate,
        )
        serialize(VB_FILE, res)
        isdefined(Main, :KaimonGate) && Main.KaimonGate.progress("breaks: $(g) done")
    end
    return res
end

displaced(orders) = mean(abs.(reduce(vcat, [σ .- (1:length(σ)) for σ in orders])) .> 0)

function breaks_table(res)
    return map(["WT", "mutant"]) do g
        r = res[g]
        has = [!isempty(c) for c in r.cuts]
        pb = [1 - b.p_breaks[1] for b in r.breaks]
        # a true cut is found if p_cut summed over ±1 organ around it exceeds 0.5
        found = [sum(r.breaks[s].p_cut[max(τ - 1, 1):min(τ + 1, VB_NORGANS)]) > 0.5 for s in eachindex(r.cuts) for τ in r.cuts[s]]
        (;
            group = g, true_breaks = sum(length, r.cuts), stems_with = count(has),
            detected = count(pb[has] .> 0.5), false_alarms = count(pb[.!has] .> 0.5), cuts_found = (count(found), length(found)),
            rate = (round(r.rate.mean; digits = 4), round.(r.rate.ci95; digits = 4)),
            displaced = (decoded = round(displaced(r.dec); digits = 3), truth = round(displaced(r.orders); digits = 3)),
            κ = round(exp(r.θ.logμκ); digits = 2), η = round.(r.θ.η[6:8]; digits = 2),
        )
    end
end
