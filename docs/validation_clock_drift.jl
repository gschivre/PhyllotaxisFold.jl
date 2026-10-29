# Manual §6.7: does a drifting readiness clock look like delay tails? Simulated data only.
# Orders from τ_i = r_i + c_i, with a clock of i.i.d. intervals r_{i+1} − r_i = 1 + ξ_i, ξ ~ N(0, σξ²) (errors add up
# like a random walk) and Gaussian commitment c_i ~ N(0, σ0²). The delay law is fitted on the TRUE orders (one sample
# per stem, log P_A = 0), so the objective is Σ log P_f(σ_true): no angles, only the question of the timing model.
# Under independent jitter the lag-1 swap rate matches σc² = σ0² + σξ²/2; a random-walk clock has more lag-2 swaps.
# Run in the docs environment with threads (results cached in clock_drift.jls):
#   include("docs/validation_clock_drift.jl"); CD = run_clock_drift(); clock_row(CD[(0.4, 1)])

using PhyllotaxisFold, Random, Statistics, Serialization

const CD_FILE = joinpath(@__DIR__, "clock_drift.jls")

function clock_orders(rng, S, n, σξ, σ0)
    return map(1:S) do _
        r = [0.0; cumsum(1.0 .+ σξ .* randn(rng, n - 1))]
        sortperm(r .+ σ0 .* randn(rng, n))
    end
end

true_samples(orders) = [(; orders = [σ], w = [1.0], logA = [0.0]) for σ in orders]

function displacement_counts(orders)
    d = reduce(vcat, [σ .- (1:length(σ)) for σ in orders])
    return (;
        n = length(d), d1 = count(==(1), abs.(d)) / length(d), d2 = count(==(2), abs.(d)) / length(d),
        d3 = count(>=(3), abs.(d)) / length(d),
    )
end

function fit_clock(orders, σ0, σξ)
    smp = true_samples(orders)
    σm = sqrt(σ0^2 + σξ^2 / 2)
    core = fit_delay_law(smp, :core; init = DelayLaw(σm))
    both = fit_delay_law(smp, :both; init = DelayLaw(σm; πL = 0.02, mL = 1.0, πE = 0.02, mE = 1.0))
    inf = delay_law_inference(smp, both)
    matched = relative_loglik(smp, DelayLaw(σm)) # i.i.d. law with the lag-1-matched sd
    return (; core, both, inf, matched, disp = displacement_counts(orders), tails = tail_masses(both.law))
end

function run_clock_drift(; σ0 = 0.3, σξs = [0.0, 0.1, 0.2, 0.3, 0.4], reps = 3, S = 60, n = 45, Sbig = 1200)
    res = isfile(CD_FILE) ? deserialize(CD_FILE) : Dict{Any, Any}()
    for σξ in σξs, rep in 0:reps
        key = (σξ, rep) # rep 0: the large data set
        haskey(res, key) && continue
        rng = Xoshiro(hash((σξ, rep, 31)))
        orders = clock_orders(rng, rep == 0 ? Sbig : S, n, σξ, σ0)
        res[key] = fit_clock(orders, σ0, σξ)
        serialize(CD_FILE, res)
        isdefined(Main, :KaimonGate) && Main.KaimonGate.progress("σξ = $(σξ), rep = $(rep)")
    end
    return res
end

function clock_row(r)
    b, t = r.both.law, r.tails
    return (;
        σc = round(b.σc; digits = 3), late = round(t.late; digits = 4), early = round(t.early; digits = 4),
        off = round(t.off; digits = 4), se_off = round(r.inf.se[4]; digits = 4),
        gain = round(r.both.rel_ll - r.core.rel_ll; digits = 1), core_σc = round(r.core.law.σc; digits = 3),
        d1 = round(r.disp.d1; digits = 4), d2 = round(r.disp.d2; digits = 5), d3 = round(r.disp.d3; digits = 5),
    )
end
