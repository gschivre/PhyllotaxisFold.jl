# Manual §6.7: does a drifting readiness clock look like delay tails? Simulated data only.
# Orders from τ_i = r_i + c_i, with a clock of i.i.d. intervals r_{i+1} − r_i = 1 + ξ_i, ξ ~ N(0, σξ²) (errors add up
# like a random walk) and Gaussian commitment c_i ~ N(0, σ0²). The delay law is fitted on the TRUE orders (one sample
# per stem, log P_A = 0), so the objective is Σ log P_f(σ_true): no angles, only the question of the timing model.
# Under independent jitter the lag-1 swap rate matches σc² = σ0² + σξ²/2; a random-walk clock has more lag-2 swaps.
# Run in the docs environment with threads (results cached in clock_drift.jls):
#   include("docs/validation_clock_drift.jl"); CD = run_clock_drift(); clock_row(CD[(0.4, 1)])

using PhyllotaxisFold, Random, Statistics, Serialization

const CD_FILE = joinpath(@__DIR__, "clock_drift.jls")

# the seed of each data set (σξ, rep): the values of hash((σξ, rep, 31)) that made the cached results, written out
# because hash values may change between Julia versions (checked 2026-11-01: they regenerate the cached orders)
const CD_SEEDS = Dict(
    (0.0, 0) => 0x7441562a29687cf9,
    (0.0, 1) => 0xa23f63c83e4c7130,
    (0.0, 2) => 0xfa1a17bdac1f6826,
    (0.0, 3) => 0x397532045f60fb00,
    (0.1, 0) => 0x60438682f7482c0d,
    (0.1, 1) => 0x8b774d23deae9f2c,
    (0.1, 2) => 0x7b8c72b544f33812,
    (0.1, 3) => 0xf72ebb0b2acbc793,
    (0.2, 0) => 0xdea0bdcd8289b1aa,
    (0.2, 1) => 0xc03a15d814b83351,
    (0.2, 2) => 0x1d1a5e3db7882e56,
    (0.2, 3) => 0x2bf183c03f145bf6,
    (0.3, 0) => 0xf16518450690d9a6,
    (0.3, 1) => 0x617c9e84e3c9bdee,
    (0.3, 2) => 0x28b122257d6a80ea,
    (0.3, 3) => 0x03278cf0bb511887,
    (0.4, 0) => 0x5cfdf5193a3715c0,
    (0.4, 1) => 0x683a0d98dca83f71,
    (0.4, 2) => 0x51dd26f2ba7b9231,
    (0.4, 3) => 0xf3e929a1e7c80ec6,
)

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
    core = fit_delay_law(smp, GaussianExpTails(:core; σc = σm))
    both = fit_delay_law(smp, GaussianExpTails(:both; σc = σm, πL = 0.02, πE = 0.02))
    inf = delay_law_inference(smp, both)
    matched = relative_loglik(smp, GaussianExpTails(σm)) # i.i.d. law with the lag-1-matched sd
    return (; core, both, inf, matched, disp = displacement_counts(orders), tails = tail_masses(both.law))
end

function run_clock_drift(; σ0 = 0.3, σξs = [0.0, 0.1, 0.2, 0.3, 0.4], reps = 3, S = 60, n = 45, Sbig = 1200)
    res = isfile(CD_FILE) ? deserialize(CD_FILE) : Dict{Any, Any}()
    for σξ in σξs, rep in 0:reps
        key = (σξ, rep) # rep 0: the large data set
        haskey(res, key) && continue
        rng = Xoshiro(get(() -> hash((σξ, rep, 31)), CD_SEEDS, (σξ, rep))) # (a new setting: a seed from its hash)
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
