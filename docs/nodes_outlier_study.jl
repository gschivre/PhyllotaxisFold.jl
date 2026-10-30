# Two checks on the validation design (simulated stems only; manual §2.6 and §2.4): the cost and effect of 9 twist
# nodes per peak instead of 5, and the outlier option (uniform angles, π_o) on stems with and without outlier angles.
# Run in the docs environment after include("docs/make_validation_figures.jl"), then e.g.
#   run_variant(:heavy, 2; nnodes = 9); run_variant(:light, 1; outlier_sim = 0.02, outlier_fit = true)
#   run_variant(:light, 1; outlier_sim = 0.02, outlier_fit = true, flat_prior = true); summarize.(values(STUDY))
# Results cached in docs/nodes_outlier_study.jls.
const STUDY_FILE = joinpath(@__DIR__, "nodes_outlier_study.jls")
STUDY = isfile(STUDY_FILE) ? deserialize(STUDY_FILE) : Dict{Any, Any}()

function simulate_set(key, rep; outlier = 0.0)
    law = SCENARIOS[findfirst(s -> s[1] == key, SCENARIOS)][3]
    seed = 1000 * rep + findfirst(s -> s[1] == key, SCENARIOS)
    rng = Xoshiro(seed)
    x = stem_coordinates(rng)
    sim = simulate_stems(
        law; plants = NSTEMS, organs = NORGANS, x, KAPPA..., α = 0.95, torsion_mean = TWIST.mean,
        torsion_sd = TWIST.sd, outlier, rng
    )
    return (; law, sim, x, seed)
end

# fit + decode + posterior, timed; `outlier_fit`: the model's outlier option
function run_variant(key, rep; nnodes = 5, outlier_sim = 0.0, outlier_fit = false, flat_prior = false, nsamples = 200)
    tag = flat_prior ? (key, rep, nnodes, outlier_sim, outlier_fit, :flat) : (key, rep, nnodes, outlier_sim, outlier_fit)
    haskey(STUDY, tag) && return STUDY[tag]
    stage("$(tag): fit")
    d = simulate_set(key, rep; outlier = outlier_sim)
    prior = flat_prior ? PhyllotaxisPrior(; aπ = 1.0, bπ = 1.0) : PhyllotaxisPrior() # π_o ~ Beta(1, 1) or the default
    m = PhyllotaxisModelAD(d.sim.angles, prior; torsion_x = d.x, outlier = outlier_fit)
    t_fit = @elapsed r = fit_map_torsion!(m; prior = TPRIOR, nodes_kw = (; nnodes))
    G = length(r.fit.u) - 2
    θ = constrained_parameters(m, r.fit.u[1:G])
    stage("$(tag): decode")
    t_dec = @elapsed dec = decode_orders(m, r; full = true)
    stage("$(tag): posterior")
    t_post = @elapsed post = order_posterior(m, r; nsamples, rng = Xoshiro(d.seed + 1))
    smp = order_samples(post, θ.η)
    stage("$(tag): delay law")
    t_law = @elapsed fb = fit_delay_law(smp, GaussianExpTails(:both))
    inf = delay_law_inference(smp, fb)
    out = (;
        tag, t_fit, t_dec, t_post, t_law, θ, r.m_ω, r.s_ω, r.ω_mean, r.ω_sd, r.loglik,
        truth = (; d.sim.torsion, d.sim.orders, d.sim.o, d.sim.ψ, d.sim.outliers),
        dec = [(; x.σ, x.o, x.ψ) for x in dec], ρ = [p.ρ for p in post], k = m.k,
        est = inf.estimate, se = inf.se, profile = overtaking_profile(smp),
        nnodes_per_stem = length.(r.nodes.ω),
    )
    STUDY[tag] = out
    serialize(STUDY_FILE, STUDY)
    return out
end

function summarize(o)
    tw = o.truth.torsion
    return (;
        tag = o.tag, t = round.((o.t_fit, o.t_dec, o.t_post); digits = 1), m_ω = round(o.m_ω; digits = 3),
        s_ω = round(o.s_ω; digits = 3), logμκ = round(o.θ.logμκ; digits = 3), σκ = round(o.θ.σκ; digits = 3),
        πo = round(o.θ.outlier; digits = 4),
        twist_rmse = round(sqrt(mean(abs2, o.ω_mean .- tw)); digits = 4),
        twist_rmse_trim = round(sqrt(mean(abs2, filter(<(0.3), abs.(o.ω_mean .- tw)))); digits = 4),
        cover95 = round(mean(abs.(o.ω_mean .- tw) .< 1.96 .* o.ω_sd); digits = 3),
        correct = round(mean(mean(d.σ .== σ) for (d, σ) in zip(o.dec, o.truth.orders)); digits = 4),
        off = (round(o.est.off; digits = 3), round(o.se.off; digits = 3)), σc = round(o.est.σc; digits = 3),
        q = round.(o.profile.q[1:3]; digits = 4), loglik = round(o.loglik; digits = 1),
    )
end
