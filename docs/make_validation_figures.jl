# Validation of the whole pipeline on simulated stems (manual §6, docs/src/validation.md): fitting, decoding and the commitment-timing
# law, against the truth used to simulate. Simulated data only.
#
# Design close to the Arabidopsis data: 60 stems of 45 organs, internodes long at the base and shorter higher up (mm),
# κ ≈ 6.4 (log-normal across stems), 95 % golden spirals, a twist of 1.0 ± 0.7 °/mm. Three delay laws (plastochrons),
# the tailed ones taken from fits to a wild type and to a mutant with many off-time organs:
#   no tails     σc = 0.42
#   light tails  σc = 0.42, 5 % late (m 1.5), 5.4 % early (m 1.04)
#   heavy tails  σc = 0.39, 19.6 % late (m 1.09), 26.9 % early (m 1.01)
# three replicates each. Per replicate: the propensity model with the twist integrated (fit_map_torsion!), the MAP
# decoding, 200 exact posterior order samples (cached), then the :core, :late, :both delay laws with robust standard
# errors; per replicate pair, the light and heavy laws with a shared core sd.
#
# Run in the docs environment, with threads (the replicates are cached in validation.jls, and
# run_validation(; refit_timing = true) refits only the delay laws from the cached order samples):
#   julia -t auto --project=docs docs/make_validation_figures.jl

using PhyllotaxisFold, Random, Statistics, Serialization, LinearAlgebra
using CairoMakie

const VFIG = joinpath(@__DIR__, "src", "figures")
const VCACHE = joinpath(@__DIR__, "validation.jls")
const NSTEMS, NORGANS = 60, 45
const TWIST = (mean = 1.0, sd = 0.7) # °/mm
const KAPPA = (logμκ = 1.85, σκ = 0.15)
const TPRIOR = TorsionPrior(mt = 0.0, st = 2.0, νt = 4.0, τt = 1.0) # °/mm
const SCENARIOS = [
    (:none, "no tails", DelayLaw(0.42)),
    (:light, "light tails", DelayLaw(0.42; πL = 0.05, mL = 1.5, πE = 0.054, mE = 1.04)),
    (:heavy, "heavy tails", DelayLaw(0.39; πL = 0.196, mL = 1.09, πE = 0.269, mE = 1.01)),
]
const SCOL = Dict(:none => :gray40, :light => "#2a78d6", :heavy => "#eb6834")
const NREP = 3

# twist coordinate of each stem (mm below each organ): internodes from ≈ 9 mm at the base to ≈ 3 mm, log-normal noise
function stem_coordinates(rng)
    return map(1:NSTEMS) do _
        ℓ = [exp(log(9.0 - 6.0 * (p - 1) / (NORGANS - 2)) + 0.35 * randn(rng)) for p in 1:(NORGANS - 1)]
        [0.0; cumsum(ℓ)]
    end
end

# a progress message per stage when run through KaimonGate (no-op otherwise)
stage(msg) = isdefined(Main, :KaimonGate) && Main.KaimonGate.progress(msg)

function run_replicate(law, seed)
    stage("seed $(seed): simulating and fitting")
    rng = Xoshiro(seed)
    x = stem_coordinates(rng)
    sim = simulate_stems(
        law; plants = NSTEMS, organs = NORGANS, x, KAPPA..., α = 0.95, torsion_mean = TWIST.mean,
        torsion_sd = TWIST.sd, rng
    )
    m = PhyllotaxisModelAD(sim.angles; torsion_x = x)
    r = fit_map_torsion!(m; prior = TPRIOR)
    G = length(r.fit.u) - 2
    θ = constrained_parameters(m, r.fit.u[1:G])
    stage("seed $(seed): decoding")
    dec = decode_orders(m, r; full = true)
    stage("seed $(seed): posterior orders")
    post = order_posterior(m, r; nsamples = 200, rng = Xoshiro(seed + 1))
    smp = order_samples(post, θ.η)
    return (;
        law, sim = (; sim.orders, sim.delays, sim.torsion, sim.κ, sim.ψ, sim.o), x,
        fit = (; θ, r.m_ω, r.s_ω, r.τ_mean, r.τ_sd, r.loglik), dec = [(; d.σ, d.o, d.ψ) for d in dec],
        ρ = [p.ρ for p in post], k = m.k, smp,
    )
end

# the delay laws of one replicate, from its cached order samples
function timing_fits(v)
    smp, law = v.smp, v.law
    fits = Dict{Symbol, Any}()
    fits[:core] = fit_delay_law(smp, :core; init = DelayLaw(0.5))
    fits[:late] = fit_delay_law(smp, :late; init = DelayLaw(fits[:core].law.σc; πL = 0.05, mL = 1.0))
    fits[:both] = fit_delay_law(smp, :both)
    inf = delay_law_inference(smp, fits[:both])
    # effective sample size of the re-weighting at the fitted law (out of the 200 draws)
    lg = delay_logdensity(fits[:both].law)
    ess = map(smp) do s
        lw = [log(s.w[j]) + loglik_order_exact(s.orders[j], lg, DELAY_GRID, 1.0) - s.logA[j] for j in eachindex(s.orders)]
        w = exp.(lw .- maximum(lw))
        sum(w)^2 / sum(w .^ 2 ./ s.w ./ 200)
    end
    # the overtaking profile: of the fitted law (in inf), read from the posterior orders, and in the true orders
    free = overtaking_profile(smp)
    sample = [
        mean(count(invperm(σ)[i + d] < invperm(σ)[i] for i in 1:(length(σ) - d)) for σ in v.sim.orders) /
            mean(length(σ) - d for σ in v.sim.orders) for d in 1:4
    ]
    return (; fits, inf, free, sample, truth_ll = relative_loglik(smp, law), ess)
end

# per replicate: the light- and heavy-tailed sets fitted together, with one core sd (their true cores, 0.42 and 0.39,
# are close but not equal), and the likelihood ratio against separate cores
function shared_fits(res, rep)
    groups = [res[(:light, rep)].smp, res[(:heavy, rep)].smp]
    t = shared_core_test(groups, :both)
    return (;
        test = t, inf = delay_law_inference(groups, t.shared),
        free = compare_profiles(res[(:light, rep)].free, res[(:heavy, rep)].free),
    )
end

function run_validation(; refit_timing = false)
    res = isfile(VCACHE) ? deserialize(VCACHE) : Dict{Any, Any}()
    for rep in 1:NREP, (key, _, law) in SCENARIOS
        if !haskey(res, (key, rep))
            res[(key, rep)] = run_replicate(law, 1000 * rep + findfirst(s -> s[1] == key, SCENARIOS))
            serialize(VCACHE, res)
        end
        if refit_timing || !haskey(res[(key, rep)], :free)
            v = res[(key, rep)]
            res[(key, rep)] = (; v[(:law, :sim, :x, :fit, :dec, :ρ, :k, :smp)]..., timing_fits(v)...)
            serialize(VCACHE, res)
        end
    end
    for rep in 1:NREP
        (refit_timing || !haskey(res, (:shared, rep))) || continue
        res[(:shared, rep)] = shared_fits(res, rep)
        serialize(VCACHE, res)
    end
    return res
end

# the nine replicates (the cache also holds the shared-core fits, under (:shared, rep))
replicates(res) = [res[(key, rep)] for (key, _, _) in SCENARIOS for rep in 1:NREP]

const VTHEME = Theme(
    fontsize = 15, Axis = (
        xgridvisible = false, ygridcolor = (:black, 0.07), topspinevisible = false,
        rightspinevisible = false, titlefont = :bold, titlesize = 15,
    )
)

# the colour code of the three delay laws, shared by the figures
law_legend!(pos) = Legend(
    pos, [MarkerElement(marker = :circle, color = SCOL[k], markersize = 10) for (k, _, _) in SCENARIOS],
    [l for (_, l, _) in SCENARIOS], "delay law"; framevisible = false
)

#################################################################################################################
# v1. fitting: the angle parameters and the per-stem twist
#################################################################################################################
function fig_v1(res)
    return with_theme(VTHEME) do
        fig = Figure(size = (1500, 430))
        R = [(key, res[(key, rep)]) for (key, _, _) in SCENARIOS for rep in 1:NREP]
        ax = Axis(fig[1, 1]; title = "(a) Twist rate of each stem", xlabel = "true twist (°/mm)", ylabel = "estimated (posterior mean)")
        for (key, v) in R
            scatter!(ax, v.sim.torsion, v.fit.τ_mean; color = (SCOL[key], 0.35), markersize = 5)
        end
        lim = extrema(reduce(vcat, [v.sim.torsion for (_, v) in R]))
        lines!(ax, [lim...], [lim...]; color = :black, linestyle = :dash)
        r = cor(reduce(vcat, [v.sim.torsion for (_, v) in R]), reduce(vcat, [v.fit.τ_mean for (_, v) in R]))
        text!(ax, 0.03, 0.97; text = "r = $(round(r; digits = 3))", space = :relative, align = (:left, :top))
        ax = Axis(fig[1, 2]; title = "(b) Twist population", xticks = (1:2, [rich("mean m", subscript("ω")), rich("sd s", subscript("ω"))]), ylabel = "°/mm")
        for (j, (key, v)) in enumerate(R)
            scatter!(ax, [1, 2] .+ 0.04 .* (j - 5), [v.fit.m_ω, v.fit.s_ω]; color = SCOL[key], markersize = 9)
        end
        scatter!(ax, [1, 2], [TWIST.mean, TWIST.sd]; color = :black, marker = :hline, markersize = 40, label = "truth")
        xlims!(ax, 0.5, 2.5)
        ax = Axis(fig[1, 3]; title = "(c) Angular precision", xticks = (1:2, [rich("log μ", subscript("κ")), rich("σ", subscript("κ"))]), ylabel = "value")
        for (j, (key, v)) in enumerate(R)
            scatter!(ax, [1, 2] .+ 0.04 .* (j - 5), [v.fit.θ.logμκ, v.fit.θ.σκ]; color = SCOL[key], markersize = 9)
        end
        scatter!(ax, [1, 2], [KAPPA.logμκ, KAPPA.σκ]; color = :black, marker = :hline, markersize = 40)
        xlims!(ax, 0.5, 2.5)
        ax = Axis(
            fig[1, 4]; title = "(d) Spiral of each stem", xticks = (1:2, ["orientation", "golden / Lucas"]),
            ylabel = "decoded correctly (fraction of stems)", limits = (0.5, 2.5, 0.8, 1.01)
        )
        for (j, (key, v)) in enumerate(R)
            scatter!(
                ax, [1, 2] .+ 0.04 .* (j - 5), [
                    mean(d.o == o for (d, o) in zip(v.dec, v.sim.o)),
                    mean(d.ψ == ψ for (d, ψ) in zip(v.dec, v.sim.ψ)),
                ]; color = SCOL[key], markersize = 9
            )
        end
        law_legend!(fig[1, 5])
        save(joinpath(VFIG, "v1_fitting.png"), fig; px_per_unit = 2)
        fig
    end
end

#################################################################################################################
# v2. decoding: calibration of the posterior marginals, decoded vs true displacements
#################################################################################################################
function fig_v2(res)
    return with_theme(VTHEME) do
        fig = Figure(size = (1650, 450))
        k = first(replicates(res)).k
        # (a) reliability: predicted P(σ[p] = p + d) against the frequency with which it is true
        edges = [0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax = Axis(
            fig[1, 1]; title = "(a) Posterior of the order is calibrated", xlabel = "predicted probability",
            ylabel = "observed frequency", aspect = 1
        )
        lines!(ax, [0, 1], [0, 1]; color = :black, linestyle = :dash, linewidth = 1)
        # no tails left out: almost every ρ is near 0 or 1, so the middle bins hold < 30 pairs
        for (key, _, _) in SCENARIOS
            key == :none && continue
            pr, ob = Float64[], Bool[]
            for rep in 1:NREP
                v = res[(key, rep)]
                for (ρ, σ) in zip(v.ρ, v.sim.orders), p in axes(ρ, 1), d in -k:k
                    q = ρ[p, d + k + 1]
                    q > 0 || continue
                    push!(pr, q); push!(ob, σ[p] == p + d)
                end
            end
            mids, fr, se = Float64[], Float64[], Float64[]
            for b in 1:(length(edges) - 1)
                sel = edges[b] .< pr .≤ edges[b + 1]
                count(sel) < 30 && continue
                push!(mids, mean(pr[sel])); f = mean(ob[sel]); push!(fr, f); push!(se, sqrt(f * (1 - f) / count(sel)))
            end
            rangebars!(ax, mids, fr .- 2se, fr .+ 2se; color = SCOL[key])
            scatterlines!(ax, mids, fr; color = SCOL[key], markersize = 8)
        end
        # (b) decoded (MAP) displacement against the true one, all scenarios pooled
        C = zeros(2k + 1, 2k + 1)
        for v in replicates(res), (d, σ) in zip(v.dec, v.sim.orders), p in eachindex(σ)
            dt, dd = σ[p] - p, d.σ[p] - p
            (abs(dt) ≤ k && abs(dd) ≤ k) && (C[dt + k + 1, dd + k + 1] += 1)
        end
        Cn = C ./ sum(C; dims = 2)
        ax = Axis(
            fig[1, 2]; title = "(b) Decoded vs true displacement", xlabel = "true displacement d", ylabel = "decoded d",
            xticks = -k:k, yticks = -k:k, aspect = 1
        )
        hm = heatmap!(ax, -k:k, -k:k, Cn; colormap = :Blues, colorrange = (0, 1))
        for i in -k:k, j in -k:k
            C[i + k + 1, j + k + 1] > 0 && Cn[i + k + 1, j + k + 1] ≥ 0.01 && text!(
                ax, i, j; text = string(round(Int, 100 * Cn[i + k + 1, j + k + 1])),
                align = (:center, :center), fontsize = 10, color = Cn[i + k + 1, j + k + 1] > 0.5 ? :white : :black
            )
        end
        Colorbar(fig[1, 3], hm; label = "fraction of the true displacement")
        # (c) per stem: fraction of displaced organs, decoded vs true
        ax = Axis(fig[1, 4]; title = "(c) Displaced organs per stem", xlabel = "true fraction", ylabel = "decoded (MAP)", aspect = 1)
        for (key, _, _) in SCENARIOS, rep in 1:NREP
            v = res[(key, rep)]
            scatter!(
                ax, [mean(σ .!= eachindex(σ)) for σ in v.sim.orders], [mean(d.σ .!= eachindex(d.σ)) for d in v.dec];
                color = (SCOL[key], 0.45), markersize = 6
            )
        end
        lines!(ax, [0, 0.8], [0, 0.8]; color = :black, linestyle = :dash)
        acc = mean(mean(d.σ .== σ) for v in replicates(res) for (d, σ) in zip(v.dec, v.sim.orders))
        text!(
            ax, 0.03, 0.97; text = "positions decoded correctly: $(round(100acc; digits = 1)) %", space = :relative,
            align = (:left, :top), fontsize = 12
        )
        law_legend!(fig[1, 5])
        save(joinpath(VFIG, "v2_decoding.png"), fig; px_per_unit = 2)
        fig
    end
end

#################################################################################################################
# v3. commitment timing through the re-weighting: the delay law recovered
#################################################################################################################
function fig_v3(res)
    return with_theme(VTHEME) do
        fig = Figure(size = (1800, 820))
        quantities = [
            (:σc, rich("core sd σ", subscript("c"), " (plastochron)"), f -> f.σc), (:late, "P(e > 1): late by > 1", f -> tail_masses(f).late),
            (:early, "P(e < −1): early by > 1", f -> tail_masses(f).early), (:off, "P(|e| > 1): off-time", f -> tail_masses(f).off),
        ]
        for (c, (q, lab, tf)) in enumerate(quantities)
            ax = Axis(fig[1, c]; title = rich("($('a' + c - 1)) ", lab), xticks = (1:3, [l for (_, l, _) in SCENARIOS]), ylabel = "estimate ± 2 se")
            for (i, (key, _, law)) in enumerate(SCENARIOS)
                t = tf(law)
                lines!(ax, [i - 0.35, i + 0.35], [t, t]; color = :black, linewidth = 2)
                for rep in 1:NREP
                    v = res[(key, rep)]
                    e, se = getfield(v.inf.estimate, q), getfield(v.inf.se, q)
                    xx = i + 0.15 * (rep - 2)
                    rangebars!(ax, [xx], [e - 2se], [e + 2se]; color = SCOL[key], whiskerwidth = 6)
                    scatter!(ax, [xx], [e]; color = SCOL[key], markersize = 10)
                end
            end
        end
        # (e) the fitted delay densities against the truth (log scale), replicate 1
        ax = Axis(
            fig[2, 1:2]; title = "(e) Delay density: truth (black) and fits (replicates)", xlabel = "delay e (plastochrons)",
            ylabel = "density", yscale = log10, limits = (-4, 5, 1.0e-4, 2)
        )
        z = collect(DELAY_GRID)
        sel = -4 .≤ z .≤ 5
        for (key, _, law) in SCENARIOS
            for rep in 1:NREP
                lines!(ax, z[sel], exp.(delay_logdensity(res[(key, rep)].fits[:both].law))[sel]; color = (SCOL[key], 0.6), linewidth = 2)
            end
            lines!(ax, z[sel], exp.(delay_logdensity(law))[sel]; color = :black, linewidth = 1.5, linestyle = :dash)
        end
        # (f) what the tails add, and the optimism of the fit over the truth
        ax = Axis(
            fig[2, 3:4]; title = "(f) Log-likelihood gains (Δℓ)", xticks = (1:3, [l for (_, l, _) in SCENARIOS]),
            ylabel = "Δℓ", yscale = Makie.pseudolog10, yticks = [0, 1, 3, 10, 30, 100, 300]
        )
        for (i, (key, _, _)) in enumerate(SCENARIOS), rep in 1:NREP
            v = res[(key, rep)]
            scatter!(ax, [i - 0.12], [max(v.fits[:both].rel_ll - v.fits[:core].rel_ll, 0.0)]; color = SCOL[key], marker = :circle, markersize = 10)
            scatter!(ax, [i + 0.12], [max(v.fits[:both].rel_ll - v.truth_ll, 0.0)]; color = SCOL[key], marker = :utriangle, markersize = 10)
        end
        axislegend(
            ax, [MarkerElement(marker = :circle, color = :gray30, markersize = 10), MarkerElement(marker = :utriangle, color = :gray30, markersize = 10)],
            ["two tails − core only", "fitted − true law (optimism)"]; position = :lt, framevisible = false, labelsize = 12
        )
        save(joinpath(VFIG, "v3_timing_recovery.png"), fig; px_per_unit = 2)
        fig
    end
end

#################################################################################################################
# v4. the exact order likelihood against Monte Carlo, and the weights of the re-weighting
#################################################################################################################
function fig_v4(res; N = 1_000_000, n = 7)
    return with_theme(VTHEME) do
        fig = Figure(size = (1250, 450))
        ax = Axis(
            fig[1, 1]; title = "(a) Exact P(σ) vs frequency in $(N ÷ 1000)k draws ($(n) sites)", xlabel = "log P(σ), exact recursion",
            ylabel = "log frequency", aspect = 1
        )
        for (key, _, law) in SCENARIOS
            rng = Xoshiro(3)
            cnt = Dict{Vector{Int}, Int}()
            for _ in 1:N
                σ = sortperm((1:n) .+ [rand(rng, law) for _ in 1:n])
                cnt[σ] = get(cnt, σ, 0) + 1
            end
            top = filter(p -> last(p) ≥ 20, collect(cnt))
            scatter!(ax, [loglik_order_exact(σ, law) for (σ, _) in top], [log(c / N) for (_, c) in top]; color = (SCOL[key], 0.6), markersize = 7)
        end
        lines!(ax, [-12, 0], [-12, 0]; color = :black, linestyle = :dash)
        ax = Axis(fig[1, 2]; title = "(b) Effective sample size of the weights", xlabel = "ESS per stem (out of 200 draws)", ylabel = "stems")
        for (key, _, _) in SCENARIOS
            e = reduce(vcat, [res[(key, rep)].ess for rep in 1:NREP])
            hist!(ax, e; bins = 0:10:210, color = (SCOL[key], 0.45), strokewidth = 0)
        end
        law_legend!(fig[1, 3])
        save(joinpath(VFIG, "v4_exact_likelihood_and_weights.png"), fig; px_per_unit = 2)
        fig
    end
end

# a table of the recovered quantities (method §6)
function validation_table(res)
    rows = map([(key, rep) for (key, _, _) in SCENARIOS for rep in 1:NREP]) do (key, rep)
        v = res[(key, rep)]
        t = tail_masses(v.law)
        r3(x) = round(x; digits = 3)
        (;
            key, rep, σc = (v.law.σc, r3(v.inf.estimate.σc), r3(v.inf.se.σc)), late = (r3(t.late), r3(v.inf.estimate.late), r3(v.inf.se.late)),
            early = (r3(t.early), r3(v.inf.estimate.early), r3(v.inf.se.early)), off = (r3(t.off), r3(v.inf.estimate.off), r3(v.inf.se.off)), tails_gain = round(v.fits[:both].rel_ll - v.fits[:core].rel_ll; digits = 1),
            optimism = round(v.fits[:both].rel_ll - v.truth_ll; digits = 1), twist = (r3(v.fit.m_ω), r3(v.fit.s_ω)), logμκ = r3(v.fit.θ.logμκ),
            correct = round(mean(mean(d.σ .== σ) for (d, σ) in zip(v.dec, v.sim.orders)); digits = 3), ess_median = round(median(v.ess); digits = 1),
        )
    end
    return rows
end

# the overtaking profile q_1 … q_3 per replicate: the law's (population), the true orders' (sample), the fitted law's
# and the model-free estimate, with standard errors
function profile_table(res)
    r4(x) = round(x; digits = 4)
    return map([(key, rep) for (key, _, _) in SCENARIOS for rep in 1:NREP]) do (key, rep)
        v = res[(key, rep)]
        (;
            key, rep, law = r4.(overtaking(v.law)[1:3]), sample = r4.(v.sample[1:3]),
            fitted = (r4.(v.inf.profile.q[1:3]), r4.(v.inf.profile.se[1:3])), free = (r4.(v.free.q[1:3]), r4.(v.free.se[1:3])),
        )
    end
end

# the shared-core fits of the light- and heavy-tailed sets, per replicate
function shared_table(res)
    r3(x) = round(x; digits = 3)
    return map(1:NREP) do rep
        v = res[(:shared, rep)]
        e, se = v.inf.estimate, v.inf.se
        (;
            rep, σc = (r3(e[1].σc), r3(se[1].σc)), off_light = (r3(e[1].off), r3(se[1].off)), off_heavy = (r3(e[2].off), r3(se[2].off)),
            Δoff = (r3(v.inf.Δoff[1].estimate), r3(v.inf.Δoff[1].se)), lr = round(v.test.lr; digits = 2), p = round(v.test.p; digits = 3),
            Δq_joint = (round.(v.inf.Δprofile[1].diff[1:3]; digits = 4), round.(v.inf.Δprofile[1].se[1:3]; digits = 4)),
            Δq_free = (round.(v.free.diff; digits = 4), round.(v.free.se; digits = 4), round(v.free.chi2; digits = 1)),
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    res = run_validation()
    fig_v1(res); fig_v2(res); fig_v3(res); fig_v4(res)
end
