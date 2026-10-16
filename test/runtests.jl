using PhyllotaxisFold
using Test, Random, Statistics, LinearAlgebra
using Distributions, ForwardDiff, LogExpFunctions
using LogDensityProblems: LogDensityProblems
using Bijectors: Bijectors
using AdditionalDistributions: HalfTDist

const TABLES = joinpath(pkgdir(PhyllotaxisFold), "tables", "tables_k4.jls")
const RAMP_TABLES = joinpath(pkgdir(PhyllotaxisFold), "tables", "tables_ramp_k4.jls")

# all permutations of 1:n with |σ[i] − i| ≤ k
function banded_perms(n, k)
    out = Vector{Vector{Int}}()
    used = falses(n)
    σ = zeros(Int, n)
    function rec(i)
        if i > n
            push!(out, copy(σ))
            return
        end
        for j in max(1, i - k):min(n, i + k)
            used[j] && continue
            used[j] = true; σ[i] = j
            rec(i + 1)
            used[j] = false
        end
    end
    rec(1)
    return out
end

@testset verbose = true "PhyllotaxisFold" begin

@testset "Permanents: forward / backward / Viterbi against enumeration" begin
    rng = Xoshiro(1)
    n, k = 7, 2
    L = randn(rng, n, 2k + 1) # log-weights
    perms = banded_perms(n, k)
    lw(σ) = sum(L[i, σ[i] - i + k + 1] for i in 1:n)
    logZ = logsumexp(lw.(perms))
    for logspace in (true, false)
        ws = PermanentWorkspace(k, n; logspace)
        for i in 1:n, d in -k:k
            ws.w[i, d + k + 1] = L[i, d + k + 1]
        end
        @test forward!(ws, n) ≈ logZ
        rho = backward!(ws, n)
        exact = [sum(exp(lw(σ) - logZ) for σ in perms if σ[i] == i + d; init = 0.0) for i in 1:n, d in -k:k]
        @test rho ≈ exact atol = 1e-12
    end
    @test map_permutation(PermutationGraph(k), L, n) == perms[argmax(lw.(perms))]
    # tridiagonal case (k = 1): the number of banded permutations is a Fibonacci number
    ws1 = PermanentWorkspace(1, 10)
    toeplitz_weights!(ws1, zeros(3), 10)
    @test exp(forward!(ws1, 10)) ≈ 89
    # sampling frequencies
    ws = PermanentWorkspace(k, n)
    ws.w[1:n, :] .= L
    counts = Dict{Vector{Int}, Int}()
    for _ in 1:20_000
        σ = sample_permutation!(rng, ws, n)
        ws.w[1:n, :] .= L
        counts[σ] = get(counts, σ, 0) + 1
    end
    σmax = perms[argmax(lw.(perms))]
    @test counts[σmax] / 20_000 ≈ exp(lw(σmax) - logZ) atol = 0.01
    @test sort(cycle_lengths([2, 1, 3, 5, 6, 4])) == [1, 2, 3]
end

@testset "Permanents: scaled linear space falls back to log-space only when needed" begin
    rng = Xoshiro(3)
    k, n = 4, 30
    η = [-3.0, -2.5, -2.0, -1.0, 0.0, -1.0, -2.0, -2.5, -3.0]
    y = [mod(-(i - 1) * PhyllotaxisFold.Golden_angle + 0.08randn(rng), 2π) for i in 1:n]
    yout = copy(y); yout[12] += π # one outlier
    function setw!(ws, y, κ)
        for d in -k:k, i in max(1, 1 - d):min(n, n - d)
            ws.w[i, d + k + 1] = η[d + k + 1] + κ * (cos(y[i] + (i + d - 1) * PhyllotaxisFold.Golden_angle) - 1)
        end
    end
    function both(y, κ)
        wl, ws = PermanentWorkspace(k, n; logspace = true), PermanentWorkspace(k, n; logspace = false)
        setw!(wl, y, κ); setw!(ws, y, κ)
        return forward!(wl, n), forward!(ws, n), copy(backward!(wl, n)), copy(backward!(ws, n)), ws.nfallback[]
    end
    # moderate κ, and clean angles at very large κ (absorbed by the row maxima): scaled pass, no fallback
    for (yy, κ) in ((y, 30.0), (yout, 30.0), (y, 3e4))
        ll, ls, rl, rs, nf = both(yy, κ)
        @test ls ≈ ll
        @test rs ≈ rl atol = 1e-10
        @test nf == 0
    end
    # an outlier at large κ: its best column belongs to a neighbour and the constrained mass underflows; the call is
    # redone in log-space, with the same value, marginals and dual gradient
    ll, ls, rl, rs, nf = both(yout, 3e3)
    @test isfinite(ls) && ls ≈ ll && rs ≈ rl
    @test nf ≥ 1
    g(ls, κ) = ForwardDiff.derivative(x -> (w = PermanentWorkspace(typeof(x), k, n; logspace = ls); setw!(w, yout, x); forward!(w, n)), κ)
    @test g(false, 3e3) ≈ g(true, 3e3)
end

@testset "PropensityPriors: bijector and prior" begin
    m, v = (0.5, 1.25, 2.0, 3.75), (0.1, 0.75, 2.0, 4.0)
    μΔ, σΔ = η_prior(m, v)
    @test all(exp.(μΔ .+ σΔ .^ 2 ./ 2) .≈ m) # gap means
    @test all((exp.(σΔ .^ 2) .- 1) .* exp.(2 .* μΔ .+ σΔ .^ 2) .≈ v) # gap variances
    for D in (LogNormalSoftPlus, ReverseLogNormalSoftPlus)
        d = D([μΔ...], [σΔ...])
        η = rand(Xoshiro(2), d)
        b = Bijectors.bijector(d)
        θ = b(η)
        @test Bijectors.inverse(b)(θ) ≈ η
        _, lj = Bijectors.with_logabsdet_jacobian(Bijectors.inverse(b), θ)
        @test lj ≈ logabsdet(ForwardDiff.jacobian(Bijectors.inverse(b), θ))[1]
        @test isfinite(logpdf(d, η))
    end
end

if isfile(TABLES)
    tabs = load_tables(TABLES)
    (; tab, ptab) = tabs

    @testset "PassageTime" begin
        d = FoldPassageTime(tab, 2.0; scale = 3.0)
        xs = range(1e-3, quantile(d, 1 - 1e-9); length = 20_001)
        @test sum(pdf.(d, xs)) * step(xs) ≈ 1 atol = 1e-3
        @test cdf(d, quantile(d, 0.3)) ≈ 0.3 atol = 1e-6
        # bicubic reader = table at the grid nodes
        iμ, iz = 9, 300
        @test passage_logg(tab, tab.μ_grid[iμ], tab.z_grid[iz]) ≈ tab.logg[iz, iμ] atol = 1e-9
        # standardised: mean 0, sd 1
        col = exp.(passage_column(tab, 2.0))
        zs = collect(tab.z_grid)
        @test sum(col .* zs) * step(tab.z_grid) ≈ 0 atol = 1e-2
        @test sum(col .* zs .^ 2) * step(tab.z_grid) ≈ 1 atol = 2e-2
    end

    @testset "OrderStatistics" begin
        η = zeros(2ptab.k + 1)
        propensity!(η, nothing, ptab, ptab.μ_grid[10], ptab.logγ_grid[20])
        @test η ≈ ptab.η[:, 10, 20] atol = 1e-12
        # analytic Jacobian of the bicubic interpolation
        ∂η = zeros(2ptab.k + 1, 2)
        propensity!(η, ∂η, ptab, 2.3, log(1.7))
        f(x) = (e = zeros(eltype(x), 2ptab.k + 1); propensity!(e, nothing, ptab, x[1], x[2]); e)
        @test ∂η ≈ ForwardDiff.jacobian(f, [2.3, log(1.7)]) atol = 1e-8
        # the matched η reproduce the target bulk marginals
        p = zero_mean_tilt(ptab.pd[:, 10, 20] ./ sum(ptab.pd[:, 10, 20]), ptab.k)
        ws = PermanentWorkspace(ptab.k, 60)
        toeplitz_weights!(ws, ptab.η[:, 10, 20], 60); forward!(ws, 60)
        rho = backward!(ws, 60)
        m = vec(sum(rho[(2ptab.k + 1):(60 - 2ptab.k), :]; dims = 1)); m ./= sum(m)
        @test m ≈ p atol = 1e-6
    end

    @testset "FoldAngleModel: gradient and Hessian" begin
        sim = simulate_stems(tab, 2.0, 2.0; plants = 6, organs = 25, rng = Xoshiro(3))
        model = FoldModel(sim.angles, ptab; ngrid_ϕ = 8, ngrid_κ = 4)
        u = prior_mean_parameters(model) .+ [0.1, -0.2, 0.1, 0.05, 0.2]
        g = similar(u)
        logdensity_and_gradient!(g, model, u)
        fd = [(logdensity_and_gradient!(nothing, model, u .+ 1e-5 .* (1:5 .== i)) - logdensity_and_gradient!(nothing, model, u .- 1e-5 .* (1:5 .== i))) / 2e-5 for i in 1:5]
        @test g ≈ fd rtol = 1e-4
        l, g2, H = LogDensityProblems.logdensity_gradient_and_hessian(model, u)
        @test g2 ≈ g
        @test issymmetric(H)
    end

    @testset "FoldAngleModel with tempo: reduces to model B, per-row gradient and Hessian" begin
        r = exp.(range(log(1.6), log(0.6); length = 30))
        sim = simulate_stems(tab, 2.0, 1.5; plants = 3, organs = 30, tempo = r, β = 0.3, rng = Xoshiro(3))
        mB = FoldModel(sim.angles, ptab; ngrid_ϕ = 8, ngrid_κ = 4)
        u5 = prior_mean_parameters(mB) .+ [0.1, -0.2, 0.1, 0.05, 0.2]
        # constant tempo: model B exactly, and β drops out of the likelihood
        m1 = FoldModel(sim.angles, ptab; ngrid_ϕ = 8, ngrid_κ = 4, tempo = [ones(30) for _ in 1:3])
        @test LogDensityProblems.dimension(m1) == 6 && has_tempo(m1) && !has_tempo(mB)
        gb, g1 = zeros(5), zeros(6)
        lb = logdensity_and_gradient!(gb, mB, u5)
        l1 = logdensity_and_gradient!(g1, m1, [u5; 0.7])
        @test l1 ≈ lb - 0.5 * 0.2^2 - log(sqrt(2π)) # prior β ~ N(0.5, 1)
        @test g1[1:5] ≈ gb
        @test g1[6] ≈ -0.2 atol = 1e-12
        @test decode_orders(m1, [u5; 0.7]) == decode_orders(mB, u5)
        # a tempo profile: analytic gradient and dual-pass Hessian against finite differences
        m = FoldModel(sim.angles, ptab; ngrid_ϕ = 8, ngrid_κ = 4, tempo = [r for _ in 1:3])
        u = [u5; 0.4]
        l, g, H = LogDensityProblems.logdensity_gradient_and_hessian(m, u)
        f(w) = LogDensityProblems.logdensity(m, w)
        e(i) = 1e-5 .* (1:6 .== i)
        @test l ≈ f(u)
        @test g ≈ [(f(u .+ e(i)) - f(u .- e(i))) / 2e-5 for i in 1:6] rtol = 1e-5
        gβ(w) = LogDensityProblems.logdensity_and_gradient(m, w)[2][6]
        @test H[:, 6] ≈ [(gβ(u .+ e(i)) - gβ(u .- e(i))) / 2e-5 for i in 1:6] rtol = 1e-4
        @test issymmetric(H)
    end

    @testset "FoldAngleModel with tempo: unknown offsets summed out" begin
        TP = [0.6 + 1.2 * exp(-k / 6) for k in 1:60] # plastochron by flower rank: steep at the base
        offs = [0, 2, 1]
        sim = simulate_stems(tab, 2.0, 1.5; plants = 3, organs = 30, tempo = s -> TP[(1:30) .+ offs[s]] ./ mean(TP[1:32]), β = 0.5, rng = Xoshiro(8))
        kw = (; ngrid_ϕ = 8, ngrid_κ = 4)
        u = [prior_mean_parameters(FoldModel(sim.angles, ptab; kw...)) .+ [0.1, -0.2, 0.1, 0.05, 0.2]; 0.4]
        # one known offset = `tempo` with the same relative plastochron
        m0 = FoldModel(sim.angles, ptab; kw..., plastochron = TP, offsets = 1)
        rel = relative_plastochron(TP, [collect(2:31) for _ in 1:3])
        mt = FoldModel(sim.angles, ptab; kw..., tempo = rel.r)
        @test LogDensityProblems.logdensity(m0, u) ≈ LogDensityProblems.logdensity(mt, u)
        # candidates 0:2: each stem's likelihood is the log of the prior-weighted sum over its offsets
        m = FoldModel(sim.angles, ptab; kw..., plastochron = TP, offsets = 0:2)
        @test m.offsets == [[0, 1, 2] for _ in 1:3]
        # reference: one `tempo` model per offset, with the shared normalisation T̄ (prior-weighted mean over the candidates)
        T̄ = sum(sum(TP[(1:30) .+ o]) for o in 0:2) / 90
        lc = [[fold_plant_loglik(FoldModel(sim.angles, ptab; kw..., tempo = [TP[(1:30) .+ o] ./ T̄ for _ in 1:3]), u, s) for o in 0:2] for s in 1:3]
        @test [fold_plant_loglik(m, u, s) for s in 1:3] ≈ [logsumexp(lc[s] .- log(3)) for s in 1:3]
        post = offset_posterior(m, u)
        @test all(p -> sum(p) ≈ 1, post)
        @test post[1] ≈ exp.(lc[1] .- logsumexp(lc[1]))
        # analytic gradient through the sum over offsets
        f(w) = LogDensityProblems.logdensity(m, w)
        e(i) = 1e-5 .* (1:6 .== i)
        g = LogDensityProblems.logdensity_and_gradient(m, u)[2]
        @test g ≈ [(f(u .+ e(i)) - f(u .- e(i))) / 2e-5 for i in 1:6] rtol = 1e-5
        @test length(decode_orders(m, u)) == 3
    end

    @testset "FoldAngleModel with tempo: β recovered on simulated stems" begin
        r = exp.(range(log(1.8), log(0.6); length = 50)); r ./= mean(r)
        for β in (0.0, 1.0)
            sim = simulate_stems(tab, 2.0, 1.6; plants = 60, organs = 50, κ = exp(2.0), tempo = r, β, rng = Xoshiro(10 + round(Int, β)))
            m = FoldModel(sim.angles, ptab; ngrid_ϕ = 16, ngrid_κ = 6, tempo = [r for _ in 1:60])
            fit = fit_map(m)
            se = sqrt(-inv(fit.H)[6, 6])
            @test abs(fit.θ.β - β) < 3 * se
            @test se < 0.3
        end
    end
end

@testset "Plastochron: piecewise-linear rate with per-plant arrest" begin
    p = [0.1, log(2.5), log(0.6), log(14.0), log(0.2)] # rise 2.5 → ≈ 4 flowers / day, decline from day ≈ 14 to arrest
    # closed forms: F is the integral of λ, and λ is 0 after a finite arrest
    τ = 13.0
    @test flowering_cumulative(10.0, p, τ) ≈ sum(flowering_rate(t, p, τ) * 1e-4 for t in 0.5e-4:1e-4:10) rtol = 1e-6
    @test flowering_rate(40.0, p, τ) == 0 && flowering_cumulative(40.0, p, τ) ≈ flowering_cumulative(60.0, p, τ)
    TP = plastochron_by_rank(p, 30; τ)
    function opening_time(y) # F(t) = y by bisection
        lo, hi = 0.0, 40.0
        for _ in 1:80
            m = (lo + hi) / 2
            flowering_cumulative(m, p, τ) < y ? (lo = m) : (hi = m)
        end
        return lo
    end
    @test TP ≈ [1 / flowering_rate(opening_time(k - 0.5), p, τ) for k in 1:30] rtol = 1e-6
    # simulated plants recorded until 3 days without flowers: population parameters and per-plant change points
    rng = Xoshiro(17)
    plants = map(1:30) do _
        τi = exp(p[4] + exp(p[5]) * randn(rng))
        c, acc, z, k = Float64[], 0.0, 0, 0
        while z < 3 && k < 80
            k += 1
            n = rand(rng, Poisson(max(flowering_cumulative(Float64(k), p, τi) - flowering_cumulative(k - 1.0, p, τi), 1e-12)))
            acc += n; push!(c, acc)
            z = (n == 0 && acc > 0 ? z + 1 : 0)
        end
        c[5] = NaN # a day without record
        (; c, τ = τi)
    end
    fit = fit_flowering([pl.c for pl in plants])
    @test fit.converged
    @test all(abs.((fit.p[1:4] .- p[1:4]) ./ fit.se[1:4]) .< 3)
    τhat = [(post = tau_posterior(fit, pl.c); sum(post.w .* post.τ)) for pl in plants]
    @test cor(τhat, [pl.τ for pl in plants]) > 0.85
    # per-plant plastochron over the first 80 % of each plant's flowers
    errs = map(plants) do pl
        n = round(Int, 0.8 * (filter(!isnan, pl.c)[end] - 1))
        maximum(abs, plant_plastochron(fit, pl.c, n) ./ plastochron_by_rank(p, n; τ = pl.τ) .- 1)
    end
    @test median(errs) < 0.12
    # relative plastochron: one curve for all stems, or one per stem
    rel = relative_plastochron(TP, [1:20, 3:25])
    @test mean(reduce(vcat, rel.r)) ≈ 1
    @test rel.r[2][1] ≈ TP[3] / rel.T̄
    rel2 = relative_plastochron([TP, 2 .* TP], [1:10, 1:10])
    @test rel2.r[2] ≈ 2 .* rel2.r[1]
end

@testset "PropensityModel" begin
    data = [rand(Xoshiro(s), VonMises(0.0, 1.0), 20) for s in 1:3]
    model = PhyllotaxisModelAD(data, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4)
    θ = prior_mean(model)
    @test isfinite(LogDensityProblems.logdensity(model, θ))
    lin = PhyllotaxisModelAD(data, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4, logspace = false)
    @test LogDensityProblems.logdensity(lin, θ) ≈ LogDensityProblems.logdensity(model, θ)
    prob = TransformedLogDensityProblem(model)
    u = Bijectors.bijector(model)(θ)
    @test all(isfinite, ForwardDiff.hessian(v -> LogDensityProblems.logdensity(prob, v), u))
    s = sample_prior!(model; n = 60, Nsample = 2, rng = Xoshiro(5)) # n larger than the data (legacy scoping bug)
    @test length(s[1].data) == 60
    dec = decode_sequence!(model, θ, data[1])
    @test isperm(dec.σ)
    # asymmetric variant: the prior of the late side is the mirror image of the early side
    asym = PhyllotaxisModelAD(data, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4, symmetric = false)
    ηa = full_eta(asym, prior_mean(asym))
    @test ηa ≈ reverse(ηa)
end

@testset "Torsion: helpers and per-stem rates in models A and B" begin
    @test torsion_coordinate(4) == [0.0, 1.0, 2.0, 3.0]
    @test torsion_coordinate([2.0, 3.0, 5.0]) == [0.0, 2.0, 5.0, 10.0]
    @test torsion_step([0.0, 2.0, 5.0, 10.0]) ≈ 10 / 3
    @test scan_rate(τ -> -(τ - 0.37)^2, 5.0) ≈ 0.37 atol = 0.005 # scan in degrees per mean step of 5 units
    (; tab, ptab) = load_tables(TABLES)
    sim = simulate_stems(tab, 3.0, 1.5; plants = 8, organs = 40, κ = exp(1.8), torsion_mean = 4.0, torsion_sd = 3.0, rng = Xoshiro(21))
    mB = FoldModel(sim.angles, ptab; ngrid_ϕ = 16, ngrid_κ = 6)
    u = unconstrained_parameters(mB; μ = 3.0, logγ = log(1.5), logμκ = 1.8, σκ = 0.1, α = 0.95)
    l0 = logdensity_and_gradient!(nothing, mB, u)
    τB = copy(update_torsion!(mB, u))
    @test cor(τB, sim.torsion) > 0.95
    @test maximum(abs, τB .- sim.torsion) < 1.5 # degrees per position
    @test logdensity_and_gradient!(nothing, mB, u) > l0 # correcting the twist raises the likelihood
    mA = PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6)
    τA = copy(update_torsion!(mA, prior_mean(mA)))
    @test cor(τA, sim.torsion) > 0.95
end

@testset "Shared angle likelihood: A and B agree, analytic gradient of A" begin
    (; tab, ptab) = load_tables(TABLES)
    sim = simulate_stems(tab, 3.0, 1.5; plants = 3, organs = 30, rng = Xoshiro(3))
    mB = FoldModel(sim.angles, ptab; ngrid_ϕ = 16, ngrid_κ = 6)
    mA = PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6, symmetric = false)
    u = unconstrained_parameters(mB; μ = 3.13, logγ = log(1.5), logμκ = 1.8, σκ = 0.1, α = 0.95)
    η = zeros(9); propensity!(η, nothing, ptab, 3.13, log(1.5))
    θ = vcat(η[1:4], η[6:9], 1.8, 0.1, 0.95)
    @test all(fold_plant_loglik(mB, u, s) ≈ propensity_plant_loglik(mA, θ, s) for s in 1:3) # same likelihood at the same η
    for symmetric in (true, false)
        prob = TransformedLogDensityProblem(PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6, symmetric))
        v = Bijectors.bijector(prob.prob)(prior_mean(prob.prob)) .+ 0.1
        l, g, H = angle_vgh(prob, v, 2)
        @test l ≈ LogDensityProblems.logdensity(prob, v)
        @test g ≈ ForwardDiff.gradient(w -> LogDensityProblems.logdensity(prob, w), v) rtol = 1e-6
        @test H ≈ ForwardDiff.hessian(w -> LogDensityProblems.logdensity(prob, w), v) rtol = 1e-5
    end
end

@testset "Torsion: rates integrated out (adaptive quadrature)" begin
    # quadrature on a bimodal, sharp log-likelihood against a fine grid, for a few population laws
    f(ω) = logaddexp(-0.5 * ((ω - 1.3) / 0.08)^2, -2.0 - 0.5 * ((ω - 0.4) / 0.1)^2)
    ω, logw, modes = stem_nodes(f, 5.0) # coordinate step 5: scan ±5 per unit
    @test length(modes) == 2
    grid = range(-3, 5; length = 200_001)
    for (m, s) in ((1.0, 0.8), (0.0, 2.0), (2.0, 0.5))
        exact = log(sum(exp(f(x) - 0.5 * ((x - m) / s)^2) / (s * sqrt(2π)) for x in grid) * step(grid))
        @test logsumexp(node_logterms(ω, logw, f.(ω), m, s)) ≈ exact atol = 1e-3
    end
    (; tab, ptab) = load_tables(TABLES)
    sim = simulate_stems(tab, 3.0, 1.5; plants = 8, organs = 40, κ = exp(1.8), torsion_mean = 4.0, torsion_sd = 3.0, rng = Xoshiro(21))
    mB = FoldModel(sim.angles, ptab; ngrid_ϕ = 16, ngrid_κ = 6)
    FM = PhyllotaxisFold.FoldAngleModel
    tp = TorsionPrior(m0 = 0.0, sm = 10.0, νs = 4.0, τs = 5.0) # per position
    u = unconstrained_parameters(mB; μ = 3.13, logγ = log(1.5), logμκ = 1.8, σκ = 0.1, α = 0.95)
    nodes = torsion_nodes(mB, u)
    v = [u; 3.5; log(2.5)]
    l0, g, H = torsion_vgh(mB, tp, nodes, v, 2)
    fv(w) = torsion_vgh(mB, tp, nodes, w, 0)[1]
    @test g ≈ [(fv(v .+ 1e-5 .* (1:7 .== i)) - fv(v .- 1e-5 .* (1:7 .== i))) / 2e-5 for i in 1:7] rtol = 1e-4
    @test issymmetric(round.(H; digits = 6)) && l0 ≈ fv(v)
    # fits: population law and rates recovered, models A and B agree
    rB = fit_map_torsion(mB; prior = tp, recenter = 0)
    rA = fit_map_torsion(TransformedLogDensityProblem(PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6));
        prior = tp, recenter = 0)
    conv = PhyllotaxisFold.FoldAngleModel.Optim.converged
    @test conv(rB.fit.result) && conv(rA.fit.result)
    @test abs(rB.m_ω - mean(sim.torsion)) < 1.0 && 0.4 < rB.s_ω / std(sim.torsion) < 2.5
    @test cor(rB.τ, sim.torsion) > 0.95 && cor(rA.τ, sim.torsion) > 0.95
    @test rB.m_ω ≈ rA.m_ω atol = 0.2
    @test all(rB.τ_sd .> 0)
end

end # testset PhyllotaxisFold
