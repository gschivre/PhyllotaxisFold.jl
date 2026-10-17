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
    τA = copy(update_torsion!(mA, prior_mean_parameters(mA)))
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
    rA = fit_map_torsion(PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6); prior = tp, recenter = 0)
    conv = PhyllotaxisFold.FoldAngleModel.Optim.converged
    @test conv(rB.fit.result) && conv(rA.fit.result)
    @test abs(rB.m_ω - mean(sim.torsion)) < 1.0 && 0.4 < rB.s_ω / std(sim.torsion) < 2.5
    @test cor(rB.τ, sim.torsion) > 0.95 && cor(rA.τ, sim.torsion) > 0.95
    @test rB.m_ω ≈ rA.m_ω atol = 0.2
    @test all(rB.τ_sd .> 0)
    # decoding with the twist: the joint MAP recovers the orders and the rates, the posterior agrees with the fit
    dec = decode_orders(mB, rB; full = true)
    @test mean(mean(d.σ .== σ) for (d, σ) in zip(dec, sim.orders)) > 0.9
    @test cor([d.τ for d in dec], sim.torsion) > 0.95
    post = order_posterior(mB, rB; nsamples = 400, rng = Xoshiro(4))
    @test [p.τ_mean for p in post] ≈ rB.τ_mean rtol = 1e-6 # same nodes, same population law, same parameters
    @test all(p -> all(sum(p.ρ; dims = 2) .≈ 1), post)
    k = mB.k
    emp = [mean(σ[i] - i == d for σ in post[1].samples) for i in 1:40, d in -k:k]
    @test maximum(abs, emp .- post[1].ρ) < 0.1 # exact samples reproduce the marginals
    # the same calls on model A
    mA = PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6)
    decA = decode_orders(mA, rA)
    @test mean(mean(σ .== τ) for (σ, τ) in zip(decA, sim.orders)) > 0.9
end

@testset "Outlier angles: gradient, Hessian, limit and recovery (models A and B)" begin
    (; tab, ptab) = load_tables(TABLES)
    sim = simulate_stems(tab, 3.0, 1.5; plants = 3, organs = 25, κ = exp(1.8), outlier = 0.05, rng = Xoshiro(3))
    fd(f, u) = [(f(u .+ 1e-5 .* (1:length(u) .== i)) - f(u .- 1e-5 .* (1:length(u) .== i))) / 2e-5 for i in eachindex(u)]
    for m in (FoldModel(sim.angles, ptab; ngrid_ϕ = 8, ngrid_κ = 4, outlier = true),
              PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 8, ngrid_κ = 4, outlier = true, symmetric = false))
        u = prior_mean_parameters(m) .+ 0.1
        l, g, H = angle_vgh(m, u, 2)
        f(w) = angle_vgh(m, w, 0)[1]
        @test g ≈ fd(f, u) rtol = 1e-5
        @test H ≈ ForwardDiff.hessian(f, u) rtol = 1e-4
    end
    # π_o → 0 gives back the model without outliers
    m0 = FoldModel(sim.angles, ptab; ngrid_ϕ = 8, ngrid_κ = 4)
    m1 = FoldModel(sim.angles, ptab; ngrid_ϕ = 8, ngrid_κ = 4, outlier = true)
    u0 = prior_mean_parameters(m0)
    @test angle_loglikelihood(m1, [u0; -40.0]) ≈ angle_loglikelihood(m0, u0) atol = 1e-8
    # recovery of π_o on simulated stems (model A, the law of the data)
    η = [-3.5, -2.5, -1.2, 0.0, -1.2, -2.5, -3.5]
    sA = simulate_stems(η; plants = 40, organs = 40, logμκ = 2.0, σκ = 0.2, outlier = 0.06, rng = Xoshiro(9))
    μΔ, σΔ = η_prior((0.5, 1.25, 2.0), (0.1, 0.75, 2.0))
    mA = PhyllotaxisModelAD(sA.angles, PhyllotaxisPrior(; μΔ, σΔ); ngrid_ϕ = 16, ngrid_κ = 6, outlier = true)
    fA = fit_map(mA)
    se = sqrt(-inv(fA.H)[end, end]) * fA.θ.outlier * (1 - fA.θ.outlier) # delta method on logit π_o
    @test abs(fA.θ.outlier - 0.06) < 3 * se + 0.01
end

@testset "Models A and B share one API" begin
    (; tab, ptab) = load_tables(TABLES)
    sim = simulate_stems(tab, 3.0, 1.5; plants = 3, organs = 30, rng = Xoshiro(3))
    mB = FoldModel(sim.angles, ptab; ngrid_ϕ = 16, ngrid_κ = 6)
    mA = PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 16, ngrid_κ = 6, symmetric = false)
    uB = unconstrained_parameters(mB; μ = 3.13, logγ = log(1.5), logμκ = 1.8, σκ = 0.1, α = 0.95)
    η = zeros(9); propensity!(η, nothing, ptab, 3.13, log(1.5))
    uA = unconstrained_parameters(mA; η, logμκ = 1.8, σκ = 0.1, α = 0.95)
    θA = constrained_parameters(mA, uA)
    @test θA.η ≈ η && θA.σκ ≈ 0.1
    @test [stem_loglik(mA, uA, s) for s in 1:3] ≈ [stem_loglik(mB, uB, s) for s in 1:3]
    @test angle_loglikelihood(mA, uA) ≈ angle_loglikelihood(mB, uB)
    @test decode_orders(mA, uA) == decode_orders(mB, uB)
    prob = TransformedLogDensityProblem(mA)
    @test angle_loglikelihood(prob, uA) ≈ angle_loglikelihood(mA, uA)
    @test LogDensityProblems.dimension(prob) == LogDensityProblems.dimension(mA) == length(uA)
    # spiral of each stem: golden vs Lucas and the orientation, recovered from the angles
    sp = simulate_stems(tab, 3.0, 1.5; plants = 30, organs = 30, κ = exp(1.8), α = 0.6, rng = Xoshiro(12))
    m = FoldModel(sp.angles, ptab; ngrid_ϕ = 16, ngrid_κ = 6)
    post = spiral_posterior(m, unconstrained_parameters(m; μ = 3.0, logγ = log(1.5), logμκ = 1.8, σκ = 0.1, α = 0.6))
    @test mean((p.p_lucas > 0.5) == (ψ == PhyllotaxisFold.Lucas_angle) for (p, ψ) in zip(post, sp.ψ)) ≥ 0.9
    @test mean((p.p_ccw > 0.5) == (o == 1) for (p, o) in zip(post, sp.o)) ≥ 0.9
end

@testset "B-mix: mixture law, 3-D propensity table, model gradient" begin
    # the cached plateau table (margins −1:0.25:10) is the margin quadrature: building one here costs minutes on one thread
    (; tab) = load_tables(TABLES)
    # the mixture law is normalised, standardised by its median and IQR/1.349
    col = mixture_passage_column(tab, 1.5, 1.0)
    z = collect(col.z_grid)
    g = exp.(col.logg[:, 1])
    @test sum(g) * step(col.z_grid) ≈ 1 atol = 2e-3
    Fat(x) = exp(col.logF[searchsortedfirst(z, x), 1])
    @test Fat(0.0) ≈ 0.5 atol = 0.01
    Fz = exp.(col.logF[:, 1])
    zq(p) = z[searchsortedfirst(Fz, p)]
    @test zq(0.75) - zq(0.25) ≈ 1.349 atol = 0.05 # IQR = 1.349 scales (quartiles asymmetric: a skewed law)
    @test exp(col.logS[end ÷ 2, 1]) + exp(col.logF[end ÷ 2, 1]) ≈ 1 atol = 1e-3
    # a heavy late tail: the mixture survival at z = 10 is far above the plateau law's at its mean margin
    @test col.logS[searchsortedfirst(z, 10.0), 1] > passage_column(tab, 1.5; field = :logS)[searchsortedfirst(collect(tab.z_grid), 10.0)] + 5
    # small 3-D table; at s_μ = 0 it is model B's law (up to the scale convention, sd vs IQR/1.349)
    mt = mixture_propensity_table(tab, 3; μ_grid = collect(1.0:0.5:3.0), s_grid = collect(0.0:0.5:1.5),
        logγ_grid = collect(range(log(1.5) - 0.8, log(1.5) + 0.8; length = 9)), J = 40)
    # same readiness spacing T_P = 1.5 IQR-scales = 1.5 scale / sd passage sds
    ptab = propensity_table(tab, 3; logγ_grid = [log(1.5) + log(mt.scale[3, 1] / tab.sd[findfirst(==(2.0), tab.μ_grid)])])
    @test mt.η[:, 3, 1, 5] ≈ ptab.η[:, findfirst(==(2.0), tab.μ_grid), 1] atol = 0.02 # logγ_grid[5] = log(1.5)
    @test exp(mt.logγ_grid[5]) ≈ 1.5 rtol = 1e-12
    @test mt.maxerr < 1e-3
    # KL projection of a heavy-tailed target (the hda9 WT region, μ̄ = 3, s_μ = 2, γ = 2.95): the plain scaling
    # oscillated at err 0.31; the damped scaling + Gauss–Newton polish reaches the least-squares optimum
    ptail = [0.00432, 0.00683, 0.013, 0.0378, 0.612, 0.253, 0.048, 0.00555, 0.000444]
    ttail = zero_mean_tilt(ptail ./ sum(ptail), 4)
    wsT = PermanentWorkspace(4, 60; logspace = false)
    ηT, _, errT = match_propensity(ttail, 4; ws = wsT)
    @test errT < 1e-3
    mT(η) = PhyllotaxisFold.OrderStatistics._bulk_marginals!(zeros(9), wsT, η, 60, 8, 4)
    rT(η) = log.(ttail) .- log.(mT(η))
    JT = reduce(hcat, [(rT(ηT .+ 1e-6 .* (1:9 .== j)) .- rT(ηT)) ./ 1e-6 for j in 1:9])
    @test norm(JT' * rT(ηT)) < 1e-8 # stationary: what is left of r is out of reach at finite n
    # tricubic interpolation: exact at the nodes, analytic Jacobian
    η = zeros(7); J = zeros(7, 3)
    propensity!(η, nothing, mt, 2.0, 0.5, mt.logγ_grid[4])
    @test η ≈ mt.η[:, 3, 2, 4]
    propensity!(η, J, mt, 2.2, 0.7, log(1.3))
    f(x) = (e = zeros(eltype(x), 7); propensity!(e, nothing, mt, x[1], x[2], x[3]); e)
    @test J ≈ ForwardDiff.jacobian(f, [2.2, 0.7, log(1.3)]) atol = 1e-8
    # the mixture flattens the costs of long displacements
    @test mt.η[7, 3, 4, 5] - mt.η[6, 3, 4, 5] > mt.η[7, 3, 1, 5] - mt.η[6, 3, 1, 5] + 0.5
    # B-mix model: s_μ is one more parameter, analytic gradient and dual-pass Hessian (also with tempo and outliers)
    sim = simulate_stems(tab, 2.0, 1.5; plants = 3, organs = 25, sμ = 0.8, rng = Xoshiro(5))
    @test length(sim.orders) == 3 && all(isperm, sim.orders)
    r = exp.(range(log(1.4), log(0.7); length = 25))
    for kw in ((;), (; outlier = true), (; tempo = [r for _ in 1:3]))
        m = FoldModel(sim.angles, mt; ngrid_ϕ = 8, ngrid_κ = 4, kw...)
        @test has_mixture(m)
        u = prior_mean_parameters(m) .+ 0.05
        @test constrained_parameters(m, u).sμ > 0
        l, gr, H = angle_vgh(m, u, 2)
        fv(w) = angle_vgh(m, w, 0)[1]
        @test gr ≈ [(fv(u .+ 1e-5 .* (1:length(u) .== i)) - fv(u .- 1e-5 .* (1:length(u) .== i))) / 2e-5 for i in eachindex(u)] rtol = 1e-5
        @test H ≈ ForwardDiff.hessian(fv, u) rtol = 1e-4
    end
    # the simulated delays carry the mixture's late tail: more organs displaced late by ≥ 2 than with s_μ = 0
    late(orders) = mean(count(i -> σ[i] - i ≤ -2, eachindex(σ)) for σ in orders)
    s0 = simulate_stems(tab, 2.0, 1.5; plants = 300, organs = 40, sμ = 0.0, rng = Xoshiro(6))
    s1 = simulate_stems(tab, 2.0, 1.5; plants = 300, organs = 40, sμ = 1.5, rng = Xoshiro(6))
    @test late(s1.orders) > late(s0.orders)
end

@testset "Level 1: simulator and the statistics that separate it from Level 0" begin
    (; tab) = load_tables(TABLES)
    μ, γ = 1.0, 1.7
    kw = (; plants = 200, organs = 60, rng = Xoshiro(2))
    L0 = simulate_paced(tab, μ, γ; architecture = :level0, kw...)
    L1 = simulate_paced(tab, μ, γ; architecture = :level1, p = 3, q = 5, w = 0.5, kw...)
    @test L1.R > 0 && all(isperm, L1.orders)
    # mean plastochron T̄ = γ in sd units for both architectures
    @test mean(mean(diff(t)) for t in L0.times) ≈ γ rtol = 0.02
    @test mean(mean(diff(t)) for t in L1.times) ≈ γ rtol = 0.05
    # the Level-0 timeline levels off near 2/γ² · γ² = 2 sd² (two independent delays); Level 1 keeps growing
    s0, s1 = timeline_structure(L0.times, γ), timeline_structure(L1.times, γ)
    @test s0[20] < 1.5 * s0[5]
    @test s1[20] > 2 * s1[5]
    # Level-0 intervals: lag-1 autocorrelation negative (a late organ makes a long then a short interval)
    @test interval_autocorrelation(L0.times)[1] < -0.2
    # cascades: runs of displaced positions, a swap is a run of 2
    @test cascade_lengths([[2, 1, 3, 5, 6, 4]]) == [2, 3]
    # Level 0 here is exactly model B's simulator (same law of the orders): same displaced fraction
    sB = simulate_stems(tab, μ, γ; plants = 200, organs = 60, rng = Xoshiro(3))
    @test mean(mean(σ .!= eachindex(σ)) for σ in L0.orders) ≈ mean(mean(σ .!= eachindex(σ)) for σ in sB.orders) atol = 0.03
end

end # testset PhyllotaxisFold
