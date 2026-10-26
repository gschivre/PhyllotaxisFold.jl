using PhyllotaxisFold
using Test, Random, Statistics, LinearAlgebra
using Distributions, ForwardDiff, LogExpFunctions
using LogDensityProblems: LogDensityProblems
using Bijectors: Bijectors
using SpecialFunctions: SpecialFunctions
using Aqua: Aqua
using ExplicitImports: check_no_stale_explicit_imports

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
        return
    end
    rec(1)
    return out
end

# test data: fold-commit stems with independent delays (the model of the analysis) with a twist per position
const LAW = DelayLaw(0.45; πL = 0.05, mL = 1.2, πE = 0.05, mE = 1.0)

@testset verbose = true "PhyllotaxisFold" begin

    @testset "Common: log Iₙ and the Bessel ratio, values and dual derivatives" begin
        # against SpecialFunctions (besselix = e^{-x} Iₙ(x)), on both sides of the switch to the asymptotic series at x = 100
        for n in (0, 1, 3), x in (1.0e-3, 0.5, 4.0, 30.0, 99.0, 101.0, 600.0)
            @test logbesseli(n, x) ≈ log(SpecialFunctions.besselix(n, x)) + x rtol = 1.0e-10 atol = 1.0e-14
        end
        @test logbesseli(0, 0.0) == 0 && logbesseli(2, 0.0) == -Inf
        for x in (0.3, 5.0, 150.0, 2.0e4)
            @test besseliratio(x) ≈ SpecialFunctions.besselix(1, x) / SpecialFunctions.besselix(0, x) rtol = 1.0e-8
            # d log I₀ / dx = I₁ / I₀, and the dual rule of the ratio against a central difference
            @test ForwardDiff.derivative(z -> logbesseli(0, z), x) ≈ besseliratio(x) rtol = 1.0e-8
            # dual rule against a central difference, or against A'(x) ≈ 1/(2x²) + 1/(4x³) where differences lose precision
            ref = x < 1.0e3 ? (besseliratio(x + 1.0e-4 * x) - besseliratio(x - 1.0e-4 * x)) / (2.0e-4 * x) : 1 / (2x^2) + 1 / (4x^3)
            @test ForwardDiff.derivative(besseliratio, x) ≈ ref rtol = 1.0e-5
        end
        @test ForwardDiff.derivative(besseliratio, 0.0) == 0.5
    end


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
            @test rho ≈ exact atol = 1.0e-12
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
        @test logprob_permutation(PermanentWorkspace(k, n), [-1.0, -0.5, 0.0, -0.5, -1.0], collect(1:n)) ≈
            -forward!((w = PermanentWorkspace(k, n); toeplitz_weights!(w, [-1.0, -0.5, 0.0, -0.5, -1.0], n); w), n)
        # a tilt of the costs, η_d → η_d + a d, changes no order probability (the displacements of an order sum to zero):
        # with free early and late costs, this direction would be set by the prior alone
        ηa = [-1.3, -0.4, 0.0, -0.7, -2.0]
        wa = PermanentWorkspace(k, n)
        @test all(logprob_permutation(wa, ηa, σ) ≈ logprob_permutation(wa, ηa .+ 0.8 .* (-k:k), σ) for σ in perms)
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
        for (yy, κ) in ((y, 30.0), (yout, 30.0), (y, 3.0e4))
            ll, ls, rl, rs, nf = both(yy, κ)
            @test ls ≈ ll
            @test rs ≈ rl atol = 1.0e-10
            @test nf == 0
        end
        # an outlier at large κ: its best column belongs to a neighbour and the constrained mass underflows; the call is
        # redone in log-space, with the same value, marginals and dual gradient
        ll, ls, rl, rs, nf = both(yout, 3.0e3)
        @test isfinite(ls) && ls ≈ ll && rs ≈ rl
        @test nf ≥ 1
        g(ls, κ) = ForwardDiff.derivative(x -> (w = PermanentWorkspace(typeof(x), k, n; logspace = ls); setw!(w, yout, x); forward!(w, n)), κ)
        @test g(false, 3.0e3) ≈ g(true, 3.0e3)
    end

    @testset "PropensityPriors: bijector and prior" begin
        m, v = (0.5, 1.25, 2.0, 3.75), (0.1, 0.75, 2.0, 4.0)
        μΔ, σΔ = η_prior(m, v)
        @test all(exp.(μΔ .+ σΔ .^ 2 ./ 2) .≈ m) # gap means
        @test all((exp.(σΔ .^ 2) .- 1) .* exp.(2 .* μΔ .+ σΔ .^ 2) .≈ v) # gap variances
        d = LogNormalSoftPlus([μΔ...], [σΔ...])
        η = rand(Xoshiro(2), d)
        @test issorted(η; rev = true) && η[1] < 0 # η_1 > η_2 > … > η_k, all negative
        b = Bijectors.bijector(d)
        θ = b(η)
        @test Bijectors.inverse(b)(θ) ≈ η
        _, lj = Bijectors.with_logabsdet_jacobian(Bijectors.inverse(b), θ)
        @test lj ≈ logabsdet(ForwardDiff.jacobian(Bijectors.inverse(b), θ))[1]
        @test isfinite(logpdf(d, η))
        @test mean(d) ≈ -cumsum([m...])
        @test mean(rand(Xoshiro(3), d, 200_000); dims = 2) ≈ mean(d) rtol = 0.02
    end

    @testset "Plastochron: piecewise-linear rate with per-plant arrest" begin
        p = [0.1, log(2.5), log(0.6), log(14.0), log(0.2)] # rise 2.5 → ≈ 4 flowers / day, decline from day ≈ 14 to arrest
        # closed forms: F is the integral of λ, and λ is 0 after a finite arrest
        τ = 13.0
        @test flowering_cumulative(10.0, p, τ) ≈ sum(flowering_rate(t, p, τ) * 1.0e-4 for t in 0.5e-4:1.0e-4:10) rtol = 1.0e-6
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
        @test TP ≈ [1 / flowering_rate(opening_time(k - 0.5), p, τ) for k in 1:30] rtol = 1.0e-6
        # simulated plants recorded until 3 days without flowers: population parameters and per-plant change points
        rng = Xoshiro(17)
        plants = map(1:30) do _
            τi = exp(p[4] + exp(p[5]) * randn(rng))
            c, acc, z, k = Float64[], 0.0, 0, 0
            while z < 3 && k < 80
                k += 1
                n = rand(rng, Poisson(max(flowering_cumulative(Float64(k), p, τi) - flowering_cumulative(k - 1.0, p, τi), 1.0e-12)))
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
        # delays in hours: the mean plastochron of the measured organs and its delta-method standard error
        ranks = [1:20, 1:25]
        T = mean_plastochron(fit, ranks)
        Th(q) = 24 * mean(plastochron_by_rank(q, 25)[reduce(vcat, collect.(ranks))])
        @test T.T ≈ Th(fit.p)
        J = [(Th(fit.p .+ 1.0e-6 .* (1:5 .== i)) - Th(fit.p .- 1.0e-6 .* (1:5 .== i))) / 2.0e-6 for i in 1:5]
        @test T.se ≈ sqrt(J' * fit.V * J) rtol = 1.0e-4
        @test fit.se ≈ sqrt.(diag(fit.V))
        h = in_hours(0.4, 0.05, T)
        @test h.estimate ≈ 0.4 * T.T && h.se ≈ hypot(T.T * 0.05, 0.4 * T.se)
    end

    @testset "PropensityModel" begin
        data = [rand(Xoshiro(s), VonMises(0.0, 1.0), 20) for s in 1:3]
        model = PhyllotaxisModelAD(data, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4, logspace = true)
        θ = prior_mean(model)
        @test isfinite(LogDensityProblems.logdensity(model, θ))
        lin = PhyllotaxisModelAD(data, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4, logspace = false)
        @test LogDensityProblems.logdensity(lin, θ) ≈ LogDensityProblems.logdensity(model, θ)
        prob = TransformedLogDensityProblem(model)
        u = Bijectors.bijector(model)(θ)
        @test all(isfinite, ForwardDiff.hessian(v -> LogDensityProblems.logdensity(prob, v), u))
        s = sample_prior(model; n = 60, Nsample = 2, rng = Xoshiro(5)) # n larger than the data
        @test length(s[1].data) == 60 && isperm(s[1].permutation)
        @test_throws ArgumentError PhyllotaxisModelAD([data[1][1:3]]) # band k = 4 longer than the stem
        # realised frequencies: with steep costs nearly every organ is a fixed point; the cycle fractions sum to one
        steep = unconstrained_parameters(model; η = [-12.0, -24.0, -36.0, -48.0], logμκ = 1.5, σκ = 0.2, α = 0.95)
        θs = model.binv(steep)
        fd, _ = estimate_displacement_frequencies(model, [θs]; N = 3, n = 30, rng = Xoshiro(1))
        @test fd[1] > 0.999
        @test sum(fd) ≈ 1 atol = 1.0e-6
        fp, _ = estimate_permutation_frequencies(model, nothing; N = 200, n = 30, rng = Xoshiro(2))
        @test sum(fp) ≈ 1
        @test all(sum(displacement_probability(model, θ); dims = 1) .≈ 1)
        @test sum(displacement_probability(model, θ)[2:end]) ≈ 2 * sum(exp, θ[1:4]) / (1 + 2 * sum(exp, θ[1:4]))
        # symmetric costs: k of them, mirrored in the full vector
        @test LogDensityProblems.dimension(model) == 4 + 3
        ηf = full_eta(model, θ)
        @test ηf ≈ reverse(ηf) && ηf[5] == 0
        # prior mean of σκ: the half-t mean, 0.25 for ν = 4, τ = 0.25; the prior covers its own mean
        @test θ[6] ≈ 0.25
        @test all(prior_CIs(model)[:, 1] .< θ .< prior_CIs(model)[:, 2])
    end

    @testset "simulate_stems(η): η_1..η_k of any length, or the full vector" begin
        maxdisp(r) = maximum(maximum(abs.(σ .- eachindex(σ))) for σ in r.orders)
        for η in ([-1.0, -2.5, -4.0], [-1.0, -2.5, -4.0, -6.0]) # k odd (was read as a full vector with k = 1) and even
            r = simulate_stems(η; plants = 300, organs = 40, rng = Xoshiro(1))
            @test 1 < maxdisp(r) ≤ length(η)
            full = simulate_stems(vcat(reverse(η), 0.0, η); plants = 300, organs = 40, rng = Xoshiro(1))
            @test full.orders == r.orders
        end
    end


    @testset "Torsion: helpers and per-stem rates" begin
        @test torsion_coordinate(4) == [0.0, 1.0, 2.0, 3.0]
        @test torsion_coordinate([2.0, 3.0, 5.0]) == [0.0, 2.0, 5.0, 10.0]
        @test torsion_step([0.0, 2.0, 5.0, 10.0]) ≈ 10 / 3
        @test scan_rate(τ -> -(τ - 0.37)^2, 5.0) ≈ 0.37 atol = 0.005 # scan in degrees per mean step of 5 units
        x, fx = golden_section_max(t -> (; v = -(t - 1.234)^2), 0.0, 3.0; tol = 1.0e-8, score = r -> r.v)
        @test x ≈ 1.234 atol = 1.0e-6
        @test fx.v ≈ -(x - 1.234)^2
        # the twist coordinate of the simulators: explicit positions give the default stems, mm give another twist
        η = [-0.8, -1.9, -2.4, -3.0]
        pos = [collect(0.0:29.0) for _ in 1:3]
        @test simulate_stems(η; plants = 3, organs = 30, torsion_mean = 2.0, rng = Xoshiro(4)).angles ==
            simulate_stems(η; plants = 3, organs = 30, x = pos, torsion_mean = 2.0, rng = Xoshiro(4)).angles
        @test simulate_stems(LAW; plants = 3, organs = 30, torsion_mean = 2.0, rng = Xoshiro(4)).angles ==
            simulate_stems(LAW; plants = 3, organs = 30, x = pos, torsion_mean = 2.0, rng = Xoshiro(4)).angles
        @test_throws ArgumentError simulate_stems(η; plants = 3, organs = 30, x = pos[1:2])
        sim = simulate_stems(LAW; plants = 8, organs = 40, logμκ = 1.8, torsion_mean = 4.0, torsion_sd = 3.0, rng = Xoshiro(21))
        mA = PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6)
        u = prior_mean_parameters(mA)
        l0 = angle_loglikelihood(mA, u)
        τA = copy(update_torsion!(mA, u))
        @test cor(τA, sim.torsion) > 0.95
        @test maximum(abs, τA .- sim.torsion) < 1.5 # degrees per position
        @test angle_loglikelihood(mA, u) > l0 # correcting the twist raises the likelihood
    end

    @testset "Angle likelihood: analytic gradient and Hessian" begin
        sim = simulate_stems(LAW; plants = 3, organs = 30, logμκ = 1.8, rng = Xoshiro(3))
        prob = TransformedLogDensityProblem(PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6))
        v = Bijectors.bijector(prob.prob)(prior_mean(prob.prob)) .+ 0.1
        l, g, H = angle_vgh(prob, v, 2)
        @test l ≈ LogDensityProblems.logdensity(prob, v)
        @test g ≈ ForwardDiff.gradient(w -> LogDensityProblems.logdensity(prob, w), v) rtol = 1.0e-6
        @test H ≈ ForwardDiff.hessian(w -> LogDensityProblems.logdensity(prob, w), v) rtol = 1.0e-5
    end

    @testset "Torsion: rates integrated out (adaptive quadrature); decoding and order posterior" begin
        # quadrature on a bimodal, sharp log-likelihood against a fine grid, for a few population laws
        f(ω) = logaddexp(-0.5 * ((ω - 1.3) / 0.08)^2, -2.0 - 0.5 * ((ω - 0.4) / 0.1)^2)
        ω, logw, modes = stem_nodes(f, 5.0) # coordinate step 5: scan ±5 per unit
        @test length(modes) == 2
        grid = range(-3, 5; length = 200_001)
        for (m, s) in ((1.0, 0.8), (0.0, 2.0), (2.0, 0.5))
            exact = log(sum(exp(f(x) - 0.5 * ((x - m) / s)^2) / (s * sqrt(2π)) for x in grid) * step(grid))
            @test logsumexp(node_logterms(ω, logw, f.(ω), m, s)) ≈ exact atol = 1.0e-3
        end
        sim = simulate_stems(LAW; plants = 8, organs = 40, logμκ = 1.8, torsion_mean = 4.0, torsion_sd = 3.0, rng = Xoshiro(21))
        m = PhyllotaxisModelAD(sim.angles, PhyllotaxisPrior(); ngrid_ϕ = 16, ngrid_κ = 6)
        tp = TorsionPrior(m0 = 0.0, sm = 10.0, νs = 4.0, τs = 5.0) # per position
        u = prior_mean_parameters(m)
        nodes = torsion_nodes(m, u)
        v = [u; 3.5; log(2.5)]
        P = length(v)
        l0, g, H = torsion_vgh(m, tp, nodes, v, 2)
        fv(w) = torsion_vgh(m, tp, nodes, w, 0)[1]
        @test g ≈ [(fv(v .+ 1.0e-5 .* (1:P .== i)) - fv(v .- 1.0e-5 .* (1:P .== i))) / 2.0e-5 for i in 1:P] rtol = 1.0e-4
        @test issymmetric(round.(H; digits = 6)) && l0 ≈ fv(v)
        # fit: population law and rates recovered
        r = fit_map_torsion!(m; prior = tp, recenter = 0)
        @test PhyllotaxisFold.PropensityModel.Optim.converged(r.fit.result)
        @test abs(r.m_ω - mean(sim.torsion)) < 1.0 && 0.4 < r.s_ω / std(sim.torsion) < 2.5
        @test cor(r.τ, sim.torsion) > 0.95 && all(r.τ_sd .> 0)
        # decoding with the twist: the joint MAP recovers the orders and the rates, the posterior agrees with the fit
        dec = decode_orders(m, r; full = true)
        @test mean(mean(d.σ .== σ) for (d, σ) in zip(dec, sim.orders)) > 0.9
        @test cor([d.τ for d in dec], sim.torsion) > 0.95
        post = order_posterior(m, r; nsamples = 400, rng = Xoshiro(4))
        @test [p.τ_mean for p in post] ≈ r.τ_mean rtol = 1.0e-6 # same nodes, same population law, same parameters
        @test all(p -> all(sum(p.ρ; dims = 2) .≈ 1), post)
        k = m.k
        emp = [mean(σ[i] - i == d for σ in post[1].samples) for i in 1:40, d in -k:k]
        @test maximum(abs, emp .- post[1].ρ) < 0.1 # exact samples reproduce the marginals
    end

    @testset "Outlier angles: gradient, Hessian, limit and recovery" begin
        sim = simulate_stems(LAW; plants = 3, organs = 25, logμκ = 1.8, outlier = 0.05, rng = Xoshiro(3))
        fd(f, u) = [(f(u .+ 1.0e-5 .* (1:length(u) .== i)) - f(u .- 1.0e-5 .* (1:length(u) .== i))) / 2.0e-5 for i in eachindex(u)]
        m = PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 8, ngrid_κ = 4, outlier = true)
        u = prior_mean_parameters(m) .+ 0.1
        l, g, H = angle_vgh(m, u, 2)
        f(w) = angle_vgh(m, w, 0)[1]
        @test g ≈ fd(f, u) rtol = 1.0e-5
        @test H ≈ ForwardDiff.hessian(f, u) rtol = 1.0e-4
        # π_o → 0 gives back the model without outliers
        m0 = PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 8, ngrid_κ = 4)
        m1 = PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 8, ngrid_κ = 4, outlier = true)
        u0 = prior_mean_parameters(m0)
        @test angle_loglikelihood(m1, [u0; -40.0]) ≈ angle_loglikelihood(m0, u0) atol = 1.0e-8
        # recovery of π_o on simulated stems (orders from the banded permanent, the law of the model)
        η = [-3.5, -2.5, -1.2, 0.0, -1.2, -2.5, -3.5]
        sA = simulate_stems(η; plants = 40, organs = 40, logμκ = 2.0, σκ = 0.2, outlier = 0.06, rng = Xoshiro(9))
        μΔ, σΔ = η_prior((0.5, 1.25, 2.0), (0.1, 0.75, 2.0))
        mA = PhyllotaxisModelAD(sA.angles, PhyllotaxisPrior(; μΔ, σΔ); ngrid_ϕ = 16, ngrid_κ = 6, outlier = true)
        fA = fit_map(mA)
        se = sqrt(-inv(fA.H)[end, end]) * fA.θ.outlier * (1 - fA.θ.outlier) # delta method on logit π_o
        @test abs(fA.θ.outlier - 0.06) < 3 * se + 0.01
    end

    @testset "Propensity model API: parameters, likelihood, spiral of each stem" begin
        sim = simulate_stems(LAW; plants = 3, organs = 30, logμκ = 1.8, rng = Xoshiro(3))
        mA = PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 16, ngrid_κ = 6)
        η = [-3.0, -2.4, -1.9, -0.8, 0.0, -0.8, -1.9, -2.4, -3.0]
        uA = unconstrained_parameters(mA; η, logμκ = 1.8, σκ = 0.1, α = 0.95)
        θA = constrained_parameters(mA, uA)
        @test θA.η ≈ η && θA.σκ ≈ 0.1
        @test unconstrained_parameters(mA; η = η[6:end], logμκ = 1.8, σκ = 0.1, α = 0.95) ≈ uA
        @test_throws ArgumentError unconstrained_parameters(
            mA; η = [-3.0, -2.4, -1.9, -0.8, 0.0, -0.9, -1.8, -2.6, -3.1],
            logμκ = 1.8, σκ = 0.1, α = 0.95
        ) # an asymmetric full vector
        @test_throws ArgumentError unconstrained_parameters(mA; η = η[6:8], logμκ = 1.8, σκ = 0.1, α = 0.95)
        @test sum(stem_loglik(mA, uA, s) for s in 1:3) ≈ angle_loglikelihood(mA, uA)
        prob = TransformedLogDensityProblem(mA)
        @test angle_loglikelihood(prob, uA) ≈ angle_loglikelihood(mA, uA)
        @test LogDensityProblems.dimension(prob) == LogDensityProblems.dimension(mA) == length(uA)
        @test decode_orders(prob, uA) == decode_orders(mA, uA)
        # spiral of each stem: golden vs Lucas and the orientation, recovered from the angles
        sp = simulate_stems(LAW; plants = 30, organs = 30, logμκ = 1.8, α = 0.6, rng = Xoshiro(12))
        m = PhyllotaxisModelAD(sp.angles; ngrid_ϕ = 16, ngrid_κ = 6)
        post = spiral_posterior(m, unconstrained_parameters(m; η = [-0.8, -1.9, -2.4, -3.0], logμκ = 1.8, σκ = 0.1, α = 0.6))
        @test mean((p.p_lucas > 0.5) == (ψ == PhyllotaxisFold.Lucas_angle) for (p, ψ) in zip(post, sp.ψ)) ≥ 0.9
        @test mean((p.p_ccw > 0.5) == (o == 1) for (p, o) in zip(post, sp.o)) ≥ 0.9
    end

    @testset "CommitmentTiming: delay laws" begin
        rng = Xoshiro(31)
        zg = DELAY_GRID
        @test rand(DelayLaw(0.4)) isa Float64 # default RNG
        for f in (DelayLaw(0.4), LAW, DelayLaw(0.3; πL = 0.2, mL = 0.8, πE = 0.3, mE = 1.5))
            g = exp.(delay_logdensity(f)) .* step(zg)
            @test sum(g) ≈ 1
            e = [rand(rng, f) for _ in 1:400_000]
            @test sum(g .* zg) ≈ mean(e) atol = 0.01 # the density on the grid is the law of the sampler
            @test sqrt(sum(g .* zg .^ 2) - sum(g .* zg)^2) ≈ std(e) rtol = 0.01
            t = tail_masses(f)
            @test t.late ≈ mean(e .> 1) atol = 0.003
            @test t.early ≈ mean(e .< -1) atol = 0.003
            @test t.off ≈ t.late + t.early
            @test tail_masses(f; a = 2).late ≈ mean(e .> 2) atol = 0.003
        end
        # 231: the organ of the lowest site sits two positions late; 312: the highest site's organ two positions early
        @test order_diagnostics([[1, 2, 3, 4], [2, 3, 1, 4], [3, 1, 2, 4]]) == (; displaced = 0.5, n231 = 1, n312 = 1)
    end

    @testset "CommitmentTiming: exact order likelihood" begin
        s, zg = 0.43, DELAY_GRID
        lg = [-(z / s)^2 / 2 - log(s * sqrt(2π)) for z in zg]
        # two sites: P(in order) = Φ(1 / (s √2))
        @test exp(loglik_order_exact([1, 2], lg, zg, 1.0)) ≈ cdf(Normal(), 1 / (s * sqrt(2))) atol = 5.0e-4
        # the probabilities of all orders of 4 sites sum to 1 (ties counted as ordered made them sum above 1), with tails too
        perms = [[a, b, c, d] for a in 1:4 for b in 1:4 for c in 1:4 for d in 1:4 if allunique((a, b, c, d))]
        @test sum(exp(loglik_order_exact(σ, lg, zg, 1.0)) for σ in perms) ≈ 1 atol = 2.0e-3
        f = DelayLaw(0.3; πL = 0.2, mL = 0.8, πE = 0.3, mE = 1.5)
        @test sum(exp(loglik_order_exact(σ, f)) for σ in perms) ≈ 1 atol = 2.0e-3
        # against the frequencies of the orders drawn by the sampler (6 sites)
        rng = Xoshiro(8)
        N = 400_000
        cnt = Dict{Vector{Int}, Int}()
        for _ in 1:N
            σ = sortperm((1:6) .+ [rand(rng, f) for _ in 1:6])
            cnt[σ] = get(cnt, σ, 0) + 1
        end
        top = sort(collect(cnt); by = last, rev = true)[1:12]
        for (σ, c) in top
            p = c / N
            @test exp(loglik_order_exact(σ, f)) ≈ p atol = 4 * sqrt(p * (1 - p) / N) + 1.0e-3
        end
    end

    @testset "CommitmentTiming: re-weighting identity and recovery from known orders" begin
        k = 4
        η = [-3.1, -2.5, -1.9, -0.8, 0.0, -0.8, -1.9, -2.5, -3.1]
        # if the law scored is the propensity model's own order law, every stem term is exactly 0
        sim = simulate_stems(η; plants = 5, organs = 30, rng = Xoshiro(2))
        post = [(; samples = [simulate_stems(η; plants = 1, organs = 30, rng = Xoshiro(100 * s + j)).orders[1] for j in 1:20]) for s in 1:5]
        smp = order_samples(post, η)
        @test all(sum(s.w) ≈ 1 for s in smp)
        # one workspace per call: stem_terms is threaded over stems
        @test all(abs.(stem_terms(smp, σ -> logprob_permutation(PermanentWorkspace(k, 30), η, σ))) .< 1.0e-10)
        @test_throws ArgumentError order_samples([(; samples = [[6, 1, 2, 3, 4, 5, 7]])], η) # beyond the band
        # stems whose order is known (a posterior concentrated on the true order): the scorer reduces to the exact
        # likelihood, so fit_delay_law is a maximum-likelihood fit of the law; tail masses and core sd recovered. The costs
        # only shift every term by a constant: a wide band (k = 8) so that every simulated order is inside it
        η = [-1.0 * abs(d) for d in -8:8]
        truth = DelayLaw(0.42; πL = 0.06, mL = 1.3, πE = 0.05, mE = 1.0)
        orders = simulate_stems(truth; plants = 150, organs = 40, rng = Xoshiro(5)).orders
        known = order_samples([(; samples = [σ]) for σ in orders], η)
        @test relative_loglik(known, truth) ≈ sum(loglik_order_exact(σ, truth) for σ in orders) - sum(s.logA[1] for s in known)
        fit = fit_delay_law(known, :both; init = DelayLaw(0.5; πL = 0.1, mL = 1.0, πE = 0.1, mE = 1.0))
        @test fit.converged
        inf = delay_law_inference(known, fit)
        tt = tail_masses(truth)
        @test abs(inf.estimate.s - truth.s) < 3 * inf.se.s + 0.01
        @test abs(inf.estimate.late - tt.late) < 3 * inf.se.late + 0.005
        @test abs(inf.estimate.early - tt.early) < 3 * inf.se.early + 0.005
        @test all(x -> x > 0, values(inf.se))
        # no false tails: from a core-only truth, the two-tail fit gains almost nothing over the core-only fit
        core_orders = simulate_stems(DelayLaw(0.42); plants = 150, organs = 40, rng = Xoshiro(6)).orders
        kc = order_samples([(; samples = [σ]) for σ in core_orders], η)
        f0, f2 = fit_delay_law(kc, :core; init = DelayLaw(0.5)), fit_delay_law(kc, :both)
        @test f0.law.s ≈ 0.42 atol = 0.02
        @test f2.rel_ll - f0.rel_ll < 5 # χ²(4)/2 is below 5 with probability > 0.99
        # without tails, a short tail holding most sites only reshapes the core: the off-time fraction of the two-tail law
        # is then loosely determined, and checked against its own standard error
        i2 = delay_law_inference(kc, f2)
        @test abs(i2.estimate.off - tail_masses(DelayLaw(0.42)).off) < 3 * i2.se.off
    end

    @testset "CommitmentTiming: the overtaking profile" begin
        # Gaussian delays: q_d = Φ(−d / (s√2)); a tailed law against simulated pairs
        @test overtaking(DelayLaw(0.42)) ≈ [cdf(Normal(), -d / (0.42 * sqrt(2))) for d in 1:4] rtol = 0.01
        f = DelayLaw(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)
        rng = Xoshiro(17)
        e = [rand(rng, f) - rand(rng, f) for _ in 1:400_000]
        @test overtaking(f) ≈ [mean(e .> d) for d in 1:4] atol = 0.001
        @test_throws ArgumentError overtaking(f; d = [0.33]) # not a multiple of the grid step
        # model-free from known orders: the fraction of inverted pairs, with clustered standard errors
        η = [-1.0 * abs(d) for d in -8:8]
        orders = simulate_stems(f; plants = 150, organs = 40, rng = Xoshiro(23)).orders
        known = order_samples([(; samples = [σ]) for σ in orders], η)
        pf = overtaking_profile(known)
        @test pf.d == 1:4 && size(pf.V) == (4, 4)
        @test pf.q[1] ≈ mean(count(invperm(σ)[i + 1] < invperm(σ)[i] for i in 1:39) / 39 for σ in orders)
        @test all(abs.(pf.q .- overtaking(f)) .< 3 .* pf.se .+ 1.0e-4)
        # the fitted law's profile, with its covariance, agrees with the model-free one
        fit = fit_delay_law(known, :both)
        inf = delay_law_inference(known, fit)
        @test inf.profile.q ≈ overtaking(fit.law)
        @test [inf.estimate.q1, inf.estimate.q2] ≈ inf.profile.q[1:2]
        @test all(abs.(inf.profile.q[1:3] .- pf.q[1:3]) .< 3 .* pf.se[1:3])
        # comparison: a group against itself (no difference) and against a law with more overtaking
        t0 = compare_profiles(pf, pf)
        @test t0.diff == zeros(3) && t0.df == 3
        g = DelayLaw(0.42; πL = 0.2, mL = 1.5, πE = 0.2, mE = 1.0)
        # (the profile needs only the orders and their weights, not the propensity costs)
        pg = overtaking_profile([(; orders = [σ], w = [1.0]) for σ in simulate_stems(g; plants = 150, organs = 40, rng = Xoshiro(24)).orders])
        t1 = compare_profiles(pf, pg)
        @test all(t1.diff .> 0) && t1.p < 1.0e-6
    end

    @testset "CommitmentTiming: exact derivatives and the shared core" begin
        CT = PhyllotaxisFold.CommitmentTiming
        η = [-1.0 * abs(d) for d in -8:8]
        known(f, S, seed) = order_samples([(; samples = [σ]) for σ in simulate_stems(f; plants = S, organs = 35, rng = Xoshiro(seed)).orders], η)
        # the gradient of the summed terms (ForwardDiff through the recursion) against central differences
        smp = known(DelayLaw(0.42; πL = 0.06, mL = 1.3, πE = 0.05, mE = 1.0), 20, 3)
        obj(v) = sum(CT._stem_terms(smp, CT._law(v, :both), DELAY_GRID))
        v = [log(0.4), -2.0, 0.3, log(1.2), 0.1]
        h = 1.0e-5
        @test CT.ForwardDiff.gradient(obj, v) ≈ [(obj(v .+ h .* (1:5 .== i)) - obj(v .- h .* (1:5 .== i))) / 2h for i in 1:5] rtol = 1.0e-5
        # two groups with the same core and different tails: the shared-core fit recovers the core and each group's
        # off-time fraction, and the likelihood ratio does not reject the shared core
        fA, fB = DelayLaw(0.42; πL = 0.04, mL = 1.2, πE = 0.04, mE = 1.0), DelayLaw(0.42; πL = 0.15, mL = 1.2, πE = 0.12, mE = 1.0)
        groups = [known(fA, 120, 11), known(fB, 120, 12)]
        t = shared_core_test(groups, :both)
        @test t.shared.converged && length(t.shared.laws) == 2
        @test t.shared.laws[1].s == t.shared.laws[2].s
        @test t.shared.laws[1].s ≈ 0.42 atol = 0.05
        @test t.df == 1 && t.lr ≥ 0 && t.p > 0.001
        @test t.shared.rel_ll ≈ sum(t.shared.rel_ll_groups)
        inf = delay_law_inference(groups, t.shared)
        @test length(inf.estimate) == 2
        for (g, f) in enumerate((fA, fB))
            @test abs(inf.estimate[g].off - tail_masses(f).off) < 3 * inf.se[g].off + 0.01
        end
        @test inf.Δoff[1].estimate ≈ inf.estimate[2].off - inf.estimate[1].off
        @test inf.Δprofile[1].diff ≈ inf.profile[2].q .- inf.profile[1].q
        @test inf.Δprofile[1].p < 0.01 # the heavier tails overtake more
        @test abs(inf.Δoff[1].estimate - (tail_masses(fB).off - tail_masses(fA).off)) < 3 * inf.Δoff[1].se + 0.01
    end

    @testset "CommitmentTiming: the whole pipeline on simulated stems (angles → orders → delay law)" begin
        # angles from a known delay law; fit the propensity model; exact posterior order samples; delay law fitted through
        # the re-weighting: the tail masses must come back
        truth = DelayLaw(0.42; πL = 0.08, mL = 1.2, πE = 0.08, mE = 1.0)
        sim = simulate_stems(truth; plants = 40, organs = 35, logμκ = 1.9, σκ = 0.15, rng = Xoshiro(41))
        m = PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 16, ngrid_κ = 6)
        r = fit_map(m)
        post = order_posterior(m, r.u; nsamples = 60, rng = Xoshiro(42))
        smp = order_samples(post, constrained_parameters(m, r.u).η)
        fit = fit_delay_law(smp, :both)
        inf = delay_law_inference(smp, fit)
        tt = tail_masses(truth)
        @test abs(inf.estimate.off - tt.off) < 3 * inf.se.off + 0.01
        @test abs(inf.estimate.s - truth.s) < 3 * inf.se.s + 0.03
        # the true law scores close to the fitted one (no large optimism)
        @test 0 ≤ fit.rel_ll - relative_loglik(smp, truth) < 10
    end

    @testset "Code quality: Aqua and explicit imports" begin
        # ambiguities in the package's own methods only (its dependencies have their own)
        Aqua.test_all(PhyllotaxisFold; ambiguities = (; recursive = false))
        @test check_no_stale_explicit_imports(PhyllotaxisFold) === nothing
    end

end # testset PhyllotaxisFold
