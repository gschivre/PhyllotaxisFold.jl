using PhyllotaxisFold
using Test, Random, Statistics, LinearAlgebra
using Distributions, ForwardDiff, LogExpFunctions
using LogDensityProblems: LogDensityProblems
using Bijectors: Bijectors
using SpecialFunctions: SpecialFunctions
using Aqua: Aqua
using ExplicitImports: check_no_stale_explicit_imports
import PhyllotaxisFold: delay_logpdf, free_parameters, with_free_parameters, reported_parameters, nshared_parameters, delay_grid
# internals, by their qualified names (one module; the permanents are the submodule `Permanents`)
const AL = PhyllotaxisFold
const TO = PhyllotaxisFold
# internals tested directly: the banded permanents and the numerical tools
using PhyllotaxisFold.Permanents: PermutationGraph, PermanentWorkspace, toeplitz_weights!, forward!, forward_linear!, backward!,
    sample_permutation!, map_permutation, logprob_permutation, cycle_lengths, prefix_logperms!
using PhyllotaxisFold: logbesseli, besseliratio, marginal_quantiles

#################################################################################################################
# helpers and shared fixtures
#################################################################################################################

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

# bytes allocated by f(args...), after a first call
allocs(f::F, args::Vararg{Any, N}) where {F, N} = (f(args...); @allocated f(args...)) # (specialised on f: Julia 1.10)

# central differences of f at u along the directions in `dirs`
fd_dir(f, u, dirs; h = 1.0e-5) = [(f(u .+ h .* e) - f(u .- h .* e)) / 2h for e in dirs]

# the delay law of the simulated stems (orders from independent delays, the model of the analysis)
const LAW = GaussianExpTails(0.45; πL = 0.05, mL = 1.2, πE = 0.05, mE = 1.0)
# wide costs for stems whose order is known: a band k = 8 holds every simulated order
const ηWIDE = [-1.0 * abs(d) for d in -8:8]
# order samples of stems whose order is known (a posterior concentrated on the true order): the re-weighting then
# reduces to the exact likelihood, so a delay-law fit on them is a maximum-likelihood fit
known_samples(f, S, n, seed) = order_samples([(; samples = [σ]) for σ in simulate_stems(f; plants = S, organs = n, rng = Xoshiro(seed)).orders], ηWIDE)

# delay laws defined outside the package, as a user would, those of the manual page "Custom delay laws"; and a
# one-sided law (support z ≥ 0) whose overtaking profile is known exactly
struct StudentDelay{T <: Real} <: DelayLaw
    s::T
    ν::T
end
delay_logpdf(f::StudentDelay, z) = -(f.ν + 1) / 2 * log1p((z / f.s)^2 / f.ν)
free_parameters(f::StudentDelay) = [log(f.s), log(f.ν)]
with_free_parameters(::StudentDelay, v) = StudentDelay(exp(v[1]), exp(v[2]))
reported_parameters(f::StudentDelay) = (; s = f.s, ν = f.ν)
nshared_parameters(::StudentDelay) = 1 # no rand: simulate_stems uses the fallback sampler

# the Student t on its own, wider grid (delay_grid)
struct WideStudent{T <: Real} <: DelayLaw
    s::T
    ν::T
end
const WIDE_GRID = range(-60.0, 60.0; step = 0.05)
delay_logpdf(f::WideStudent, z) = delay_logpdf(StudentDelay(f.s, f.ν), z)
free_parameters(f::WideStudent) = [log(f.s), log(f.ν)]
with_free_parameters(::WideStudent, v) = WideStudent(exp(v[1]), exp(v[2]))
delay_grid(::WideStudent) = WIDE_GRID

struct GumbelDelay{T <: Real} <: DelayLaw
    β::T
end
delay_logpdf(f::GumbelDelay, z) = z / f.β - exp(z / f.β)
free_parameters(f::GumbelDelay) = [log(f.β)]
with_free_parameters(::GumbelDelay, v) = GumbelDelay(exp(v[1]))
reported_parameters(f::GumbelDelay) = (; β = f.β)
Base.rand(rng::AbstractRNG, f::GumbelDelay) = f.β * log(-log(rand(rng)))
# its order law in closed form (Plackett–Luce): going up, site j with probability ∝ e^{−j/β} among the unused sites
function plackett_luce_logP(σ, β)
    used = falses(length(σ))
    lp = 0.0
    for j in σ
        lp += -j / β - logsumexp(-l / β for l in eachindex(σ) if !used[l])
        used[j] = true
    end
    return lp
end

struct ExpDelay{T <: Real} <: DelayLaw
    m::T
end
delay_logpdf(f::ExpDelay, z) = z < 0 ? -Inf : -z / f.m
free_parameters(f::ExpDelay) = [log(f.m)]
with_free_parameters(::ExpDelay, v) = ExpDelay(exp(v[1]))

# daily cumulative counts of open flowers of plants with flowering parameters p (`Plastochron`), each recorded until 3
# days without a new flower; and each plant's change point τ
function simulate_counts(rng, p, plants)
    return map(1:plants) do _
        τi = exp(p[4] + exp(p[5]) * randn(rng))
        c, acc, z, k = Float64[], 0.0, 0, 0
        while z < 3 && k < 80
            k += 1
            n = rand(rng, Poisson(max(flowering_cumulative(Float64(k), p, τi) - flowering_cumulative(k - 1.0, p, τi), 1.0e-12)))
            acc += n; push!(c, acc)
            z = (n == 0 && acc > 0 ? z + 1 : 0)
        end
        (; c, τ = τi)
    end
end

# The code examples of the manual and the README, run as written: the Julia blocks of a page in order (indented ones
# dedented), each run with `include_string` in a sandbox module that stands in for the reader's session. The sandbox
# shrinks the work, not the code: coarse grids and quick twist fits through two wrappers, small simulated data in
# place of the reader's, and a few textual substitutions of sizes.
doc_path(page) = joinpath(@__DIR__, "..", page == "README.md" ? page : joinpath("docs", "src", page))
doc_blocks(page) = [replace(m.captures[1], r"^    "m => "") for m in eachmatch(r"```(?:julia|@example \w+)\n(.*?)```"s, read(doc_path(page), String))]
function doc_sandbox(; data...)
    mod = Module(:ManualExample)
    Core.eval(
        mod, quote
            import PhyllotaxisFold
            using Random, Statistics
            PhyllotaxisModelAD(args...; kw...) = PhyllotaxisFold.PhyllotaxisModelAD(args...; ngrid_ϕ = 8, ngrid_κ = 4, kw...)
            fit_map(m; kw...) = PhyllotaxisFold.fit_map(m; recenter = 0, nodes_kw = (; nnodes = 3), kw...)
            using PhyllotaxisFold
        end
    )
    for (k, v) in data
        Core.eval(mod, :($k = $v))
    end
    return mod
end
run_blocks!(mod, page, idxs; subs = ()) =
    foreach(i -> include_string(mod, foldl(replace, subs; init = doc_blocks(page)[i]), "$(page) block $(i)"), idxs)

# stems with a twist (degrees per position) for the angle-model and twist tests: one simulation, a coarse nuisance grid
const TWIST = simulate_stems(LAW; plants = 8, organs = 40, logμκ = 1.8, torsion_mean = 4.0, torsion_sd = 3.0, rng = Xoshiro(21))
twist_model(; torsion = nothing) = PhyllotaxisModelAD(TWIST.angles, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4, outlier = false, torsion)

# spiral_breaks' log Z(j) by brute force on a short stem (no twist): every segmentation, every sequence of part values,
# each part by a direct permanent at each phase and κ node (its sites at ϕ + o (i − 1) ψ in the stem's indices)

function brute_breaks(fit, s, kd)
    m = fit.model
    p = PhyllotaxisFold.stem_params(m, fit.u)
    g = m.grid
    y = m.data[s]
    n, k = length(y), m.k
    κs = exp.(p.logμκ .+ p.σκ .* g.κ_nodes)
    lps = PhyllotaxisFold._angle_logprobs(p.α)
    L = length(g.angles)
    ws = PermanentWorkspace(Float64, k, n; logspace = true)
    πo = something(p.outlier, 0.0)
    function part(a, b, o, j, q, ϕ)
        nn, κ = b - a + 1, κs[q]
        lI = PhyllotaxisFold.logbesseli(0, κ)
        for i in 1:nn, d in -k:k
            1 ≤ i + d ≤ nn || continue
            site = ϕ + o * (a + i + d - 2) * g.angles[j]
            lv = log((1 - πo) * exp(κ * cos(y[a + i - 1] - site) - lI) / (2π) + πo / (2π))
            ws.w[i, d + k + 1] = p.η[d + k + 1] + lv
        end
        l = forward!(ws, nn)
        toeplitz_weights!(ws, p.η, nn)
        return l - forward!(ws, nn)
    end
    lat = true # every part has its own phase
    Nϕ = length(g.ϕ_grid)
    # the shared values: (ϕ index if the lattice is shared, o if not breaking, j if not breaking, q if not breaking)
    sh_ϕ = lat ? (0,) : Tuple(1:Nϕ)
    sh_o = :orientation in kd ? (0,) : (-1, 1)
    sh_j = :angle in kd ? (0,) : Tuple(1:L)
    sh_q = :precision in kd ? (0,) : Tuple(eachindex(κs))
    types = [(o, j) for j in (:angle in kd ? (1:L) : (0:0)) for o in (:orientation in kd ? (-1, 1) : (0,))]
    lpt(t) = (:orientation in kd ? -log(2) : 0.0) + (:angle in kd ? lps[t[2]] : 0.0)
    Z = fill(-Inf, n)
    memo = Dict{Any, Float64}()
    for cutmask in 0:(2^(n - 1) - 1)
        cuts = [c + 1 for c in 1:(n - 1) if (cutmask >> (c - 1)) & 1 == 1]
        bounds = [1; cuts; n + 1]
        parts = [(bounds[i], bounds[i + 1] - 1) for i in 1:(length(bounds) - 1)]
        J = length(cuts)
        for iϕ in sh_ϕ, os in sh_o, js in sh_j, qs in sh_q
            lw = (lat ? 0.0 : -log(Nϕ)) + (:orientation in kd ? 0.0 : -log(2)) + (:angle in kd ? 0.0 : lps[js]) +
                (:precision in kd ? 0.0 : g.κ_logweights[qs])
            # type sequences
            tseqs = length(types) == 1 ? [fill(types[1], length(parts))] :
                [collect(t) for t in Iterators.product(ntuple(_ -> types, length(parts))...) if all(t[i] != t[i + 1] for i in 1:(length(t) - 1))]
            for ts in tseqs
                v = lw
                for (ip, (a, b)) in enumerate(parts)
                    t = ts[ip]
                    v += ip == 1 ? lpt(t) : (length(types) == 1 ? 0.0 : lpt(t) - log1mexp(lpt(ts[ip - 1])))
                    o = :orientation in kd ? t[1] : os
                    j = :angle in kd ? t[2] : js
                    v += get!(memo, (a, b, o, j, iϕ, qs)) do
                        logsumexp(
                            (lat ? -log(Nϕ) : 0.0) + (:precision in kd ? g.κ_logweights[q] : 0.0) + part(a, b, o, j, q, g.ϕ_grid[ϕi])
                                for ϕi in (lat ? (1:Nϕ) : (iϕ,)) for q in (:precision in kd ? eachindex(κs) : (qs,))
                        )
                    end
                end
                Z[J + 1] = logaddexp(Z[J + 1], v)
            end
        end
    end
    return Z
end

@testset verbose = true "PhyllotaxisFold" begin

    #############################################################################################################
    # numerical tools and banded permanents
    #############################################################################################################

    @testset "Common: log Iₙ and the Bessel ratio, values and dual derivatives" begin
        # against SpecialFunctions (besselix = e^{-x} Iₙ(x)), on both sides of the switch to the asymptotic series at x = 100
        for n in (0, 1, 3), x in (1.0e-3, 0.5, 4.0, 30.0, 99.0, 101.0, 600.0)
            @test logbesseli(n, x) ≈ log(SpecialFunctions.besselix(n, x)) + x rtol = 1.0e-10 atol = 1.0e-14
        end
        @test logbesseli(0, 0.0) == 0 && logbesseli(2, 0.0) == -Inf
        for x in (0.3, 5.0, 150.0, 2.0e4)
            @test besseliratio(x) ≈ SpecialFunctions.besselix(1, x) / SpecialFunctions.besselix(0, x) rtol = 1.0e-8
            # d log I₀ / dx = I₁ / I₀
            @test ForwardDiff.derivative(z -> logbesseli(0, z), x) ≈ besseliratio(x) rtol = 1.0e-8
            # dual rule against a central difference, or against A'(x) ≈ 1/(2x²) + 1/(4x³) where differences lose precision
            ref = x < 1.0e3 ? (besseliratio(x + 1.0e-4 * x) - besseliratio(x - 1.0e-4 * x)) / (2.0e-4 * x) : 1 / (2x^2) + 1 / (4x^3)
            @test ForwardDiff.derivative(besseliratio, x) ≈ ref rtol = 1.0e-5
        end
        @test ForwardDiff.derivative(besseliratio, 0.0) == 0.5
    end

    @testset "Permanents: forward / backward / Viterbi / sampling against enumeration" begin
        rng = Xoshiro(1)
        n, k = 7, 2
        L = randn(rng, n, 2k + 1) # log-weights
        perms = banded_perms(n, k)
        lw(σ) = sum(L[i, σ[i] - i + k + 1] for i in 1:n)
        logZ = logsumexp(lw.(perms))
        for logspace in (true, false)
            ws = PermanentWorkspace(k, n; logspace)
            ws.w[1:n, :] .= L
            @test forward!(ws, n) ≈ logZ
            exact = [sum(exp(lw(σ) - logZ) for σ in perms if σ[i] == i + d; init = 0.0) for i in 1:n, d in -k:k]
            @test backward!(ws, n) ≈ exact atol = 1.0e-12
        end
        @test map_permutation(PermutationGraph(k), L, n) == perms[argmax(lw.(perms))]
        # tridiagonal case (k = 1): the number of banded permutations is a Fibonacci number
        ws1 = PermanentWorkspace(1, 10)
        toeplitz_weights!(ws1, zeros(3), 10)
        @test exp(forward!(ws1, 10)) ≈ 89
        # exact sampling: frequencies of the most probable order
        ws = PermanentWorkspace(k, n)
        counts = Dict{Vector{Int}, Int}()
        for _ in 1:20_000
            ws.w[1:n, :] .= L
            σ = sample_permutation!(rng, ws, n)
            counts[σ] = get(counts, σ, 0) + 1
        end
        σmax = perms[argmax(lw.(perms))]
        @test counts[σmax] / 20_000 ≈ exp(lw(σmax) - logZ) atol = 0.01
        @test sort(cycle_lengths([2, 1, 3, 5, 6, 4])) == [1, 2, 3]
        η = [-1.0, -0.5, 0.0, -0.5, -1.0]
        @test sum(exp(logprob_permutation(PermanentWorkspace(k, n), η, σ)) for σ in perms) ≈ 1
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
        # linear weights (`forward_linear!`, the outlier mixture): the same value and marginals as the log-weights, also
        # where the scaled passes fall back to log-space (the log-weights are then rebuilt from the linear ones)
        function linw!(ws, y, κ; π = 0.02)
            for d in -k:k, i in max(1, 1 - d):min(n, n - d)
                c = cos(y[i] + (i + d - 1) * PhyllotaxisFold.Golden_angle) - 1
                ws.ew[i, d + k + 1] = exp(η[d + k + 1]) * ((1 - π) * exp(κ * c) + π * exp(logbesseli(0, κ) - κ))
            end
        end
        for κ in (30.0, 3.0e3, 3.0e5), logspace in (false, true)
            wl, wr = PermanentWorkspace(k, n; logspace), PermanentWorkspace(k, n; logspace = true)
            linw!(wl, yout, κ)
            wr.w[1:n, :] .= log.(wl.ew[1:n, :])
            @test forward_linear!(wl, n) ≈ forward!(wr, n)
            @test backward!(wl, n) ≈ backward!(wr, n) atol = 1.0e-10
        end
        # a backward pass that underflows after a scaled linear forward pass rebuilds the log-weights
        wl = PermanentWorkspace(k, n)
        linw!(wl, yout, 30.0)
        forward_linear!(wl, n)
        wl.w .= NaN # what the linear pass leaves in w must never be read
        wl.ew[12, :] .*= 1.0e-200 # the scaled backward normaliser of row 12 now underflows
        ρl = backward!(wl, n)
        @test wl.inlog[] && all(isfinite, ρl) && all(sum(ρl; dims = 2) .≈ 1)
    end

    #############################################################################################################
    # step 1: the propensity model, the angle likelihood and the twist
    #############################################################################################################

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
        q = marginal_quantiles(d, 0.5; N = 20_000)
        @test all(q .< 0) && issorted(q; rev = true)
    end

    @testset "Propensity model: densities, parameters, frequencies, simulation" begin
        data = [rand(Xoshiro(s), VonMises(0.0, 1.0), 20) for s in 1:3]
        model = PhyllotaxisModelAD(data, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4, logspace = true, outlier = false)
        θ = prior_mean(model)
        @test isfinite(LogDensityProblems.logdensity(model, θ))
        lin = PhyllotaxisModelAD(data, PhyllotaxisPrior(); ngrid_ϕ = 8, ngrid_κ = 4, logspace = false, outlier = false)
        @test LogDensityProblems.logdensity(lin, θ) ≈ LogDensityProblems.logdensity(model, θ)
        @test_throws ArgumentError PhyllotaxisModelAD([data[1][1:3]]) # band k = 4 longer than the stem
        # symmetric costs: k of them, mirrored in the full vector; the prior covers its own mean
        @test LogDensityProblems.dimension(model) == 4 + 3
        @test vec(log_propensity(model, θ)) == θ[1:4]
        ηf = full_eta(model, θ)
        @test ηf ≈ reverse(ηf) && ηf[5] == 0
        @test θ[6] ≈ 0.25 # prior mean of σκ: the half-t mean for ν = 4, τ = 0.25
        @test all(prior_CIs(model)[:, 1] .< θ .< prior_CIs(model)[:, 2])
        θr = rand_priorparameters(model; rng = Xoshiro(8))
        @test length(θr) == 7 && isfinite(LogDensityProblems.logdensity(model, θr))
        # the transformed problem: log density = angle log-likelihood + log prior with its Jacobian, and its analytic
        # gradient
        prob = TransformedLogDensityProblem(model)
        u = unconstrained_parameters(model, θ)
        @test u ≈ prior_mean_parameters(model)
        @test LogDensityProblems.logdensity(prob, u) ≈ angle_loglikelihood(model, u) + AL.model_logprior(model, u)
        @test LogDensityProblems.dimension(prob) == length(u)
        @test LogDensityProblems.capabilities(typeof(prob)) == LogDensityProblems.LogDensityOrder{1}()
        lg, gg = LogDensityProblems.logdensity_and_gradient(prob, u .+ 0.1)
        @test lg ≈ LogDensityProblems.logdensity(prob, u .+ 0.1)
        @test gg ≈ ForwardDiff.gradient(w -> LogDensityProblems.logdensity(prob, w), u .+ 0.1) rtol = 1.0e-6
        @test sum(stem_loglik(model, u, s) for s in 1:3) ≈ angle_loglikelihood(model, u)
        # parameters: round trip, a half or a full symmetric η, and the errors
        η = [-3.0, -2.4, -1.9, -0.8, 0.0, -0.8, -1.9, -2.4, -3.0]
        uA = unconstrained_parameters(model; η, logμκ = 1.8, σκ = 0.1, α = 0.95)
        θA = constrained_parameters(model, uA)
        @test θA.η ≈ η && θA.σκ ≈ 0.1 && θA.outlier == 0 && !haskey(θA, :m_ω)
        p = AL.stem_params(model, uA)
        @test [p.logμκ, p.σκ] ≈ [θA.logμκ, θA.σκ] && p.α == θA.α && p.α isa NTuple{1} && p.outlier === nothing
        @test collect(θA.angle_probs) ≈ [0.95, 0.05]
        @test unconstrained_parameters(model; η = η[6:end], logμκ = 1.8, σκ = 0.1, α = 0.95) ≈ uA
        @test_throws ArgumentError unconstrained_parameters(model; η = [-3.0, -2.4, -1.9, -0.8, 0.0, -0.9, -1.8, -2.6, -3.1], logμκ = 1.8, σκ = 0.1, α = 0.95) # asymmetric
        @test_throws ArgumentError unconstrained_parameters(model; η = η[6:8], logμκ = 1.8, σκ = 0.1, α = 0.95)
        # the dimension counts every parameter: π_o with outliers, (m_ω, s_ω) with the twist
        mo = PhyllotaxisModelAD(data; ngrid_ϕ = 8, ngrid_κ = 4)
        mt = PhyllotaxisModelAD(data; ngrid_ϕ = 8, ngrid_κ = 4, torsion = TorsionPrior())
        @test LogDensityProblems.dimension(mo) == 8 && LogDensityProblems.dimension(mt) == 10
        @test length(prior_mean(mt)) == length(prior_mean_parameters(mt)) == size(prior_CIs(mt), 1) == 10
        @test all(prior_CIs(mt)[:, 1] .< prior_mean(mt) .< prior_CIs(mt)[:, 2])
        ut = unconstrained_parameters(mt; η, logμκ = 1.8, σκ = 0.1, α = 0.95, outlier = 0.03, m_ω = 1.5, s_ω = 0.4)
        θt = constrained_parameters(mt, ut)
        @test θt.outlier ≈ 0.03 && θt.m_ω ≈ 1.5 && θt.s_ω ≈ 0.4
        # the twist model's log density integrates each stem's rate over the nodes placed by fit_map
        @test_throws ArgumentError LogDensityProblems.logdensity(mt, prior_mean(mt))
        # displacement probabilities and realised frequencies
        @test all(sum(displacement_probability(model, θ); dims = 1) .≈ 1)
        @test sum(displacement_probability(model, θ)[2:end]) ≈ 2 * sum(exp, θ[1:4]) / (1 + 2 * sum(exp, θ[1:4]))
        steep = AL.stem_params(model, unconstrained_parameters(model; η = [-12.0, -24.0, -36.0, -48.0], logμκ = 1.5, σκ = 0.2, α = 0.95)).η[6:end]
        fd, _ = estimate_displacement_frequencies(model, [steep]; N = 3, n = 30, rng = Xoshiro(1))
        @test fd[1] > 0.999 # steep costs: nearly every organ in place
        @test sum(fd) ≈ 1 atol = 1.0e-6
        fp, _ = estimate_permutation_frequencies(model, nothing; N = 200, n = 30, rng = Xoshiro(2))
        @test sum(fp) ≈ 1
        s = sample_prior(model; n = 60, Nsample = 2, rng = Xoshiro(5)) # n larger than the data
        @test length(s[1].data) == 60 && isperm(s[1].permutation) && isempty(s[1].outliers) && s[1].twist == 0
        # the prior predictive draws the outliers and the twist of the model
        so = sample_prior(mt; n = 60, Nsample = 200, rng = Xoshiro(6))
        @test mean(length(x.outliers) for x in so) / 60 ≈ 0.02 atol = 0.01 # prior mean of π_o
        # the twist: ω ~ N(m_ω, s_ω²), m_ω ~ N(0, 2²), s_ω ~ half-t(4, 1): sd² = 2² + E[s_ω²] = 4 + 4 / (4 − 2)
        @test std([x.twist for x in so]) ≈ sqrt(6) rtol = 0.2
        # simulate_stems(η): η_1..η_k of any length (k odd or even), or the full vector
        maxdisp(r) = maximum(maximum(abs.(σ .- eachindex(σ))) for σ in r.orders)
        for η in ([-1.0, -2.5, -4.0], [-1.0, -2.5, -4.0, -6.0])
            r = simulate_stems(η; plants = 300, organs = 40, rng = Xoshiro(1))
            @test 1 < maxdisp(r) ≤ length(η)
            @test simulate_stems(vcat(reverse(η), 0.0, η); plants = 300, organs = 40, rng = Xoshiro(1)).orders == r.orders
        end
    end

    @testset "Angle likelihood: analytic gradients and Hessian, per stem and summed" begin
        m = PhyllotaxisModelAD(TWIST.angles[1:3]; ngrid_ϕ = 8, ngrid_κ = 4, outlier = false)
        prob = TransformedLogDensityProblem(m)
        v = prior_mean_parameters(m) .+ 0.1
        l, g, H = AL.angle_vgh(m, v, nothing, 2)
        @test l ≈ LogDensityProblems.logdensity(prob, v)
        @test g ≈ ForwardDiff.gradient(w -> LogDensityProblems.logdensity(prob, w), v) rtol = 1.0e-6
        @test H ≈ ForwardDiff.hessian(w -> LogDensityProblems.logdensity(prob, w), v) rtol = 1.0e-5
        @test AL.angle_vgh(m, v, nothing, 1)[2] ≈ g && AL.angle_vgh(m, v, nothing, 0)[1] ≈ l
        # fixed twist rates: the same derivatives, and the rate removes the twist from the angles
        ωs = [1.0, -2.0, 3.0]
        l2, g2, _ = AL.angle_vgh(m, v, ωs, 1)
        @test g2 ≈ ForwardDiff.gradient(w -> AL.angle_vgh(m, w, ωs, 0)[1], v) rtol = 1.0e-6
        @test stem_loglik(m, v, 2; ω = -2.0) ≈ stem_loglik(PhyllotaxisModelAD([corrected_angles(m, 2; ω = -2.0)]; ngrid_ϕ = 8, ngrid_κ = 4, outlier = false), v, 1)
        # outlier angles: gradient and Hessian, and π_o → 0 gives back the model without outliers
        mo = PhyllotaxisModelAD(TWIST.angles[1:3]; ngrid_ϕ = 8, ngrid_κ = 4, outlier = true)
        uo = prior_mean_parameters(mo) .+ 0.1
        lo, go, Ho = AL.angle_vgh(mo, uo, nothing, 2)
        fo(w) = AL.angle_vgh(mo, w, nothing, 0)[1]
        @test go ≈ ForwardDiff.gradient(fo, uo) rtol = 1.0e-6
        @test Ho ≈ ForwardDiff.hessian(fo, uo) rtol = 1.0e-4
        u0 = prior_mean_parameters(m)
        @test angle_loglikelihood(mo, [u0; -40.0]) ≈ angle_loglikelihood(m, u0) atol = 1.0e-8
    end

    @testset "Angle likelihood: type stability and allocations (regression guards)" begin
        for outlier in (false, true)
            m = PhyllotaxisModelAD(TWIST.angles[1:2]; ngrid_ϕ = 4, ngrid_κ = 3, outlier)
            u = prior_mean_parameters(m)
            p = @inferred AL.stem_params(m, u)
            @test (p.outlier === nothing) == !outlier
            @inferred AL.model_logprior(m, u)
            @inferred AL.params_jacobian(m, u)
            # with_grad is a run-time flag: a small union of the two return types, resolved by union splitting
            VG = Union{Tuple{Float64, Nothing}, Tuple{Float64, Vector{Float64}}}
            @inferred VG AL._data_value_grad(m, u, nothing, true)
            @inferred VG AL._data_value_grad(m, u, [0.5, 1.0], false)
            fws = AL._workspace(m, Float64)
            gθ = zeros(AL.nmodel(m))
            @inferred AL._stem_ll!(m, p, 1, 0.5, fws, gθ)
            @inferred AL._stem_ll!(m, p, 1, 0.5, fws)
            # the stem kernel allocates only its few per-κ-node vectors, the permanent nothing (measured inside a
            # function, after a first call)
            @test allocs(AL._stem_ll!, m, p, 1, 0.5, fws, gθ) < 2_000
            @test allocs(PhyllotaxisFold.Permanents.forward!, fws.perm, 40) == 0
        end
        # dual numbers through the kernel (the Hessian pass)
        m = PhyllotaxisModelAD(TWIST.angles[1:2]; ngrid_ϕ = 4, ngrid_κ = 3)
        ud = AL._seed(prior_mean_parameters(m))
        @inferred Union{Tuple{eltype(ud), Nothing}, Tuple{eltype(ud), Vector{eltype(ud)}}} AL._data_value_grad(m, ud, nothing, true)
    end

    @testset "Canonical angles: one, two (default) or more, stick-breaking prior" begin
        # the prior: M stick-breaking Betas for M + 1 angles, numbers converted, the lengths checked
        @test PhyllotaxisPrior() isa PhyllotaxisPrior{4, 1}
        p3 = PhyllotaxisPrior(; aα = (95, 999), bα = (5, 1), angles = (Golden_angle, Lucas_angle, π))
        @test p3 isa PhyllotaxisPrior{4, 2} && p3.angles[3] === Float64(π)
        @test PhyllotaxisPrior(; aα = (), bα = (), angles = Golden_angle) isa PhyllotaxisPrior{4, 0}
        @test PhyllotaxisPrior(; aα = 90, bα = 10).aα === (90.0,)
        @test_throws ArgumentError PhyllotaxisPrior(; aα = (95, 999), bα = (5, 1)) # two Betas, two angles
        @test_throws ArgumentError PhyllotaxisPrior(; aα = (95,), bα = (5, 1), angles = (1.0, 2.0, 3.0))
        @test_throws ArgumentError PhyllotaxisPrior(; angles = (Golden_angle, 2π - Golden_angle)) # the same spirals
        @test PhyllotaxisPrior{4}() isa PhyllotaxisPrior{4, 1}
        @test_throws ArgumentError PhyllotaxisPrior{3}()
        # stick-breaking probabilities and their derivatives
        lp = AL._angle_logprobs((0.9, 0.99))
        @test collect(exp.(lp)) ≈ [0.9, 0.1 * 0.99, 0.1 * 0.01] && AL._angle_logprobs(()) == (0.0,)
        @test [AL._dlogp_dalpha((0.9, 0.99), i, j) for i in 1:2, j in 1:3] ≈
            ForwardDiff.jacobian(a -> collect(AL._angle_logprobs(Tuple(a))), [0.9, 0.99])'
        # noble numbers: golden and Lucas, folded into (0, π], and the condition |ad − bc| = 1
        @test noble_angle(1, 0, 2, 1) ≈ Golden_angle && noble_angle(1, 0, 3, 1) ≈ Lucas_angle
        @test 0 < noble_angle(1, 1, 3, 2) ≤ π # 2π (1 + τ⁻¹) / (3 + 2τ⁻¹) folded
        @test_throws ArgumentError noble_angle(1, 0, 2, 2)
        # likelihood, gradient and Hessian with one and with three canonical angles
        for (pr, M) in ((PhyllotaxisPrior(; aα = (), bα = (), angles = Golden_angle), 0), (p3, 2))
            for outlier in (false, true)
                m = PhyllotaxisModelAD(TWIST.angles[1:3], pr; ngrid_ϕ = 8, ngrid_κ = 4, outlier)
                @test AL.nmodel(m) == 4 + 2 + M + outlier && length(AL._parnames(m)) == AL.nmodel(m)
                v = prior_mean_parameters(m) .+ 0.1
                f(w) = AL.angle_vgh(m, w, nothing, 0)[1]
                l, g, H = AL.angle_vgh(m, v, nothing, 2)
                @test g ≈ ForwardDiff.gradient(f, v) rtol = 1.0e-6
                @test H ≈ ForwardDiff.hessian(f, v) rtol = 1.0e-4
                θ = constrained_parameters(m, v)
                @test length(θ.α) == M && length(θ.angle_probs) == M + 1 && sum(θ.angle_probs) ≈ 1
                @test unconstrained_parameters(m; η = θ.η, θ.logμκ, θ.σκ, θ.α, θ.outlier) ≈ v
                p = @inferred AL.stem_params(m, v)
                fws = AL._workspace(m, Float64)
                @inferred AL._stem_ll!(m, p, 1, 0.5, fws, zeros(AL.nmodel(m)))
            end
        end
        # the twist integrated out, three angles: gradient and Hessian against central differences (the node-integrated
        # value is a Float64 path, so ForwardDiff cannot run through it)
        mt3 = PhyllotaxisModelAD(TWIST.angles[1:3], p3; ngrid_ϕ = 8, ngrid_κ = 4, torsion = TorsionPrior(mt = 0.0, st = 10.0, νt = 4.0, τt = 5.0))
        u3 = prior_mean_parameters(mt3)[1:AL.nmodel(mt3)]
        nd3 = AL._place_nodes(mt3, u3)
        v3 = [u3; 3.5; log(2.5)]
        _, g3, H3 = AL.twist_vgh(mt3, nd3, v3, 2)
        E = Matrix(1.0I, length(v3), length(v3))
        h = 1.0e-5
        fv3(w) = AL.twist_vgh(mt3, nd3, w, 0)[1]
        gv3(w) = AL.twist_vgh(mt3, nd3, w, 1)[2]
        @test g3 ≈ [(fv3(v3 .+ h .* e) - fv3(v3 .- h .* e)) / 2h for e in eachcol(E)] rtol = 1.0e-5
        @test H3 ≈ reduce(hcat, [(gv3(v3 .+ h .* e) - gv3(v3 .- h .* e)) / 2h for e in eachcol(E)]) rtol = 1.0e-4
        # a single angle: the likelihood of the default prior with α → 1
        mg = PhyllotaxisModelAD(TWIST.angles[1:3], PhyllotaxisPrior(; aα = (), bα = (), angles = Golden_angle); ngrid_ϕ = 8, ngrid_κ = 4)
        m2 = PhyllotaxisModelAD(TWIST.angles[1:3]; ngrid_ϕ = 8, ngrid_κ = 4)
        θg = constrained_parameters(mg, prior_mean_parameters(mg))
        @test angle_loglikelihood(mg, prior_mean_parameters(mg)) ≈
            angle_loglikelihood(m2, unconstrained_parameters(m2; η = θg.η, θg.logμκ, θg.σκ, α = 1 - 1.0e-12, θg.outlier)) rtol = 1.0e-8
        # stems at 180° are recognised when 180° is in the list; spiral stems are not
        sd = simulate_stems(LAW; plants = 40, organs = 30, logμκ = 1.8, α = (0.5, 0.5), angles = p3.angles, rng = Xoshiro(21))
        m3 = PhyllotaxisModelAD(sd.angles, p3; ngrid_ϕ = 16, ngrid_κ = 4, outlier = false)
        post = spiral_posterior(m3, unconstrained_parameters(m3; η = [-0.8, -1.9, -2.4, -3.0], logμκ = 1.8, σκ = 0.1, α = (0.5, 0.5)))
        # (Float64(π) == π is false: π is compared exactly)
        @test mean((p.p_angle[3] > 0.5) == (ψ == p3.angles[3]) for (p, ψ) in zip(post, sd.ψ)) ≥ 0.95
        @test count(==(p3.angles[3]), sd.ψ) > 5
        # the prior predictive and the simulator draw the angles of the list
        @test all(s -> s.canonical_angle in p3.angles, sample_prior(m3; Nsample = 20, rng = Xoshiro(3)))
        @test_throws ArgumentError simulate_stems(LAW; plants = 2, angles = p3.angles) # α: two probabilities needed
    end

    @testset "Angle model: recovery of the outlier rate and of the spiral of each stem" begin
        # outliers: π_o recovered on stems whose orders come from the banded permanent (the law of the model)
        η = [-3.5, -2.5, -1.2, 0.0, -1.2, -2.5, -3.5]
        sA = simulate_stems(η; plants = 30, organs = 35, logμκ = 2.0, σκ = 0.2, outlier = 0.06, rng = Xoshiro(9))
        μΔ, σΔ = η_prior((0.5, 1.25, 2.0), (0.1, 0.75, 2.0))
        mA = PhyllotaxisModelAD(sA.angles, PhyllotaxisPrior(; μΔ, σΔ); ngrid_ϕ = 8, ngrid_κ = 4, outlier = true)
        fA = fit_map(mA)
        @test fA isa PhyllotaxisFit && fA.converged && fA.twist === nothing
        # the convergence keywords reach the optimiser: one Newton step is not enough, and is reported as such
        f1 = fit_map(mA; iterations = 1)
        @test f1.iterations == 1 && !f1.converged && !fit_diagnostics(f1).converged
        @test occursin("NOT CONVERGED", sprint(show, MIME"text/plain"(), f1))
        @test abs(fA.θ.outlier - 0.06) < 3 * stderror(fA)[end] + 0.01
        # an angle that is not a finite number is refused, and a fit cannot start from a non-finite objective
        for v in (NaN, Inf)
            dv = deepcopy(sA.angles)
            dv[2][5] = v
            @test_throws ArgumentError PhyllotaxisModelAD(dv)
        end
        dm = [Vector{Union{Missing, Float64}}(y) for y in sA.angles]
        dm[1][1] = missing
        @test_throws ArgumentError PhyllotaxisModelAD(dm)
        tx = [PhyllotaxisFold.torsion_coordinate(length(y)) for y in sA.angles]
        tx[3][4] = NaN
        @test_throws ArgumentError PhyllotaxisModelAD(sA.angles; torsion_x = tx)
        nan_start(v, order) = (NaN, (order ≥ 1 ? fill(NaN, 2) : nothing), (order ≥ 2 ? fill(NaN, 2, 2) : nothing))
        @test_throws ErrorException PhyllotaxisFold.newton_map(nan_start, [0.0, 0.0])
        # the fit object: accessors, uncertainty and diagnostics
        @test coefnames(fA) == ["η_1", "η_2", "η_3", "log μκ", "σκ", "α", "π_o"]
        @test coef(fA) ≈ [fA.θ.η[5:end]; fA.θ.logμκ; fA.θ.σκ; fA.θ.α...; fA.θ.outlier]
        @test dof(fA) == 7 && nobs(fA) == 30 * 35
        @test loglikelihood(fA) ≈ angle_loglikelihood(mA, fA.u)
        @test sum(stem_loglik(fA)) ≈ loglikelihood(fA) && stem_loglik(fA)[2] ≈ stem_loglik(mA, fA.u, 2)
        @test fA.logposterior ≈ LogDensityProblems.logdensity(TransformedLogDensityProblem(mA), fA.u)
        V = vcov(fA)
        @test issymmetric(round.(V; sigdigits = 8)) && all(stderror(fA) .> 0)
        ct = coeftable(fA)
        @test ct.rownms == coefnames(fA) && all(ct.cols[3] .< coef(fA) .< ct.cols[4])
        d = fit_diagnostics(fA)
        @test d.hessian_posdef && d.gradnorm < 1.0e-3 && isfinite(d.log_evidence) && d.log_evidence < d.logposterior
        @test isempty(d.low_stems) # stems of the model's own law
        # a stem of uniform angles among them is flagged
        bad = [sA.angles[1:29]; [2π .* rand(Xoshiro(3), 35)]]
        fbad = fit_map(PhyllotaxisModelAD(bad, PhyllotaxisPrior(; μΔ, σΔ); ngrid_ϕ = 8, ngrid_κ = 4, outlier = true); u0 = fA.u)
        @test fit_diagnostics(fbad).low_stems == [30]
        txt = sprint(show, MIME"text/plain"(), fA)
        @test occursin("converged", txt) && occursin("π_o", txt) && occursin("Laplace log evidence", txt)
        @test occursin("PhyllotaxisFit", sprint(show, fA))
        # stems like the data, from the fit: same sizes, the fitted parameters, or a Laplace draw of them
        rep = simulate_stems(fA; rng = Xoshiro(1))
        @test length.(rep.angles) == length.(sA.angles) && rep.θ.η == fA.θ.η
        repl = simulate_stems(fA; laplace = true, rng = Xoshiro(2))
        @test repl.θ.η != fA.θ.η && length(repl.angles) == 30
        # divergence angles: a perfect spiral of either orientation reads ψ; the simulated sites (ϕ, o, ψ) are where the
        # angles fall without noise, twist or outliers
        @test divergence_angles([0.1 + 2.0 * (i - 1) for i in 1:5]; degrees = false) ≈ fill(2.0, 4)
        @test divergence_angles([1.0 - (i - 1) * PhyllotaxisFold.Golden_angle for i in 1:6]; o = -1) ≈ fill(rad2deg(PhyllotaxisFold.Golden_angle), 5)
        sx = simulate_stems([-1.0, -2.0]; plants = 3, organs = [20, 25, 30], logμκ = 12.0, rng = Xoshiro(3))
        @test length.(sx.angles) == [20, 25, 30] && all(y -> all(0 .≤ y .< 2π), sx.angles) # azimuths, as measured
        @test all(maximum(abs, sin.((sx.angles[s] .- (sx.ϕ[s] .+ sx.o[s] .* (sx.orders[s] .- 1) .* sx.ψ[s])) ./ 2)) < 0.05 for s in 1:3)
        # spiral of each stem: golden vs Lucas and the orientation
        sp = simulate_stems(LAW; plants = 30, organs = 30, logμκ = 1.8, α = 0.6, rng = Xoshiro(12))
        m = PhyllotaxisModelAD(sp.angles; ngrid_ϕ = 8, ngrid_κ = 4, outlier = false)
        post = spiral_posterior(m, unconstrained_parameters(m; η = [-0.8, -1.9, -2.4, -3.0], logμκ = 1.8, σκ = 0.1, α = 0.6))
        @test mean((p.p_angle[2] > 0.5) == (ψ == PhyllotaxisFold.Lucas_angle) for (p, ψ) in zip(post, sp.ψ)) ≥ 0.9
        @test all(p -> sum(p.p_angle) ≈ 1, post)
        @test mean((p.p_ccw > 0.5) == (o == 1) for (p, o) in zip(post, sp.o)) ≥ 0.9
    end

    @testset "Breaks: prefix permanents, the change-point recursion against brute force, and a reversed stem" begin
        # every prefix of a forward pass, against direct permanents, in both modes
        for logspace in (false, true)
            ws = PermanentWorkspace(Float64, 3, 12; logspace)
            W = randn(Xoshiro(4), 12, 7)
            ws.w[1:12, :] .= W
            pre = prefix_logperms!(zeros(12), ws, 12)
            ws2 = PermanentWorkspace(Float64, 3, 12; logspace = true)
            direct = map(1:12) do i
                ws2.w[1:i, :] .= W[1:i, :]
                forward!(ws2, i)
            end
            @test pre ≈ direct atol = 1.0e-10
        end
        # short stems, three canonical angles: log Z(j) exact for single and joint kinds
        ss = simulate_stems([-1.2, -2.6, -3.4, -4.2]; plants = 6, organs = 7, logμκ = 1.8, α = 0.7, outlier = 0.02, rng = Xoshiro(3))
        p3pr = PhyllotaxisPrior(; aα = (70, 50), bα = (30, 50), angles = (Golden_angle, Lucas_angle, π))
        fs = fit_map(PhyllotaxisModelAD(ss.angles, p3pr; ngrid_ϕ = 6, ngrid_κ = 3))
        for kd in ((:orientation,), (:phase,), (:precision,), (:orientation, :angle, :precision))
            sb = spiral_breaks(fs; kind = kd, break_rate = 0.05)
            @test sb.stems[1].logZ ≈ brute_breaks(fs, 1, kd) atol = 1.0e-9
            @test sb.stems[2].logZ[1] ≈ stem_loglik(fs)[2] # no break: the stem's likelihood
            @test all(x -> sum(x.p_breaks) ≈ 1 && x.p_cut[1] == 0 && all(0 .≤ x.p_cut .≤ 1 + 1.0e-12), sb.stems)
        end
        @test_throws ArgumentError spiral_breaks(fs; kind = :twist)
        @test_throws ArgumentError spiral_breaks(fs; kind = :offset)
        @test_throws ArgumentError spiral_breaks(fs; break_rate = 1.5)
        # a stem that reverses half-way (a new phase) among stems that do not: found, and the cut placed
        rng = Xoshiro(11)
        sr = simulate_stems(LAW; plants = 12, organs = 36, logμκ = 2.2, α = 1.0, rng)
        y = copy(sr.angles[1])
        o, ψ, ϕ2 = sr.o[1], sr.ψ[1], 1.3
        θold(j) = sr.ϕ[1] + o * (j - 1) * ψ
        θnew(j) = (j < 19 ? θold(j) : ϕ2 - o * (j - 1) * ψ)
        sr.angles[1] .= [mod2pi(y[p] + θnew(sr.orders[1][p]) - θold(sr.orders[1][p])) for p in 1:36]
        fr = fit_map(PhyllotaxisModelAD(sr.angles; ngrid_ϕ = 16, ngrid_κ = 4))
        br = spiral_breaks(fr)
        @test br.stems[1].logBF > 5 && 1 - br.stems[1].p_breaks[1] > 0.95
        @test sum(br.stems[1].p_cut[18:20]) > 0.8
        @test br.stems[1].p_part[5, 1 + (o == 1)] > 0.95 && br.stems[1].p_part[30, 1 + (o == -1)] > 0.95
        @test all(x -> 1 - x.p_breaks[1] < 0.5, br.stems[2:end])
        @test occursin("break rate", sprint(show, MIME"text/plain"(), br))
        # each kind at once
        @test keys(spiral_breaks(fs; kind = :each, break_rate = 0.05)) == (:phase, :orientation, :angle, :precision)
    end

    @testset "Torsion: coordinates, corrected angles, quadrature, per-stem rates" begin
        @test torsion_coordinate(4) == [0.0, 1.0, 2.0, 3.0]
        @test torsion_coordinate([2.0, 3.0, 5.0]) == [0.0, 2.0, 5.0, 10.0]
        @test TO.torsion_step([0.0, 2.0, 5.0, 10.0]) ≈ 10 / 3
        @test TO.scan_rate(τ -> -(τ - 0.37)^2, 5.0) ≈ 0.37 atol = 0.005 # scan in degrees per mean step of 5 units
        x, fx = TO.golden_section_max(t -> (; v = -(t - 1.234)^2), 0.0, 3.0; tol = 1.0e-8, score = r -> r.v)
        @test x ≈ 1.234 atol = 1.0e-6
        @test fx.v ≈ -(x - 1.234)^2
        # the twist coordinate of the simulators: explicit positions give the default stems
        η = [-0.8, -1.9, -2.4, -3.0]
        pos = [collect(0.0:29.0) for _ in 1:3]
        for law in (η, LAW)
            @test simulate_stems(law; plants = 3, organs = 30, torsion_mean = 2.0, rng = Xoshiro(4)).angles ==
                simulate_stems(law; plants = 3, organs = 30, x = pos, torsion_mean = 2.0, rng = Xoshiro(4)).angles
        end
        @test_throws ArgumentError simulate_stems(η; plants = 3, organs = 30, x = pos[1:2])
        # corrected angles: the twist removed, rate in degrees per unit of the coordinate
        m = twist_model()
        @test corrected_angles(m, 2; ω = 3.0) ≈ TWIST.angles[2] .- deg2rad(3.0) .* torsion_coordinate(40)
        @test TO.torsion_corrected(TWIST.angles[2], torsion_coordinate(40), 0.0) === TWIST.angles[2]
        # adaptive quadrature on a bimodal, sharp log-likelihood against a fine grid, for a few population laws; and
        # the posterior of the rate from the same nodes
        f(ω) = logaddexp(-0.5 * ((ω - 1.3) / 0.08)^2, -2.0 - 0.5 * ((ω - 0.4) / 0.1)^2)
        ω, logw, modes = TO.stem_nodes(f, 5.0) # coordinate step 5: scan ±5 per unit
        @test length(modes) == 2
        grid = range(-3, 5; length = 200_001)
        for (mω, sω) in ((1.0, 0.8), (0.0, 2.0), (2.0, 0.5))
            lg = [f(x) + logpdf(Normal(mω, sω), x) for x in grid]
            exact = logsumexp(lg) + log(step(grid))
            a = TO.node_logterms(ω, logw, f.(ω), mω, sω)
            @test logsumexp(a) ≈ exact atol = 1.0e-3
            μ, sd, mode, mass = TO.node_posterior(ω, a, modes)
            p = exp.(lg .- logsumexp(lg))
            @test μ ≈ sum(p .* grid) atol = 1.0e-3
            @test sd ≈ sqrt(sum(p .* (grid .- μ) .^ 2)) atol = 1.0e-3
            @test isapprox(mode, 1.3; atol = 0.01) && 0.5 < mass < 1 # the sharper, higher mode carries most of the mass
        end
        # per-stem rates at fixed parameters: the best mode of each stem's likelihood finds its twist, and removing it
        # raises the likelihood
        u = prior_mean_parameters(m)
        nodes = AL._place_nodes(m, u)
        τA = [first(md)[1] for md in nodes.modes]
        @test cor(τA, TWIST.torsion) > 0.95
        @test maximum(abs, τA .- TWIST.torsion) < 1.5 # degrees per position
        @test all(stem_loglik(m, u, s; ω = τA[s]) > stem_loglik(m, u, s; ω = 0.0) for s in eachindex(τA))
    end

    @testset "Torsion: rates integrated out; fit, log density, decoding and order posterior" begin
        tp = TorsionPrior(mt = 0.0, st = 10.0, νt = 4.0, τt = 5.0) # per position
        m = twist_model(; torsion = tp)
        u = prior_mean_parameters(m)[1:AL.nmodel(m)]
        nodes = AL._place_nodes(m, u)
        v = [u; 3.5; log(2.5)]
        l0, g, H = AL.twist_vgh(m, nodes, v, 2)
        fv(w) = AL.twist_vgh(m, nodes, w, 0)[1]
        dirs = [normalize(randn(Xoshiro(j), length(v))) for j in 1:2] # two random directions: the gradient as a whole
        @test fd_dir(fv, v, dirs) ≈ [g' * e for e in dirs] rtol = 1.0e-4
        @test issymmetric(round.(H; digits = 6)) && l0 ≈ fv(v)
        @test AL.twist_vgh(m, nodes, v, 1)[2] ≈ g
        # fit: population law and rates recovered
        r = fit_map(m; recenter = 0)
        @test r.converged && r.twist !== nothing && length(r.u) == 9
        @test abs(r.θ.m_ω - mean(TWIST.torsion)) < 1.0 && 0.4 < r.θ.s_ω / std(TWIST.torsion) < 2.5
        # (the posterior mean of each rate: the most probable mode can flip between two of nearly equal mass, as for stem 4)
        @test cor(r.twist.ω_mean, TWIST.torsion) > 0.95
        @test all(r.twist.ω_sd .> 0) && all(0 .< r.twist.mode_mass .≤ 1)
        @test coefnames(r)[(end - 1):end] == ["m_ω", "s_ω"] && coef(r)[end] ≈ r.θ.s_ω
        @test corrected_angles(r, 3) ≈ corrected_angles(m, 3; ω = r.twist.ω[3])
        # the log density of the model is now the objective of the fit: dimension, value and gradient agree
        prob = TransformedLogDensityProblem(m)
        @test LogDensityProblems.dimension(m) == length(r.u)
        @test LogDensityProblems.logdensity(prob, r.u) ≈ r.logposterior
        @test maximum(abs, LogDensityProblems.logdensity_and_gradient(prob, r.u)[2]) < 1.0e-3
        @test loglikelihood(r) ≈ angle_loglikelihood(m, r.u) ≈ sum(stem_loglik(r))
        @test stem_loglik(r)[3] ≈ stem_loglik(m, r.u, 3)
        lc = LogDensityProblems.logdensity(m, coef(r)) # constrained space: the same posterior without the Jacobian
        @test lc ≈ r.logposterior - Bijectors.logabsdetjac(m.binv, r.u[1:AL.nmodel(m)]) - r.u[end]
        d = fit_diagnostics(r)
        @test d.hessian_posdef && d.twist_multimodal isa Vector{Int} && isempty(d.twist_at_edge)
        @test occursin("twist integrated out", sprint(show, MIME"text/plain"(), r))
        # decoding with the twist: the joint MAP recovers the orders and the rates, the posterior agrees with the fit
        dec = decode_orders(r; full = true)
        @test mean(mean(d.σ .== σ) for (d, σ) in zip(dec, TWIST.orders)) > 0.9
        @test cor([d.ω for d in dec], TWIST.torsion) > 0.95
        @test m.nodes[] === r.twist.nodes # the model's nodes are those of the fit (its log density above)
        post = order_posterior(r; nsamples = 400, rng = Xoshiro(4))
        @test [p.ω_mean for p in post] ≈ r.twist.ω_mean rtol = 1.0e-6 # same nodes, same population law, same parameters
        @test all(p -> all(sum(p.ρ; dims = 2) .≈ 1), post)
        emp = [mean(σ[i] - i == d for σ in post[1].samples) for i in 1:40, d in -(m.k):(m.k)]
        @test maximum(abs, emp .- post[1].ρ) < 0.1 # exact samples reproduce the marginals
        @test length(spiral_posterior(r)) == length(TWIST.angles)
        # fixed rates instead (one per stem): the stems decoded at their true twist
        m0 = twist_model()
        u0 = r.u[1:AL.nmodel(m)]
        @test mean(mean(σ .== t) for (σ, t) in zip(decode_orders(m0, u0; ω = TWIST.torsion), TWIST.orders)) > 0.9
        @test [p.ω_mean for p in order_posterior(m0, u0; ω = TWIST.torsion)] == TWIST.torsion
        @test_throws DimensionMismatch decode_orders(m0, u0; ω = [1.0])
    end

    #############################################################################################################
    # step 2: commitment timing
    #############################################################################################################

    @testset "CommitmentTiming: the default law, its sampler, tails and parameters" begin
        rng = Xoshiro(31)
        zg = DELAY_GRID
        @test rand(GaussianExpTails(0.4)) isa Float64 # default RNG
        for f in (GaussianExpTails(0.4), LAW, GaussianExpTails(0.3; πL = 0.2, mL = 0.8, πE = 0.3, mE = 1.5))
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
        # free parameters: the tails present in the starting law, and back
        for (f, n) in ((GaussianExpTails(0.4), 1), (GaussianExpTails(0.4; πL = 0.1), 3), (GaussianExpTails(0.4; πE = 0.1, mE = 2.0), 3), (LAW, 5))
            v = free_parameters(f)
            @test length(v) == n
            @test free_parameters(with_free_parameters(f, v)) ≈ v
            @test delay_logdensity(with_free_parameters(f, v)) ≈ delay_logdensity(f)
        end
        @test reported_parameters(LAW) == (; σc = 0.45) && nshared_parameters(LAW) == 1
        # the named variants: starting laws with the tails they fit, from keyword values or from a law
        @test GaussianExpTails(:core) == GaussianExpTails(0.5)
        @test GaussianExpTails(:late) == GaussianExpTails(0.5; πL = 0.05, mL = 1.0)
        @test GaussianExpTails(:early; σc = 0.3, mE = 2.0) == GaussianExpTails(0.3; πE = 0.05, mE = 2.0)
        @test GaussianExpTails(:both) == GaussianExpTails(0.5; πL = 0.05, mL = 1.0, πE = 0.05, mE = 1.0)
        @test [length(free_parameters(GaussianExpTails(v))) for v in (:core, :late, :early, :both)] == [1, 3, 3, 5]
        @test GaussianExpTails(:both, LAW) == LAW && GaussianExpTails(:late, LAW) == GaussianExpTails(0.45; πL = 0.05, mL = 1.2)
        @test GaussianExpTails(:early, GaussianExpTails(0.4)).πE == 1.0e-6 # a tail absent from the law starts at 1e-6
        @test_throws ArgumentError GaussianExpTails(:tails)
        # 231: the organ of the lowest site sits two positions late; 312: the highest site's organ two positions early
        @test order_diagnostics([[1, 2, 3, 4], [2, 3, 1, 4], [3, 1, 2, 4]]) == (; displaced = 0.5, n231 = 1, n312 = 1)
    end

    @testset "CommitmentTiming: exact order likelihood" begin
        s, zg = 0.43, DELAY_GRID
        lg = [-(z / s)^2 / 2 - log(s * sqrt(2π)) for z in zg]
        # two sites: P(in order) = Φ(1 / (s √2))
        @test exp(loglik_order_exact([1, 2], lg, zg, 1.0)) ≈ cdf(Normal(), 1 / (s * sqrt(2))) atol = 5.0e-4
        # the probabilities of all orders of 4 sites sum to 1 (ties counted as ordered made them sum above 1), tails too
        perms = [[a, b, c, d] for a in 1:4 for b in 1:4 for c in 1:4 for d in 1:4 if allunique((a, b, c, d))]
        @test sum(exp(loglik_order_exact(σ, lg, zg, 1.0)) for σ in perms) ≈ 1 atol = 2.0e-3
        @test sum(exp(loglik_order_exact(σ, GaussianExpTails(0.3; πL = 0.2, mL = 0.8, πE = 0.3, mE = 1.5))) for σ in perms) ≈ 1 atol = 2.0e-3
        # Gumbel delays: the order law is Plackett–Luce, in closed form for any number of sites; an asymmetric law with a
        # long tail, checked on 45 sites
        for (j, β) in enumerate((0.25, 0.4, 0.7))
            orders = [sortperm((1:45) .+ [rand(Xoshiro(100j + i), GumbelDelay(β)) for _ in 1:45]) for i in 1:5]
            @test all(abs(loglik_order_exact(σ, GumbelDelay(β)) - plackett_luce_logP(σ, β)) < 0.01 for σ in orders)
        end
    end

    @testset "CommitmentTiming: the overtaking profile" begin
        # Gaussian delays: q_d = Φ(−d / (σ√2)); exponential delays: the difference is Laplace, q_d = e^{−d/m} / 2
        @test overtaking(GaussianExpTails(0.42)) ≈ [cdf(Normal(), -d / (0.42 * sqrt(2))) for d in 1:4] rtol = 0.01
        @test overtaking(ExpDelay(1.3)) ≈ [exp(-d / 1.3) / 2 for d in 1:4] rtol = 1.0e-6
        f = GaussianExpTails(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)
        rng = Xoshiro(17)
        e = [rand(rng, f) - rand(rng, f) for _ in 1:400_000]
        @test overtaking(f) ≈ [mean(e .> d) for d in 1:4] atol = 0.001
        @test_throws ArgumentError overtaking(f; d = [0.33]) # not a multiple of the grid step
        # model-free from known orders: the fraction of inverted pairs, with clustered standard errors
        orders = simulate_stems(f; plants = 150, organs = 40, rng = Xoshiro(23)).orders
        known = order_samples([(; samples = [σ]) for σ in orders], ηWIDE)
        pf = overtaking_profile(known)
        @test pf.d == 1:4 && size(pf.V) == (4, 4)
        @test pf.q[1] ≈ mean(count(invperm(σ)[i + 1] < invperm(σ)[i] for i in 1:39) / 39 for σ in orders)
        @test all(abs.(pf.q .- overtaking(f)) .< 3 .* pf.se .+ 1.0e-4)
        # the fitted law's profile, with its covariance, agrees with the model-free one
        fit = fit_delay_law(known) # the default: GaussianExpTails(:both)
        @test fit.init == GaussianExpTails(:both)
        inf = delay_law_inference(known, fit)
        @test inf.profile.q ≈ overtaking(fit.law)
        @test [inf.estimate.q1, inf.estimate.q2] ≈ inf.profile.q[1:2]
        @test all(abs.(inf.profile.q[1:3] .- pf.q[1:3]) .< 3 .* pf.se[1:3])
        # comparison: a group against itself (no difference) and against a law with more overtaking (the profile needs
        # only the orders and their weights, not the propensity costs)
        t0 = compare_profiles(pf, pf)
        @test t0.diff == zeros(3) && t0.df == 3
        g = GaussianExpTails(0.42; πL = 0.2, mL = 1.5, πE = 0.2, mE = 1.0)
        pg = overtaking_profile([(; orders = [σ], w = [1.0]) for σ in simulate_stems(g; plants = 150, organs = 40, rng = Xoshiro(24)).orders])
        t1 = compare_profiles(pf, pg)
        @test all(t1.diff .> 0) && t1.p < 1.0e-6
        @test_throws ArgumentError compare_profiles(pf, pg; d = 1:5)
    end

    @testset "CommitmentTiming: re-weighting identity and recovery from known orders" begin
        k = 4
        η = [-3.1, -2.5, -1.9, -0.8, 0.0, -0.8, -1.9, -2.5, -3.1]
        # if the law scored is the propensity model's own order law, every stem term is exactly 0
        post = [(; samples = [simulate_stems(η; plants = 1, organs = 30, rng = Xoshiro(100 * s + j)).orders[1] for j in 1:20]) for s in 1:5]
        smp = order_samples(post, η)
        @test all(sum(s.w) ≈ 1 for s in smp)
        # one workspace per call: stem_terms is threaded over stems
        @test all(abs.(stem_terms(smp, σ -> logprob_permutation(PermanentWorkspace(k, 30), η, σ))) .< 1.0e-10)
        @test_throws ArgumentError order_samples([(; samples = [[6, 1, 2, 3, 4, 5, 7]])], η) # beyond the band
        # known orders: relative_loglik is the exact likelihood minus the propensity terms; the fit recovers the law
        truth = GaussianExpTails(0.42; πL = 0.06, mL = 1.3, πE = 0.05, mE = 1.0)
        orders = simulate_stems(truth; plants = 150, organs = 40, rng = Xoshiro(5)).orders
        known = order_samples([(; samples = [σ]) for σ in orders], ηWIDE)
        @test relative_loglik(known, truth) ≈ sum(loglik_order_exact(σ, truth) for σ in orders) - sum(s.logA[1] for s in known)
        fit = fit_delay_law(known, GaussianExpTails(:both; πL = 0.1, πE = 0.1))
        @test fit.converged && fit.init isa GaussianExpTails && fit.v ≈ free_parameters(fit.law)
        inf = delay_law_inference(known, fit)
        tt = tail_masses(truth)
        @test inf.names == (:σc, :late, :early, :off, :q1, :q2, :q3, :q4)
        @test abs(inf.estimate.σc - truth.σc) < 3 * inf.se.σc + 0.01
        @test abs(inf.estimate.late - tt.late) < 3 * inf.se.late + 0.005
        @test abs(inf.estimate.early - tt.early) < 3 * inf.se.early + 0.005
        @test all(x -> x > 0, values(inf.se))
        # no false tails: from a core-only truth, the two-tail fit gains almost nothing over the core-only fit, and its
        # loosely determined off-time fraction is checked against its own standard error
        kc = known_samples(GaussianExpTails(0.42), 150, 40, 6)
        f0, f2 = fit_delay_law(kc, GaussianExpTails(:core)), fit_delay_law(kc, GaussianExpTails(:both))
        @test f0.law.σc ≈ 0.42 atol = 0.02
        @test 0 ≤ f2.rel_ll - f0.rel_ll < 5 # χ²(4)/2 is below 5 with probability > 0.99
        i2 = delay_law_inference(kc, f2)
        @test abs(i2.estimate.off - tail_masses(GaussianExpTails(0.42)).off) < 3 * i2.se.off
        # one-sided variants: the late fit and the early fit each gain nothing over the core (no tails in the truth)
        fl, fe = fit_delay_law(kc, GaussianExpTails(:late)), fit_delay_law(kc, GaussianExpTails(:early))
        @test fl.law.πE == 0 && fe.law.πL == 0 && fe.converged
        @test 0 ≤ fe.rel_ll - f0.rel_ll < 5 && fe.rel_ll ≤ f2.rel_ll + 1.0e-6
    end

    @testset "CommitmentTiming: exact derivatives and joint fits of groups" begin
        # the gradient of the summed terms (ForwardDiff through the density and the recursion) against central differences
        smp = known_samples(GaussianExpTails(0.42; πL = 0.06, mL = 1.3, πE = 0.05, mE = 1.0), 20, 35, 3)
        obj(v) = relative_loglik(smp, with_free_parameters(LAW, v))
        v = [log(0.4), -2.0, 0.3, log(1.2), 0.1]
        @test ForwardDiff.gradient(obj, v) ≈ fd_dir(obj, v, [1:5 .== i for i in 1:5]) rtol = 1.0e-5
        # a law with -Inf outside its support: its derivative passes through the grid density
        ke = known_samples(ExpDelay(0.8), 10, 20, 1)
        oe(v) = relative_loglik(ke, with_free_parameters(ExpDelay(1.0), v))
        @test ForwardDiff.derivative(x -> oe([x]), log(0.8)) ≈ only(fd_dir(oe, [log(0.8)], [[1.0]])) rtol = 1.0e-5
        # two groups with the same core and different tails: the shared-core fit recovers the core and each group's
        # off-time fraction, and the likelihood ratio does not reject the shared core
        fA, fB = GaussianExpTails(0.42; πL = 0.04, mL = 1.2, πE = 0.04, mE = 1.0), GaussianExpTails(0.42; πL = 0.15, mL = 1.2, πE = 0.12, mE = 1.0)
        groups = [known_samples(fA, 100, 35, 11), known_samples(fB, 100, 35, 12)]
        t = shared_core_test(groups) # the default: GaussianExpTails(:both)
        @test t.shared.converged && length(t.shared.laws) == 2 && t.shared.shared_core
        @test t.shared.laws[1].σc == t.shared.laws[2].σc
        @test t.shared.laws[1].σc ≈ 0.42 atol = 0.05
        @test t.df == 1 && t.lr ≥ 0 && t.p > 0.001
        @test t.shared.rel_ll ≈ sum(t.shared.rel_ll_groups)
        @test sum(f.rel_ll for f in t.separate) ≥ t.shared.rel_ll - 1.0e-6 # separate laws nest the shared one
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

    @testset "CommitmentTiming: delay laws defined by the user" begin
        # a Gaussian written as a Student t of huge ν: nothing but a density, scored like the default core
        g = GaussianExpTails(0.42)
        @test delay_logdensity(StudentDelay(0.42, 1.0e12)) ≈ delay_logdensity(g) rtol = 1.0e-6
        kg = known_samples(g, 40, 30, 51)
        @test relative_loglik(kg, StudentDelay(0.42, 1.0e12)) ≈ relative_loglik(kg, g) rtol = 1.0e-6
        # the Student t: fitted, its parameters reported, its profile consistent
        kt = known_samples(LAW, 40, 30, 53)
        st = fit_delay_law(kt, StudentDelay(0.5, 5.0))
        @test st.converged && st.law isa StudentDelay && st.init == StudentDelay(0.5, 5.0)
        it = delay_law_inference(kt, st)
        @test it.names == (:s, :ν, :late, :early, :off, :q1, :q2, :q3, :q4)
        @test it.estimate.s ≈ st.law.s && it.se.s > 0
        @test it.profile.q ≈ overtaking(st.law)
        @test it.estimate.late ≈ it.estimate.early # a symmetric law
        # one parameter, none reported; the Gumbel fit is close to the default core (the truth has light tails)
        gu = fit_delay_law(kg, GumbelDelay(0.5))
        @test gu.converged && delay_law_inference(kg, gu).names == (:β, :late, :early, :off, :q1, :q2, :q3, :q4)
        @test PhyllotaxisFold._names(ExpDelay(1.0)) == (:late, :early, :off, :q1, :q2, :q3, :q4)
        # joint fits need shared parameters: the Student t shares its scale, the Gumbel law shares nothing
        @test_throws ArgumentError fit_delay_law([kg, kt], GumbelDelay(0.5))
        sj = fit_delay_law([kg, kt], StudentDelay(0.5, 5.0))
        @test sj.converged && sj.laws[1].s == sj.laws[2].s
        @test sj.laws[1].ν > sj.laws[2].ν # the light-tailed group is the more Gaussian
        # simulation: a law's own sampler, or the fallback that draws from the density on the grid
        t = StudentDelay(0.4, 3.0)
        rng = Xoshiro(52)
        e = [rand(rng, t) for _ in 1:5_000]
        @test mean(e .> 1) ≈ tail_masses(t).late atol = 0.015
        @test mean(e .< -1) ≈ tail_masses(t).early atol = 0.015
        @test length(simulate_stems(t; plants = 2, organs = 10, rng).orders) == 2
        # a density that has not decayed at the edge of the grid is flagged
        @test_logs (:warn, r"edge of the grid") match_mode = :any PhyllotaxisFold._check_grid(StudentDelay(0.4, 0.3), DELAY_GRID)
        # a law's own grid (delay_grid) is the default of every function and of the fallback sampler
        @test delay_grid(t) === DELAY_GRID
        w = WideStudent(0.4, 3.0)
        @test relative_loglik(kt, w) ≈ relative_loglik(kt, t; grid = WIDE_GRID)
        @test overtaking(w) ≈ overtaking(t; grid = WIDE_GRID)
        @test loglik_order_exact([2, 1, 3], w) ≈ loglik_order_exact([2, 1, 3], t; grid = WIDE_GRID)
        wh = WideStudent(0.4, 0.3) # half its draws beyond ±30: cut off on DELAY_GRID, not on its own grid
        rng = Xoshiro(53)
        @test maximum(abs, (rand(rng, StudentDelay(0.4, 0.3)) for _ in 1:2_000)) < 30.1
        @test maximum(abs, (rand(rng, wh) for _ in 1:2_000)) > 30.1
        fw = fit_delay_law(kt, w)
        @test fw.grid === WIDE_GRID && fw.law isa WideStudent
        @test delay_law_inference(kt, fw).estimate == delay_law_inference(kt, fw; grid = WIDE_GRID).estimate
    end

    @testset "CommitmentTiming: the whole pipeline on simulated stems (angles → orders → delay law)" begin
        # angles from a known delay law; fit the propensity model; exact posterior order samples; delay law fitted through
        # the re-weighting: the tail masses must come back
        truth = GaussianExpTails(0.42; πL = 0.08, mL = 1.2, πE = 0.08, mE = 1.0)
        sim = simulate_stems(truth; plants = 40, organs = 35, logμκ = 1.9, σκ = 0.15, rng = Xoshiro(41))
        m = PhyllotaxisModelAD(sim.angles; ngrid_ϕ = 8, ngrid_κ = 4, outlier = false)
        r = fit_map(m)
        post = order_posterior(r; nsamples = 60, rng = Xoshiro(42))
        smp = order_samples(post, r.θ.η)
        fit = fit_delay_law(smp, GaussianExpTails(:both))
        inf = delay_law_inference(smp, fit)
        tt = tail_masses(truth)
        @test abs(inf.estimate.off - tt.off) < 3 * inf.se.off + 0.01
        @test abs(inf.estimate.σc - truth.σc) < 3 * inf.se.σc + 0.03
        # the true law scores close to the fitted one (no large optimism)
        @test 0 ≤ fit.rel_ll - relative_loglik(smp, truth) < 10
    end

    #############################################################################################################
    # Plastochron: delays in hours
    #############################################################################################################

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
        plants = simulate_counts(rng, p, 30)
        foreach(pl -> pl.c[5] = NaN, plants) # a day without record
        fit = fit_flowering([pl.c for pl in plants])
        @test fit.converged
        @test all(abs.((fit.p[1:4] .- p[1:4]) ./ fit.se[1:4]) .< 3)
        @test fit.se ≈ sqrt.(diag(fit.V))
        # a non-finite start: outbid by the other starts, an error alone (and the same in the delay-law fits)
        nan_p0, good_p0 = fill(NaN, 5), fit.p
        @test fit_flowering([pl.c for pl in plants]; starts = [nan_p0, good_p0]).nll ≈ fit.nll
        @test_throws ErrorException fit_flowering([pl.c for pl in plants]; starts = [nan_p0])
        @test_throws ErrorException PhyllotaxisFold._newton(v -> NaN * sum(v), [1.0, 2.0])
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
        J = fd_dir(Th, fit.p, [1:5 .== i for i in 1:5]; h = 1.0e-6)
        @test T.se ≈ sqrt(J' * fit.V * J) rtol = 1.0e-4
        h = in_hours(0.4, 0.05, T)
        @test h.estimate ≈ 0.4 * T.T && h.se ≈ hypot(T.T * 0.05, 0.4 * T.se)
    end

    @testset "Manual and README: the code examples run as written" begin
        # every function a block calls exists (or is defined in the blocks of its page)
        pages = ["README.md"; filter(endswith(".md"), readdir(joinpath(@__DIR__, "..", "docs", "src")))]
        for page in pages
            code = join(doc_blocks(page), "\n")
            defined = Set([m.captures[1] for m in eachmatch(r"(?:function|struct)\s+([A-Za-z_][\w!]*)", code)])
            union!(defined, [split(m.captures[1], '.')[end] for m in eachmatch(r"^\s*([A-Za-z_][\w!.]*)(?:\{[^}]*\})?\([^=\n]*\)\s*=[^=]"m, code)])
            called = unique([m.captures[1] for m in eachmatch(r"(?<![\w.:])([A-Za-z_\p{Greek}][\w!]*)\(", code)])
            unknown = filter(n -> !(n in defined) && !any(M -> isdefined(M, Symbol(n)), (PhyllotaxisFold, Base, Statistics, Random)), called)
            @test isempty(unknown) || (page, unknown) == ()
        end
        # small stems of the running example, with the stem length (mm) below each organ
        tiny(seed; S = 8, n = 25) = (
            x = [collect(range(0.0, 120.0; length = n)) for _ in 1:S];
            (; x, angles = simulate_stems(LAW; plants = S, organs = n, x, logμκ = 1.8, torsion_mean = 0.3, torsion_sd = 0.2, rng = Xoshiro(seed)).angles)
        )
        fewer = ["nsamples = 200" => "nsamples = 30"]
        # README: the quick start (fit, decoding, posterior, delay law, profile), then the custom delay law
        readme = doc_sandbox()
        run_blocks!(readme, "README.md", 2:3; subs = [fewer; "plants = 60, organs = 45" => "plants = 8, organs = 25"; "length = 45)) for _ in 1:60]" => "length = 25)) for _ in 1:8]"])
        @test readme.r isa PhyllotaxisFit && readme.fit.law isa GaussianExpTails && readme.st.law isa DelayLaw
        # home page
        t = tiny(2)
        home = doc_sandbox(; t.angles, t.x)
        run_blocks!(home, "index.md", 1:1; subs = fewer)
        @test home.prof.d == 1:4 && home.fit.converged
        # the worked analysis, both groups and the hours
        t1, t2 = tiny(3), tiny(4)
        counts = [pl.c for pl in simulate_counts(Xoshiro(5), [0.1, log(2.5), log(0.6), log(14.0), log(0.2)], 20)]
        work = doc_sandbox(;
            angles_wt = t1.angles, x_wt = t1.x, angles_mutant = t2.angles, x_mutant = t2.x,
            smp_wt = known_samples(LAW, 30, 25, 6), smp_mutant = known_samples(LAW, 30, 25, 12), counts_wt = counts
        )
        run_blocks!(work, "workflow.md", 1:7; subs = fewer)
        @test work.cmp.df == 3 && isfinite(work.inf2.Δoff[1].estimate)
        # custom delay laws: the Student t and the Gumbel law, fitted and simulated
        laws = doc_sandbox(; smp = known_samples(LAW, 30, 25, 8), smp_wt = known_samples(LAW, 30, 25, 9), smp_mutant = known_samples(LAW, 30, 25, 10))
        run_blocks!(laws, "custom_laws.md", 1:3)
        @test laws.st.converged && rand(Xoshiro(1), laws.GumbelDelay(0.4)) isa Float64
        # simulating stems, and the posterior predictive check on a small fit
        sim = doc_sandbox(; m = PhyllotaxisModelAD(tiny(11).angles; ngrid_ϕ = 8, ngrid_κ = 4))
        run_blocks!(sim, "simulation.md", 1:4; subs = ["for _ in 1:200]" => "for _ in 1:3]"])
        @test length(sim.sim.angles) == 70 && 0 ≤ sim.observed ≤ 1 && length(sim.replicated) == 3
        # canonical angles: the priors, a noble angle, and the example at 180°
        ang = doc_sandbox()
        run_blocks!(ang, "canonical_angles.md", 1:3; subs = ["plants = 40, organs = 40" => "plants = 8, organs = 25"])
        @test ang.prior isa PhyllotaxisPrior{4, 2} && ang.golden_only isa PhyllotaxisPrior{4, 0} && length(ang.p180) == 8
        # breaks of the spiral: two groups fitted and scanned
        t1, t2 = tiny(13), tiny(14)
        brk = doc_sandbox(; angles_wt = t1.angles, x_wt = t1.x, angles_mutant = t2.angles, x_mutant = t2.x)
        run_blocks!(brk, "breaks.md", 1:1)
        @test length(brk.p_any) == 8 && brk.b_mutant isa SpiralBreaks
        # the numbers quoted on the timing page
        tim = doc_sandbox()
        run_blocks!(tim, "timing.md", 1:1)
        @test tim.P([2, 1], tim.gauss) ≈ 0.046 atol = 5.0e-4
        @test all(isapprox.(tim.odds(tim.gauss), (0.0508, 0.0487); atol = 5.0e-4))
        @test all(isapprox.(tim.odds(tim.ex), (0.0913, 0.0941); atol = 5.0e-4))
        @test tim.P([3, 1, 2], tim.gauss) + tim.P([2, 3, 1], tim.gauss) ≈ 0.0003 atol = 5.0e-5
        @test tim.P([3, 1, 2], tim.ex) + tim.P([2, 3, 1], tim.ex) ≈ 0.024 atol = 5.0e-4
    end

    @testset "Code quality: Aqua and explicit imports" begin
        # ambiguities in the package's own methods only (its dependencies have their own)
        Aqua.test_all(PhyllotaxisFold; ambiguities = (; recursive = false))
        @test check_no_stale_explicit_imports(PhyllotaxisFold) === nothing
    end

end # testset PhyllotaxisFold
