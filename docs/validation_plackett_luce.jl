# Manual §5.5 checked against an exact answer. With Gumbel delays, e_i = −β G_i (G_i standard Gumbel), the order of the
# commitment times τ_i = i + e_i is Plackett–Luce: the organ at height p comes from site j with probability
# w_j / Σ_{l unused} w_l, w_j = e^{−j/β}. In the banded recursion of manual §3 the unused sites are known from the state,
# so this law needs one extra factor per (height, state) and its sum over all banded orders is exact: p(y | f) is then
# computed with the same nuisance grid and twist nodes as the propensity model, and the re-weighting estimate Δℓ_s(f)
# can be compared with the exact log p(y | f) − log p(y | A), stem by stem. Simulated stems only.
# Run in the docs environment with threads, after include("docs/make_validation_figures.jl"):
#   pl_checks(); PL = run_plackett_luce(); pl_summary(PL); fig_v5(PL)
# Results cached in docs/plackett_luce.jls.
using LogExpFunctions
const PF = PhyllotaxisFold
const AL = PF
using PhyllotaxisFold.Permanents: PermanentWorkspace, PermutationGraph, toeplitz_weights!, forward!
using PhyllotaxisFold: logbesseli

const PL_FILE = joinpath(@__DIR__, "plackett_luce.jls")
PLRES = isfile(PL_FILE) ? deserialize(PL_FILE) : Dict{Any, Any}()

# log P(σ) under Plackett–Luce with weights e^{−j/β} (closed form, all orders, not only banded ones)
function pl_logP(σ::AbstractVector{Int}, β::Real)
    n = length(σ)
    used = falses(n)
    lp = 0.0
    for p in 1:n
        lp += -σ[p] / β - logsumexp(-l / β for l in 1:n if !used[l])
        used[σ[p]] = true
    end
    return lp
end

# the delay law, defined as in the manual page "Custom delay laws": e = −β G, G a standard Gumbel variable (a site
# commits when the first of its cells does). Its order law is the Plackett–Luce law of pl_logP
struct GumbelDelay{T <: Real} <: DelayLaw
    β::T
end
PF.delay_logpdf(f::GumbelDelay, z) = z / f.β - exp(z / f.β)
PF.free_parameters(f::GumbelDelay) = [log(f.β)]
PF.with_free_parameters(::GumbelDelay, v) = GumbelDelay(exp(v[1]))
PF.reported_parameters(f::GumbelDelay) = (; β = f.β)
Base.rand(rng::AbstractRNG, f::GumbelDelay) = f.β * log(-log(rand(rng)))

# the order of n sites with delays f: τ_i = i + e_i
pl_order(rng, n, f::DelayLaw) = sortperm((1:n) .+ [rand(rng, f) for _ in 1:n])

# the state masks of the banded graph, in the order of PermutationGraph (bit b ↔ column i − k + b, set = used)
pl_masks(k) = [m for m in 0:((1 << (2k)) - 1) if count_ones(m) == k]

# −log D(i, s): the Plackett–Luce normaliser of the unused sites at height i in state s, relative to e^{−i/β}. Unused
# sites are the free columns of the window {i − k, …, i + k − 1} inside 1..n and every column from i + k to n.
function pl_state_factors(k, n, β)
    masks = pl_masks(k)
    sf = fill(-Inf, length(masks), n)
    for i in 1:n, (s, m) in enumerate(masks)
        terms = [-(b - k) / β for b in 0:(2k - 1) if (m >> b) & 1 == 0 && 1 ≤ i - k + b ≤ n]
        append!(terms, (-t / β for t in k:(n - i)))
        isempty(terms) || (sf[s, i] = -logsumexp(terms))
    end
    return sf
end

# forward pass in log space with a log-factor per (state, height) on the edges leaving it
function pl_forward!(ws::PermanentWorkspace, sf::AbstractMatrix, n::Int)
    (; g, F, w) = ws
    (; k, M, edge_ptr, edge_next, edge_d) = g
    fill!(view(F, :, 1:(n + 1)), -Inf)
    F[g.start_s, 1] = 0.0
    @inbounds for i in 1:n, s in 1:M
        val = F[s, i] + sf[s, i]
        isfinite(val) || continue
        for e in edge_ptr[s]:(edge_ptr[s + 1] - 1)
            d = edge_d[e]
            (1 ≤ i + d ≤ n) || continue
            nxt = edge_next[e]
            F[nxt, i + 1] = logaddexp(F[nxt, i + 1], val + w[i, d + k + 1])
        end
    end
    return F[g.end_s, n + 1]
end

# log p(y | Plackett–Luce) of one stem at fixed twist: the sum over the nuisance grid of `stem_loglik!` (pass 1), with the
# costs η replaced by the tilt −d/β, the state factors added, and no normaliser (the law is normalised over all orders)
function pl_stem_loglik!(fws, grid, y, β, sf, logμκ, σκ, α; outlier = nothing)
    (; k, ngrid_ϕ, ngrid_κ, κ_nodes, κ_logweights) = grid
    n = length(y)
    AL._fill_cosines!(fws, grid, y, α)
    tilt = [-d / β for d in -k:k]
    log_Zq = map(1:ngrid_κ) do q
        κ = exp(logμκ + σκ * κ_nodes[q])
        ot = AL._outlier_terms(outlier, κ)
        lz = map(1:(4 * ngrid_ϕ)) do c
            AL._fill_weights!(fws.perm.w, tilt, fws.C, c, κ, n, k, ot)
            pl_forward!(fws.perm, sf, n) + n * κ + fws.logp_c[c]
        end
        κ_logweights[q] + logsumexp(lz) - n * logbesseli(0, κ)
    end
    return logsumexp(log_Zq) - log(2 * ngrid_ϕ) - n * log(2π)
end

# per stem, the exact log p(y | f) with the twist integrated on the nodes of the propensity fit `r`, at its population
# law, and the same for the propensity model itself (log p(y | A)); `law = nothing` gives the latter
function exact_stem_logliks(m, r; β = nothing)
    G = length(r.u) - 2
    u, mω, sω = r.u[1:G], r.u[G + 1], exp(r.u[G + 2])
    nu = AL.stem_params(m, u)
    grid = m.grid
    nodes = r.twist.nodes
    return fetch.(
        map(eachindex(m.data)) do s
            Threads.@spawn begin
                n = length(m.data[s])
                fws = AL.StemWorkspace(Float64, m.graph, n, m.ngrid_ϕ, m.ngrid_κ)
                if β === nothing
                    logsumexp(AL._node_terms(m, nu, r.u, s, nodes, fws))
                else
                    sf = pl_state_factors(m.k, n, β)
                    ℓ = [
                        pl_stem_loglik!(fws, grid, corrected_angles(m, s; ω), β, sf, nu.logμκ, nu.σκ, nu.α; nu.outlier)
                            for ω in nodes.ω[s]
                    ]
                    logsumexp(PF.node_logterms(nodes.ω[s], nodes.logw[s], ℓ, mω, sω))
                end
            end
        end
    )
end

# every permutation of 1..n (small n only)
allperms(n) = n == 1 ? [[1]] : [vcat(σ[1:(i - 1)], n, σ[i:end]) for σ in allperms(n - 1) for i in 1:n]

# the machinery against brute force: (a) the banded sum of Plackett–Luce equals the sum of the closed form over the
# banded orders, and is 1 when the band holds every order; (b) pass 1 rebuilt here reproduces `stem_loglik!`; (c) the
# exact order likelihood of the package (a recursion on the delay grid, as for any DelayLaw) matches the closed form
function pl_checks(; β = 0.4)
    out = Dict{Symbol, Float64}()
    orders = [pl_order(Xoshiro(j), 45, GumbelDelay(β)) for j in 1:20]
    out[:grid_vs_closed_form] = maximum(abs(loglik_order_exact(σ, GumbelDelay(β)) - pl_logP(σ, β)) for σ in orders)
    for (n, k) in ((7, 2), (5, 4))
        ws = PermanentWorkspace(k, n; logspace = true)
        toeplitz_weights!(ws, [-d / β for d in -k:k], n)
        dp = pl_forward!(ws, pl_state_factors(k, n, β), n)
        brute = logsumexp(pl_logP(collect(σ), β) for σ in allperms(n) if all(abs(σ[p] - p) ≤ k for p in 1:n))
        out[Symbol("banded_n$(n)_k$(k)")] = dp - brute
        k == n - 1 && (out[:all_orders_logsum] = dp)
    end
    rng = Xoshiro(7)
    y = mod2pi.(deg2rad(137.5) .* (0:29) .+ 0.3 .* randn(rng, 30))
    ngrid_ϕ, ngrid_κ = 8, 4
    mm = PhyllotaxisModelAD([y]; ngrid_ϕ, ngrid_κ, outlier = false)
    k, grid = mm.k, mm.grid
    η = full_eta(mm, constrained_parameters(mm, prior_mean_parameters(mm)))
    fws = AL.StemWorkspace(Float64, PermutationGraph(k), 30, ngrid_ϕ, ngrid_κ; logspace = true)
    ref = AL.stem_loglik!(fws, grid, y, η, 1.5, 0.2, 0.9, false)
    # same pass with zero state factors and η in place of the tilt, minus the η normaliser
    AL._fill_cosines!(fws, grid, y, 0.9)
    zsf = zeros(length(pl_masks(k)), 30)
    log_Zq = map(1:ngrid_κ) do q
        κ = exp(1.5 + 0.2 * grid.κ_nodes[q])
        lz = map(1:(4ngrid_ϕ)) do c
            AL._fill_weights!(fws.perm.w, η, fws.C, c, κ, 30, k, nothing)
            pl_forward!(fws.perm, zsf, 30) + 30κ + fws.logp_c[c]
        end
        grid.κ_logweights[q] + logsumexp(lz) - 30logbesseli(0, κ)
    end
    toeplitz_weights!(fws.perm, η, 30)
    mine = logsumexp(log_Zq) - log(2ngrid_ϕ) - 30log(2π) - forward!(fws.perm, 30)
    out[:pass1_rebuilt] = mine - ref
    return out
end

# one data set of stems with Gumbel delays (the design of manual §6.1), the propensity model fitted with the twist,
# posterior orders; then, on a grid of β, the exact and the re-weighted log p(y | f) − log p(y | A) per stem
function run_plackett_luce(; β = 0.34, βs = 0.2:0.04:0.6, seed = 5001, nsamples = (200, 2000))
    key = (β, seed)
    haskey(PLRES, key) && return PLRES[key]
    stage("PL $(key): simulating and fitting")
    rng = Xoshiro(seed)
    x = stem_coordinates(rng)
    orders = [pl_order(rng, NORGANS, GumbelDelay(β)) for _ in 1:NSTEMS]
    sims = [AL._simulate_angles(rng, σ, x[s]; KAPPA..., α = 0.95, outlier = 0.0, torsion_mean = TWIST.mean, torsion_sd = TWIST.sd) for (s, σ) in enumerate(orders)]
    m = PhyllotaxisModelAD([z.y for z in sims]; torsion = TPRIOR, torsion_x = x)
    r = fit_map(m)
    θ = r.θ
    stage("PL $(key): exact log p(y | A)")
    ℓA = exact_stem_logliks(m, r)
    smps = map(nsamples) do N
        stage("PL $(key): $(N) posterior orders")
        order_samples(order_posterior(r; nsamples = N, rng = Xoshiro(seed + N)), θ.η)
    end
    rows = map(βs) do b
        stage("PL $(key): β = $(b)")
        exact = exact_stem_logliks(m, r; β = b) .- ℓA
        est = [stem_terms(smp, σ -> pl_logP(σ, b)) for smp in smps]
        (; β = b, exact, est)
    end
    out = (;
        β, seed, nsamples, outofband = count(σ -> any(abs(σ[p] - p) > m.k for p in eachindex(σ)), orders),
        loglikA = sum(ℓA), loglik_fit = r.loglik, rows,
    )
    PLRES[key] = out
    serialize(PL_FILE, PLRES)
    return out
end

# summed Δℓ by β, exact and re-weighted, and the per-stem errors of the re-weighting
function pl_summary(PL)
    return map(PL.rows) do row
        errs = [row.est[i] .- row.exact for i in eachindex(row.est)]
        (;
            row.β, exact = round(sum(row.exact); digits = 2),
            est = Tuple(round(sum(e); digits = 2) for e in row.est),
            max_stem_err = Tuple(round(maximum(abs, e); digits = 3) for e in errs),
            rms_stem_err = Tuple(round(sqrt(mean(abs2, e)); digits = 4) for e in errs),
        )
    end
end

# v5. the re-weighting against the exact likelihood (manual §6.9)
function fig_v5(PL)
    βs = [row.β for row in PL.rows]
    cols = ("#eb6834", "#2a78d6")
    return with_theme(VTHEME) do
        fig = Figure(size = (1450, 450))
        ax = Axis(
            fig[1, 1]; title = "(a) Δℓ summed over $(NSTEMS) stems", xlabel = "Gumbel scale β (plastochron)",
            ylabel = "log p(y | f) − log p(y | A)"
        )
        vlines!(ax, [PL.β]; color = :gray, linestyle = :dash)
        lines!(ax, βs, [sum(row.exact) for row in PL.rows]; color = :black, linewidth = 2.5, label = "exact")
        for (i, N) in enumerate(PL.nsamples)
            scatter!(
                ax, βs, [sum(row.est[i]) for row in PL.rows]; color = cols[i], marker = (i == 1 ? :circle : :xcross),
                markersize = 11, label = "re-weighting, $(N) samples"
            )
        end
        axislegend(ax; position = :cb, framevisible = false)
        ax = Axis(
            fig[1, 2]; title = "(b) Error per stem", xlabel = "Gumbel scale β (plastochron)",
            ylabel = "|re-weighting − exact|", yscale = log10
        )
        vlines!(ax, [PL.β]; color = :gray, linestyle = :dash)
        for (i, N) in enumerate(PL.nsamples)
            errs = [abs.(row.est[i] .- row.exact) for row in PL.rows]
            scatterlines!(ax, βs, [sqrt(mean(abs2, e)) for e in errs]; color = cols[i], label = "$(N) samples, typical (rms)")
            lines!(ax, βs, [maximum(e) for e in errs]; color = cols[i], linestyle = :dash, label = "$(N) samples, worst stem")
        end
        Legend(fig[1, 3], ax; framevisible = false)
        save(joinpath(VFIG, "v5_plackett_luce.png"), fig; px_per_unit = 2)
        fig
    end
end
