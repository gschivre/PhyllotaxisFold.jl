# Accuracy of the integrals over the unknowns of each stem (manual §2.6, the three "derivation and accuracy" boxes):
# the offset ϕ, then the precision log κ, then the twist ω.
#
# All three use the 40 stems of the running example (docs/make_figures.jl, `example_data`) at their fitted parameters.
# The ϕ and κ checks evaluate each stem at its most likely twist rate (the centre of the highest peak of its likelihood
# in ω), so that only the rule under test changes.
#
# 1. ϕ (rectangle rule, `ngrid_ϕ`): 16, 32, 64 and 128 nodes against 256, with 64 κ nodes in both (only the ϕ rule
#    differs), at the fitted σ_κ and at 0.3, 0.5, 0.8.
# 2. log κ (Gauss–Hermite on the population law, `ngrid_κ`): 8, 16 and 32 nodes against 64, with 128 ϕ nodes in the
#    rule and 256 in the reference (so that the ϕ error, ~1e-9, is negligible), at the same four σ_κ.
# 3. ω (adaptive rule of the package, `stem_nodes`): each stem's integral ∫ L_s(ω) N(ω | m_ω, s_ω²) dω by brute force
#    (fine grid, step 0.005 °/mm, over every region where the integrand matters), by the adaptive rule with 5, 9
#    (default) and 15 nodes per mode, and by a fixed Gauss–Hermite rule on the population law (the rule used for
#    log κ) with 32 and 128 nodes; for the fitted population law and three others (sd doubled, sd halved, mean + 0.5
#    °/mm). The adaptive nodes are placed once, at the fitted parameters, and reused for the four laws, as in the fit.
#    It also records why a fixed rule fails: the width of each stem's peak (sd of its best mode) against s_ω, and the
#    number of modes.
#
# Every stage is cached in docs/integration_accuracy_study.jls (the fit, then one entry per check) and is computed only
# when missing: `integration_accuracy_study(; refresh = [:phi])` recomputes the named stages (:fit, :phi, :kappa,
# :twist). Progress goes to docs/integration_accuracy_study.log. Timings on 18 threads: fit ≈ 6 min, ϕ and κ a few
# minutes, twist ≈ 10 min.
#
# Run in the docs environment, with threads:
#   julia -t auto --project=docs docs/integration_accuracy_study.jl
# or, in a REPL: include("docs/integration_accuracy_study.jl"); res = integration_accuracy_study();
#   phi_table(res), kappa_table(res), twist_table(res)

include(joinpath(@__DIR__, "make_figures.jl")) # example_data (guarded: makes no figure when included)
using PhyllotaxisFold: gausshermite

const IA_FILE = joinpath(@__DIR__, "integration_accuracy_study.jls")
const IA_LOG = joinpath(@__DIR__, "integration_accuracy_study.log")

ia_log(msg) = open(io -> println(io, "$(round(time(); digits = 1))  $msg"), IA_LOG, "a")

# per-stem values f(s), threaded over the stems, with a log line per stem
function per_stem(f, S, label)
    out = Vector{Any}(undef, S)
    done = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for s in 1:S
        out[s] = f(s)
        ia_log("$label: stem $s ($(Threads.atomic_add!(done, 1) + 1)/$S)")
    end
    return out
end

#################################################################################################################
# 0. the fit: parameters, and each stem's most likely twist rate
#################################################################################################################

# the fitted parameters as a flat constrained vector [η_1..η_k; log μκ; σκ; α; π_o; m_ω; s_ω], with σκ replaced
function constrained_vector(m, θ; σκ = θ.σκ)
    k = m.k
    return vcat(θ.η[(k + 2):end], θ.logμκ, σκ, θ.α..., (m.outlier ? [θ.outlier] : Float64[])..., θ.m_ω, θ.s_ω)
end
unconstrained(m, θ; σκ = θ.σκ) = PF._unconstrained(m, constrained_vector(m, θ; σκ))

function stage_fit!(cache)
    x, sim, m = example_data()
    ia_log("fit")
    r = fit_map(m)
    nodes = PF._place_nodes(m, r.u)
    cache[:fit] = (; r.θ, ω = [first(md)[1] for md in nodes.modes]) # centre of the highest peak of each stem
    return cache
end

#################################################################################################################
# 1–2. the offset ϕ and the precision log κ: per-stem log-likelihood on a grid against a fine reference
#################################################################################################################

const IA_SIGMAS = (:fitted, 0.3, 0.5, 0.8)
const IA_PHI = (16, 32, 64, 128)
const IA_KAPPA = (8, 16, 32)
const IA_REF = (256, 64) # ϕ × κ nodes of the reference

# per-stem log-likelihoods of the running example on a ϕ × κ grid, at the fitted parameters with σκ replaced
function grid_logliks(nϕ, nκ, σκ, fit)
    x, sim, _ = example_data()
    m = PhyllotaxisModelAD(sim.angles; torsion = EX_TPRIOR, torsion_x = x, ngrid_ϕ = nϕ, ngrid_κ = nκ)
    u = unconstrained(m, fit.θ; σκ = σκ === :fitted ? fit.θ.σκ : σκ)
    return Float64.(per_stem(s -> stem_loglik(m, u, s; ω = fit.ω[s]), length(sim.angles), "$(nϕ)ϕ × $(nκ)κ, σκ $σκ"))
end

function stage_grids!(cache, key)
    fit = cache[:fit]
    out = get(cache, key, Dict{Any, Any}())
    for σκ in IA_SIGMAS
        ref = get!(() -> grid_logliks(IA_REF..., σκ, fit), cache, (:reference, σκ)) # shared by ϕ and κ
        grids = key === :phi ? [(n, IA_REF[2]) for n in IA_PHI] : [(128, n) for n in IA_KAPPA]
        for g in grids
            haskey(out, (g, σκ)) && continue
            out[(g, σκ)] = grid_logliks(g..., σκ, fit) .- ref
            cache[key] = out
            serialize(IA_FILE, cache) # keep partial results
        end
    end
    cache[key] = out
    return cache
end

grid_summary(err) = (median = median(abs.(err)), worst = maximum(abs.(err)))

function phi_table(res)
    out = res[:phi]
    return [(; nϕ = n, [Symbol("σκ_", σ) => grid_summary(out[((n, IA_REF[2]), σ)]) for σ in IA_SIGMAS]...) for n in IA_PHI]
end

function kappa_table(res)
    out = res[:kappa]
    σs = [σ === :fitted ? round(res[:fit].θ.σκ; digits = 2) : σ for σ in IA_SIGMAS]
    return [(; σκ = σv, [Symbol("nκ_", n) => grid_summary(out[((128, n), σ)]) for n in IA_KAPPA]...) for (σ, σv) in zip(IA_SIGMAS, σs)]
end

#################################################################################################################
# 3. the twist ω: adaptive rule and fixed rule against brute force
#################################################################################################################

const TQ_NNODES = (5, 9, 15)   # adaptive: nodes per mode
const TQ_GH = (32, 128)        # fixed: Gauss–Hermite nodes on the population law
const TQ_STEP = 0.005          # brute force: grid step (°/mm)
const TQ_COARSE = 0.05         # brute force: coarse step that locates the regions to refine
const TQ_KEEP = 40.0           # brute force: refine where the log integrand is within this of its maximum

tq_laws(mω, sω) = [(:fitted, mω, sω), (:sd_doubled, mω, 2sω), (:sd_halved, mω, sω / 2), (:mean_shifted, mω + 0.5, sω)]

# log ∫ exp(ℓ(ω)) N(ω | m, s) dω by brute force; ℓ is evaluated on the fine grid once and shared by the laws
function brute_force(ℓ, laws)
    lo = minimum(m - 8s for (_, m, s) in laws)
    hi = maximum(m + 8s for (_, m, s) in laws)
    coarse = collect(lo:TQ_COARSE:hi)
    ℓc = ℓ.(coarse)
    keep = falses(length(coarse))
    for (_, m, s) in laws
        a = ℓc .+ logpdf.(Normal(m, s), coarse)
        keep .|= a .≥ maximum(a) - TQ_KEEP
    end
    # fine grid on the kept coarse cells (each widened by one coarse step on both sides), on a common lattice
    fine = Set{Int}()
    for i in findall(keep)
        a, b = coarse[i] - TQ_COARSE, coarse[i] + TQ_COARSE
        union!(fine, ceil(Int, a / TQ_STEP):floor(Int, b / TQ_STEP))
    end
    ω = sort!(collect(fine)) .* TQ_STEP
    ℓf = ℓ.(ω)
    return [logsumexp(ℓf .+ logpdf.(Normal(m, s), ω)) + log(TQ_STEP) for (_, m, s) in laws], length(ω)
end

# fixed rule: Gauss–Hermite nodes on the population law (normalised: N(0, 1) nodes, weights summing to 1)
function fixed_rule(ℓ, m, s, K)
    z, w = gausshermite(K; normalize = true)
    return logsumexp(log.(w) .+ ℓ.(m .+ s .* z))
end

function twist_stem(m, u, s, laws)
    ℓ(ω) = stem_loglik(m, u, s; ω)
    step = PF.torsion_step(m.torsion_x[s])
    ref, ngrid = brute_force(ℓ, laws)
    adaptive = Dict{Int, Vector{Float64}}()
    modes = nothing
    for nn in TQ_NNODES
        ω, logw, md = PF.stem_nodes(ℓ, step; nnodes = nn)
        nn == 9 && (modes = md)
        ℓn = ℓ.(ω)
        adaptive[nn] = [logsumexp(PF.node_logterms(ω, logw, ℓn, mω, sω)) for (_, mω, sω) in laws]
    end
    fixed = Dict(K => [fixed_rule(ℓ, mω, sω, K) for (_, mω, sω) in laws] for K in TQ_GH)
    return (; s, ref, ngrid, adaptive, fixed, modes)
end

function stage_twist!(cache)
    θ = cache[:fit].θ
    x, sim, m = example_data()
    u = unconstrained(m, θ)
    laws = tq_laws(θ.m_ω, θ.s_ω)
    stems = per_stem(s -> twist_stem(m, u, s, laws), length(sim.angles), "twist")
    cache[:twist] = Dict{Any, Any}(:laws => laws, :stems => stems)
    return cache
end

# median and worst |error| per stem (log-likelihood units), per rule (ranges over the four laws); the widths of the peaks
function twist_table(res)
    laws, stems = res[:twist][:laws], res[:twist][:stems]
    rules = vcat([("adaptive, $nn nodes per mode", st -> st.adaptive[nn]) for nn in TQ_NNODES],
        [("Gauss–Hermite on the population law, $K nodes", st -> st.fixed[K]) for K in TQ_GH])
    rows = map(rules) do (name, get)
        err = [abs.(get(st) .- st.ref) for st in stems] # stems × laws
        med = [median(e[l] for e in err) for l in eachindex(laws)]
        worst = [maximum(e[l] for e in err) for l in eachindex(laws)]
        (; rule = name, median = extrema(med), worst = extrema(worst))
    end
    nmodes = [length(st.modes) for st in stems]
    width = [first(st.modes)[2] for st in stems] # sd of the best mode, °/mm
    sω = res[:fit].θ.s_ω
    peaks = (;
        s_ω = sω, width_median = median(width), width_range = extrema(width), ratio_median = sω / median(width),
        modes = [(k, count(==(k), nmodes)) for k in sort!(unique(nmodes))], grid_points = extrema(st.ngrid for st in stems),
    )
    return (; laws = first.(laws), rows, peaks)
end

#################################################################################################################
# driver
#################################################################################################################

function integration_accuracy_study(; refresh = Symbol[])
    cache = isfile(IA_FILE) ? deserialize(IA_FILE) : Dict{Any, Any}()
    if :fit in refresh # new parameters: every check is stale
        empty!(cache)
    end
    for key in refresh
        key === :fit || delete!(cache, key)
        key === :phi || key === :kappa || continue
        foreach(k -> delete!(cache, k), [k for k in keys(cache) if k isa Tuple && k[1] === :reference])
    end
    for (key, run!) in ((:fit, stage_fit!), (:phi, c -> stage_grids!(c, :phi)), (:kappa, c -> stage_grids!(c, :kappa)),
            (:twist, stage_twist!))
        complete = haskey(cache, key) && (key ∉ (:phi, :kappa) || length(cache[key]) == length(IA_SIGMAS) *
                                          length(key === :phi ? IA_PHI : IA_KAPPA))
        complete && continue
        run!(cache)
        serialize(IA_FILE, cache)
        ia_log("$key done")
    end
    ia_log("all done")
    return cache
end

if abspath(PROGRAM_FILE) == @__FILE__
    res = integration_accuracy_study()
    r3(x) = round(x; sigdigits = 2)
    println("ϕ nodes (vs $(IA_REF[1]); $(IA_REF[2]) κ nodes in both): median, worst per stem")
    foreach(r -> println(r.nϕ, "  ", [(k, r3(v.median), r3(v.worst)) for (k, v) in pairs(r) if k !== :nϕ]), phi_table(res))
    println("κ nodes (vs $(IA_REF[2])): median, worst per stem")
    foreach(r -> println(r.σκ, "  ", [(k, r3(v.median), r3(v.worst)) for (k, v) in pairs(r) if k !== :σκ]), kappa_table(res))
    t = twist_table(res)
    foreach(row -> println(rpad(row.rule, 48), "median ", r3.(row.median), "   worst ", r3.(row.worst)), t.rows)
    println(t.peaks)
end
