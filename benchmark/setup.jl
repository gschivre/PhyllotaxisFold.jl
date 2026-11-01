# Shared fixtures of the performance scripts. Run from this environment:
#   julia --project=benchmark -t auto benchmark/setup.jl     (or include it in a REPL)
# The environment holds the tools (BenchmarkTools, JET, AllocCheck, DispatchDoctor, Profile), not the package's deps.
using PhyllotaxisFold, BenchmarkTools, Random, ForwardDiff, LogDensityProblems
const PF = PhyllotaxisFold
const AL = PF # (the internals, by their qualified names)
const PE = PF.Permanents

# 40 stems of 40 organs, the default grids (32 ϕ × 16 κ), outliers on; and the same stems with a twist (coarse grids)
const LAW = GaussianExpTails(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)
const SIM = simulate_stems(LAW; plants = 40, organs = 40, logμκ = 1.8, rng = Xoshiro(1)) # (the baseline of the 2026-11 rework)
const MB = PhyllotaxisModelAD(SIM.angles; outlier = true)
const UB = prior_mean_parameters(MB)
const SIMT = simulate_stems(LAW; plants = 40, organs = 40, logμκ = 1.8, torsion_mean = 2.0, torsion_sd = 1.5, rng = Xoshiro(2))
const MT = PhyllotaxisModelAD(SIMT.angles; ngrid_ϕ = 16, ngrid_κ = 8, outlier = true, torsion = TorsionPrior(st = 5.0))

# one stem: its workspace, parameters and a filled weight table
const P1 = AL.stem_params(MB, UB)
const FWS = AL._workspace(MB, Float64)
const N1 = length(MB.data[1])
AL._stem_ll!(MB, P1, 1, 0.0, FWS)
AL._fill_weights!(FWS.perm.w, P1.η, FWS.C, 5, exp(P1.logμκ), N1, MB.k, AL._outlier_terms(P1.outlier, exp(P1.logμκ)))
const W1 = copy(FWS.perm.w)

# median time (ms), memory (MB) and allocations of a benchmark
summ(b) = (time_ms = median(b).time / 1.0e6, MB = b.memory / 1.0e6, allocs = b.allocs)
