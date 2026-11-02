"""
    PhyllotaxisFold

Statistical models of the order of organs along a plant stem, from the absolute azimuths of the organs: the propensity
model (free log-propensity of each displacement) decodes the orders, and a law of commitment delays (each site commits
after an independent random delay) is fitted through its posterior orders. One module, its source files in dependency
order:

| file                  | content                                                                                  |
|-----------------------|------------------------------------------------------------------------------------------|
| `Permanents.jl`       | the submodule `Permanents`: banded permanents (graph, forward / backward, sampling, Viterbi) |
| `Common.jl`           | canonical angles; log I₀ and Bessel ratio (dual-safe), half-t                             |
| `Torsion.jl`          | stem twist: coordinates, prior, per-stem rate scan, adaptive quadrature, Newton MAP       |
| `AngleLikelihood.jl`  | the angle likelihood and its analytic gradient; the fits, `PhyllotaxisFit`; decoding      |
| `PropensityPriors.jl` | log-normal softplus prior on free log-propensities η and its bijectors                    |
| `PropensityModel.jl`  | the propensity model `PhyllotaxisModelAD`, `fit_map`, the fit's accessors and diagnostics |
| `Simulation.jl`       | simulated stems, from parameters or from a fit; divergence angles                         |
| `CommitmentTiming.jl` | commitment delay laws, exact order likelihood, re-weighting of the posterior orders       |
| `Plastochron.jl`      | plastochron by flower rank from daily open-flower counts (delays in hours)                |

The exported names are the interface for an analysis. The building blocks (the banded permanents, the numerical tools,
the internals of the likelihood) are not exported; those listed in the manual's API page are declared `public` (Julia ≥
1.11) and reached by their qualified names, e.g. `PhyllotaxisFold.Permanents.forward!`, `PhyllotaxisFold.logbesseli`.
The method is described in the manual (`docs/src`).
"""
module PhyllotaxisFold

using Random, Statistics, LinearAlgebra, Printf
using Distributions, LogExpFunctions, StatsBase, FastGaussQuadrature, ForwardDiff, Optim
using SpecialFunctions: loggamma, besselix, erfc, erfcx
using ADTypes: AutoForwardDiff
using NLSolversBase: NLSolversBase, TwiceDifferentiable
using LogDensityProblems: LogDensityProblems
using Bijectors: Bijectors
import StatsBase: coef, coefnames, vcov, stderror, coeftable, loglikelihood, dof, nobs

include("Permanents.jl")
using .Permanents: PermutationGraph, PermanentWorkspace, toeplitz_weights!, forward!, forward_linear!, backward!,
    sample_permutation!, map_permutation, logprob_permutation, cycle_lengths, _isneginf, _fullvalue
include("Common.jl")
include("Torsion.jl")
include("AngleLikelihood.jl")
include("PropensityPriors.jl")
include("PropensityModel.jl")
include("Simulation.jl")
include("CommitmentTiming.jl")
include("Plastochron.jl")

# the propensity model: data, prior, fit and its result
export PhyllotaxisModelAD, PhyllotaxisPrior, TorsionPrior, torsion_coordinate, fit_map, PhyllotaxisFit, fit_diagnostics,
    coef, coefnames, vcov, stderror, coeftable, loglikelihood, dof, nobs
# parameters and prior
export constrained_parameters, unconstrained_parameters, prior_mean_parameters, prior_mean, rand_priorparameters,
    prior_CIs, full_eta, log_propensity, LogNormalSoftPlus, η_prior
# decoding and post-processing
export decode_orders, order_posterior, spiral_posterior, stem_loglik, angle_loglikelihood, corrected_angles,
    displacement_probability, estimate_displacement_frequencies, estimate_permutation_frequencies
# LogDensityProblems
export TransformedLogDensityProblem
# simulation
export simulate_stems, sample_prior, divergence_angles, Golden_angle, Lucas_angle
# commitment timing
export DelayLaw, GaussianExpTails, delay_logpdf, free_parameters, with_free_parameters, reported_parameters,
    nshared_parameters, delay_grid, DELAY_GRID, delay_logdensity, tail_masses, overtaking, loglik_order_exact, order_diagnostics,
    order_samples, stem_terms, relative_loglik, fit_delay_law, delay_law_inference, shared_core_test,
    overtaking_profile, compare_profiles
# plastochron
export flowering_rate, flowering_cumulative, fit_flowering, plastochron_by_rank, plant_plastochron, tau_posterior,
    relative_plastochron, mean_plastochron, in_hours

# the documented internals: public, not exported (the `public` keyword needs Julia 1.11)
@static if VERSION >= v"1.11"
    eval(
        Meta.parse(
            "public Permanents, StemWorkspace, stem_loglik!, angle_vgh, twist_vgh, TorsionNodes, stem_nodes, " *
                "scan_rate, golden_section_max, newton_map, logbesseli, besseliratio, marginal_quantiles"
        )
    )
end

end # module PhyllotaxisFold
