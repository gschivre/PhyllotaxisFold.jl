"""
    PhyllotaxisFold

Statistical models of the order of organs along a plant stem, from the absolute azimuths of the organs: the propensity
model (free log-propensity of each displacement) decodes the orders, and the fold-commit timing law (each site commits
after a random delay) is fitted through its posterior orders. Submodules, in dependency order:

| submodule          | content                                                                                     |
|--------------------|---------------------------------------------------------------------------------------------|
| `Common`           | canonical angles, log I₀ and Bessel ratio (dual-safe), half-t                                |
| `Torsion`          | stem torsion: coordinates, corrected angles, per-stem rate scan, quadrature, Newton MAP       |
| `Permanents`       | banded permanent: graph, forward / backward (marginals), exact sampling, Viterbi             |
| `AngleLikelihood`  | the angle likelihood of one stem and its analytic gradient; decoding and order posterior     |
| `PropensityPriors` | log-normal softplus prior on free log-propensities η and its bijectors                        |
| `PropensityModel`  | the propensity model: free log-propensities η, `PhyllotaxisModelAD`                           |
| `CommitmentTiming` | fold-commit delay laws, exact Level-0 order likelihood, re-weighting of the posterior orders  |
| `Plastochron`      | plastochron by flower rank from daily open-flower counts (delays in hours)                    |

Every submodule's exported names are re-exported here. The method is described in `docs/method.md`. Model B (the
fold-commitment model: costs from the passage law of a switch near its fold), B-mix, B-tempo and the Level-1 simulator
are in the tagged version `with-model-B`.
"""
module PhyllotaxisFold

using Reexport

include("Common.jl")
include("Torsion.jl")
include("Permanents.jl")
include("AngleLikelihood.jl")
include("PropensityPriors.jl")
include("PropensityModel.jl")
include("CommitmentTiming.jl")
include("Plastochron.jl")

@reexport using .Common
@reexport using .Torsion
@reexport using .Permanents
@reexport using .AngleLikelihood
@reexport using .PropensityPriors
@reexport using .PropensityModel
@reexport using .CommitmentTiming
@reexport using .Plastochron

end # module PhyllotaxisFold
