"""
    PhyllotaxisFold

Statistical models of the order of organs along a plant stem, from the absolute azimuths of the organs: the propensity
model (free log-propensity of each displacement) decodes the orders, and a law of commitment delays (each site commits
after an independent random delay) is fitted through its posterior orders. Submodules, in dependency order:

| submodule          | content                                                                                     |
|--------------------|---------------------------------------------------------------------------------------------|
| `Common`           | canonical angles, log I₀ and Bessel ratio (dual-safe), half-t                                |
| `Torsion`          | stem torsion: coordinates, corrected angles, per-stem rate scan, quadrature, Newton MAP       |
| `Permanents`       | banded permanent: graph, forward / backward (marginals), exact sampling, Viterbi             |
| `AngleLikelihood`  | the angle likelihood of one stem and its analytic gradient; decoding and order posterior     |
| `PropensityPriors` | log-normal softplus prior on free log-propensities η and its bijectors                        |
| `PropensityModel`  | the propensity model: free symmetric log-propensities η, `PhyllotaxisModelAD`                 |
| `CommitmentTiming` | commitment delay laws, exact order likelihood, re-weighting of the posterior orders           |
| `Plastochron`      | plastochron by flower rank from daily open-flower counts (delays in hours)                    |

Every submodule's exported names are re-exported here. The method is described in the manual (`docs/src`).
"""
module PhyllotaxisFold

include("Common.jl")
include("Torsion.jl")
include("Permanents.jl")
include("AngleLikelihood.jl")
include("PropensityPriors.jl")
include("PropensityModel.jl")
include("CommitmentTiming.jl")
include("Plastochron.jl")

# re-export every submodule's exported names (and the submodule itself)
for mod in (Common, Torsion, Permanents, AngleLikelihood, PropensityPriors, PropensityModel, CommitmentTiming, Plastochron)
    @eval using .$(nameof(mod))
    for name in names(mod)
        @eval export $name
    end
end

end # module PhyllotaxisFold
