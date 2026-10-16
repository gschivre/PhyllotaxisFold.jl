"""
    PhyllotaxisFold

Statistical models of the order of organs along a plant stem, from the absolute azimuths of the organs, with the order
explained by a commitment switch near its fold. Submodules, in dependency order:

| submodule          | content                                                                                     |
|--------------------|---------------------------------------------------------------------------------------------|
| `Common`           | canonical angles, log I₀ and Bessel ratio (dual-safe), half-t, Keys interpolation weights    |
| `Torsion`          | stem torsion: coordinates, corrected angles, per-stem rate scan, quadrature, Newton MAP       |
| `Permanents`       | banded permanent: graph, forward / backward (marginals), exact sampling, Viterbi             |
| `AngleLikelihood`  | the angle likelihood of one stem and its analytic gradient, shared by all models             |
| `PropensityPriors` | log-normal softplus prior on free log-propensities η and its bijectors                        |
| `PassageTime`      | fold passage law (plateau, ramp): Fokker–Planck tables, `FoldPassageTime`                     |
| `OrderStatistics`  | order law of a paced meristem, propensity map (shape, γ) ↦ η, exact order likelihood, tables |
| `PropensityModel`  | model A: free log-propensities η, `PhyllotaxisModelAD`                                        |
| `FoldAngleModel`   | model B: η from the passage law, optional tempo along the stem, `FoldModel`                   |
| `Plastochron`      | plastochron by flower rank from daily open-flower counts (input of the tempo option)          |

Every submodule's exported names are re-exported here. The method is described in `docs/method.md`.
"""
module PhyllotaxisFold

using Reexport

include("Common.jl")
include("Torsion.jl")
include("Permanents.jl")
include("AngleLikelihood.jl")
include("PropensityPriors.jl")
include("PassageTime.jl")
include("OrderStatistics.jl")
include("PropensityModel.jl")
include("FoldAngleModel.jl")
include("Plastochron.jl")

@reexport using .Common
@reexport using .Torsion
@reexport using .Permanents
@reexport using .AngleLikelihood
@reexport using .PropensityPriors
@reexport using .PassageTime
@reexport using .OrderStatistics
@reexport using .PropensityModel
@reexport using .FoldAngleModel
@reexport using .Plastochron

end # module PhyllotaxisFold
