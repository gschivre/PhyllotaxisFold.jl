# API

The exported names, by source file. The building blocks that are needed for an analysis (the banded
permanents, the numerical tools, the internals of the likelihood) are not exported; they are listed at the end, by
their qualified names (declared `public` on Julia ≥ 1.11).

```@docs
PhyllotaxisFold
```

## Propensity model

```@autodocs
Modules = [PhyllotaxisFold]
Pages = ["PropensityModel.jl", "PropensityPriors.jl", "Fitting.jl"]
Private = false
```

## Simulation

```@autodocs
Modules = [PhyllotaxisFold]
Pages = ["Simulation.jl"]
Private = false
```

## Angle likelihood, decoding and order posterior

```@autodocs
Modules = [PhyllotaxisFold]
Pages = ["AngleLikelihood.jl", "Decoding.jl"]
Private = false
```

## Stem twist

```@autodocs
Modules = [PhyllotaxisFold]
Pages = ["Torsion.jl"]
Private = false
```

## Commitment timing

```@autodocs
Modules = [PhyllotaxisFold]
Pages = ["CommitmentTiming.jl"]
Private = false
```

## Plastochron

```@autodocs
Modules = [PhyllotaxisFold]
Pages = ["Plastochron.jl"]
Private = false
```

## Numerical tools

```@autodocs
Modules = [PhyllotaxisFold]
Pages = ["Common.jl"]
Private = false
```

## Internals (not exported)

The banded permanents:

```@docs
PhyllotaxisFold.Permanents
PhyllotaxisFold.Permanents.PermutationGraph
PhyllotaxisFold.Permanents.PermanentWorkspace
PhyllotaxisFold.Permanents.toeplitz_weights!
PhyllotaxisFold.Permanents.forward!
PhyllotaxisFold.Permanents.forward_linear!
PhyllotaxisFold.Permanents.backward!
PhyllotaxisFold.Permanents.sample_permutation!
PhyllotaxisFold.Permanents.map_permutation
PhyllotaxisFold.Permanents.logprob_permutation
PhyllotaxisFold.Permanents.cycle_lengths
```

The likelihood of one stem, the derivatives behind `fit_map`, and the twist quadrature:

```@docs
PhyllotaxisFold.stem_loglik!
PhyllotaxisFold.angle_vgh
PhyllotaxisFold.twist_vgh
PhyllotaxisFold.stem_nodes
PhyllotaxisFold.scan_rate
PhyllotaxisFold.golden_section_max
PhyllotaxisFold.newton_map
PhyllotaxisFold.logbesseli
PhyllotaxisFold.besseliratio
PhyllotaxisFold.marginal_quantiles
```
