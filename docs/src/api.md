# API

Every exported name of the submodules is re-exported by `PhyllotaxisFold`. The submodules are listed in dependency
order.

```@docs
PhyllotaxisFold
```

## Propensity model

```@autodocs
Modules = [PhyllotaxisFold.PropensityModel, PhyllotaxisFold.PropensityPriors]
Private = false
```

## Angle likelihood, decoding and order posterior

```@autodocs
Modules = [PhyllotaxisFold.AngleLikelihood]
Private = false
```

## Stem twist

```@autodocs
Modules = [PhyllotaxisFold.Torsion]
Private = false
```

## Commitment timing

```@autodocs
Modules = [PhyllotaxisFold.CommitmentTiming]
Private = false
```

## Banded permanents

```@autodocs
Modules = [PhyllotaxisFold.Permanents]
Private = false
```

## Plastochron

```@autodocs
Modules = [PhyllotaxisFold.Plastochron]
Private = false
```

## Numerical tools

```@autodocs
Modules = [PhyllotaxisFold.Common]
Private = false
```
