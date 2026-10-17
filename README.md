# PhyllotaxisFold.jl

*Reading the timing of organ initiation from the angles of the organs on a plant stem.*

In *Arabidopsis*, the flowers of a stem follow a spiral: each one is rotated by about 137.5° (the golden angle) from
the one below. Often two neighbouring flowers are swapped: the one initiated later sits below the other. These swaps
record *when* each organ was initiated at the tip of the plant. PhyllotaxisFold reads that record. From the angles
alone, it:

1. **reconstructs the order** in which the organs were initiated, stem by stem, with its uncertainty. It accounts for
   the noise of the measurement, the unknown spiral of each stem (orientation, golden or Lucas angle) and the twist of
   the stem;
2. **estimates the law of the initiation delay**: how precise the timing of a typical organ is, and how often an organ
   is initiated out of its turn, early or late;
3. **compares groups of plants**, for instance a mutant and its wild type, with standard errors that treat the stem as
   the unit of replication.

## What you need

For each stem, from the bottom to the top:

- the **absolute azimuth** of each organ (radians);
- optionally, the **internode lengths** (mm), so that the twist of the stem is measured per mm of stem;
- optionally, **daily counts of open flowers**, to convert delays from plastochrons into hours.

## Installation

The package is not registered yet. Install it from its repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/gschivre/PhyllotaxisFold.jl")
```

## Quick start

```julia
using PhyllotaxisFold, Random

# your data: one vector of azimuths per stem (radians, bottom to top), and the stem length (mm) below each organ.
# here, 60 simulated stems of 45 organs
x = [collect(range(0.0, 200.0; length = 45)) for _ in 1:60]
law = DelayLaw(0.42; πL = 0.05, mL = 1.5, πE = 0.05, mE = 1.0)
angles = simulate_stems(law; plants = 60, organs = 45, x, logμκ = 1.8, torsion_mean = 0.2, torsion_sd = 0.15,
    rng = Xoshiro(1)).angles

# 1. reconstruct the orders (the twist of each stem is integrated out)
m = PhyllotaxisModelAD(angles; torsion_x = x)
r = fit_map_torsion!(m; prior = TorsionPrior(m0 = 0.0, sm = 2.0, νs = 4.0, τs = 1.0))  # twist prior in °/mm
dec = decode_orders(m, r; full = true)       # the most probable order of each stem, with its twist and spiral
post = order_posterior(m, r; nsamples = 200) # the full posterior of the order: probabilities and samples

# 2. estimate the delay law
smp = order_samples(post, r.fit.θ.η)
fit = fit_delay_law(smp, :both)              # a precise core, plus late and early organs
inf = delay_law_inference(smp, fit)          # estimates with standard errors
inf.estimate.off, inf.se.off                 # the fraction of organs more than one plastochron off their turn
```

## Comparing groups

Fit each group on its own (for a mutant, against its own co-segregant wild type). Then compare the **off-time
fraction** `inf.estimate.off` of the two groups with their standard errors. This is the quantity the angles identify
best. The precision of a typical organ (`inf.estimate.s`) and the split between late and early organs are reliable
only when off-time organs are rare. The manual explains why, and shows how well each quantity is recovered on
simulated stems.

## Learn more

The manual explains the method step by step, from the biology to the equations, with one simulated stem followed
throughout. It is written in [`docs/src`](docs/src):

- [the problem](docs/src/problem.md): swaps, what is measured, and the two-step plan;
- [angles given an order](docs/src/angles.md), [summing over all orders](docs/src/orders.md) and
  [the propensity model](docs/src/propensity.md): how the orders are reconstructed;
- [commitment timing](docs/src/timing.md): how the delay law is estimated from the reconstructed orders;
- [validation on simulated stems](docs/src/validation.md): what is recovered, and how well;
- [inference and limits](docs/src/inference.md): the assumptions, and what would break them.

Earlier versions included a "fold-commitment" model of the order (tag `with-model-B`).

## Licence

MIT, see [`LICENSE`](LICENSE).
