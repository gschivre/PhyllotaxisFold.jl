# PhyllotaxisFold.jl

*Reading the timing of organ initiation from the angles of the organs on a plant stem.*

In *Arabidopsis*, the flowers of a stem follow a spiral: each one is rotated by about 137.5° (the golden angle) from
the one below. Often two neighbouring flowers are swapped: the one initiated later sits below the other. These swaps
record *when* each organ was initiated at the tip of the plant. PhyllotaxisFold reads that record. From the angles
alone, it:

1. **reconstructs the order** in which the organs were initiated, stem by stem, with its uncertainty. It accounts for
   the noise of the measurement, the unknown spiral of each stem (orientation, golden or Lucas angle) and the twist of
   the stem;
2. **measures how often organs are overtaken**: for each distance d, the probability that the organ initiated d places
   later commits first (the *overtaking profile*), which is what the orders identify. A law of the initiation delay
   interprets it: how precise the timing of a typical organ is, and how often an organ is initiated out of its turn;
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
law = DelayLaw(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)
angles = simulate_stems(law; plants = 60, organs = 45, x, logμκ = 1.8, torsion_mean = 0.2, torsion_sd = 0.15,
    rng = Xoshiro(1)).angles

# 1. reconstruct the orders (the twist of each stem is integrated out)
m = PhyllotaxisModelAD(angles; torsion_x = x)
r = fit_map_torsion!(m; prior = TorsionPrior(mt = 0.0, st = 2.0, νt = 4.0, τt = 1.0))  # twist prior in °/mm
dec = decode_orders(m, r; full = true)       # the most probable order of each stem, with its twist and spiral
post = order_posterior(m, r; nsamples = 200) # the full posterior of the order: probabilities and samples

# 2. the timing of initiation
smp = order_samples(post, r.fit.θ.η)
fit = fit_delay_law(smp, :both)              # a precise core, plus late and early organs
inf = delay_law_inference(smp, fit)          # estimates with standard errors
prof = overtaking_profile(smp)               # how often an organ is overtaken by the one d places up, d = 1…4
prof.q, prof.se                              # the result to report; inf.profile is that of the fitted law
```

## Comparing groups

Fit each group on its own (for a mutant, against its own co-segregant wild type). Then compare their **overtaking
profiles**, `compare_profiles(prof_wt, prof_mutant)`: how often an organ is overtaken by the one 1, 2 or 3 places
up. This is what the angles identify, with no assumption on the shape of the delay law. The fitted law interprets
it: the precision of a typical organ (`inf.estimate.σc`) and the fraction of organs off their turn
(`inf.estimate.off`) are reliable only when off-time organs are rare. When one group has many, fitting the two groups
together with one shared precision (`fit_delay_law([smp_wt, smp_mutant])`, tested by `shared_core_test`) can recover
them. The manual explains why, and shows how well each quantity is recovered on simulated stems.

## Learn more

The manual explains the method step by step, from the biology to the equations, with one simulated stem followed
throughout. It is written in [`docs/src`](docs/src):

- [a worked analysis](docs/src/workflow.md): a simulated wild type and mutant, analysed step by step, with the
  checks to make and how to read the results;
- [the problem](docs/src/problem.md): swaps, what is measured, and the two-step plan;
- [angles given an order](docs/src/angles.md), [summing over all orders](docs/src/orders.md) and
  [the propensity model](docs/src/propensity.md): how the orders are reconstructed;
- [commitment timing](docs/src/timing.md): how the delay law is estimated from the reconstructed orders;
- [validation on simulated stems](docs/src/validation.md): what is recovered, and how well;
- [assumptions and limits](docs/src/limits.md): the assumptions, and what would break them.

Earlier versions included a "fold-commitment" model of the order (tag `with-model-B`).

## Authorship and AI assistance

PhyllotaxisFold.jl is developed by Geoffrey Schivre. The reconstruction of the orders from the angles (step 1: the
angle model, the propensity model and the exact sum over orders) is the author's work. It is the starting point of the
commitment-timing approach of step 2.

Claude (Opus 5.5, Anthropic), an AI model used as a coding and analysis agent through Claude Code, contributed under
the author's direction:

- the commitment-timing approach: scoring a delay law on the orders reconstructed in step 1, with the exact order
  likelihood and the re-weighting identity;
- its implementation (`CommitmentTiming`: the delay-law fit with exact derivatives, the shared-core option and the
  overtaking profile), its tests and its validation on simulated stems;
- the code review and cleanup of version 0.3, and the manual.

The author chose the questions, reviewed the results and the code, made the modelling decisions, and is responsible
for the package and for its use in publications.

## Licence

MIT, see [`LICENSE`](LICENSE).
