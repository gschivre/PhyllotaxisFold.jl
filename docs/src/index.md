# PhyllotaxisFold.jl

*Reading the timing of organ initiation from the angles of the organs on a plant stem.*

The flowers of an *Arabidopsis* stem sit on a spiral: going up the stem, each flower is rotated by about 137.5°, the
golden angle, from the one below. Every so often two neighbours are swapped: the flower that should come first sits
above the other one. These swaps are a record of *when* each organ was initiated at the tip of the plant, weeks
before we look at the stem. This package reads that record. From the angles alone it reconstructs the order in which
the organs were initiated. It then estimates how precise the timing of initiation is, and how often an organ is
initiated out of its turn.

![A simulated stem: its angles, the order hidden behind them, and the commitment delays that produced that order](figures/e1_example_stem.png)

*The example stem used throughout this manual (simulated). (a) Each organ is a dot at its height and azimuth. The grey
line is the ideal spiral. (b) The angles between consecutive organs: mostly the golden angle, with a few
characteristic jumps. (c) What produced them, hidden from the observer: each site became ready at a regular pace
(circles) and committed to become an organ after a small random delay (stars). Where two stars cross, the organs are
swapped on the stem.*

The example stem is one of 40 simulated stems of 35 organs. Their delays come from one law, used throughout the
manual: a Gaussian core of sd 0.42 plastochron, plus 6 % of the sites late by an extra delay of mean 1.5 plastochron
and 6 % early by an advance of mean 1.0 plastochron ([§5.6](timing.md)).

## The story in five steps

1. [**The problem**](problem.md). What a swap is, what we measure, and why the order cannot simply be read off the
   angles. The question we really want to answer is about timing, and this leads to a two-step plan.
2. [**Angles given an order**](angles.md). If we knew the order, the angles would be easy to describe: a regular
   spiral, a little noise, a slow twist of the stem.
3. [**Summing over all orders**](orders.md). We do not know the order, and there are more possible orders than atoms
   in the Earth. One observation reduces the sum over all of them to a few thousand operations.
4. [**The propensity model**](propensity.md). The most general law of the order that keeps that sum exact. It
   reconstructs the order of every stem, with its uncertainty.
5. [**Commitment timing**](timing.md). The law of the order that a timing mechanism predicts is not of that form.
   But it can be scored exactly on the orders reconstructed in step 4, and this gives its likelihood from the angles.

Then: [**validation on simulated stems**](validation.md) (does each step recover a known truth?),
[**assumptions and limits**](limits.md), [notation and references](notation.md), and the [API](api.md).

## How to read this manual

The main text is written for a biologist who is comfortable with probabilities. It explains every idea in words,
then states it as an equation. Each modelling choice is labelled:

- **assumption**: a choice about the plant or the measurement, which could be wrong;
- **result**: a mathematical consequence of the assumptions, proved in the text.

The proofs and the numerical details are in folded boxes like this one:

!!! details "Derivation"
    The step-by-step argument behind a result, for readers who want to check it. The main text never depends on it.

Each page ends with the functions that implement it.

## Quick start

```julia
using PhyllotaxisFold, Random

# 60 simulated stems of 45 organs: the stem length (mm) below each organ, and a known delay law (plastochrons)
x = [collect(range(0.0, 200.0; length = 45)) for _ in 1:60]
law = DelayLaw(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)
sim = simulate_stems(law; plants = 60, organs = 45, x, logμκ = 1.8, torsion_mean = 0.2, torsion_sd = 0.15,
    rng = Xoshiro(1))

# step 1: the propensity model, the twist of each stem integrated out (TorsionPrior in degrees per mm)
m = PhyllotaxisModelAD(sim.angles; torsion_x = x)
r = fit_map_torsion!(m; prior = TorsionPrior(m0 = 0.0, sm = 2.0, νs = 4.0, τs = 1.0))
dec = decode_orders(m, r; full = true)        # the most probable order of each stem, with its twist and spiral
post = order_posterior(m, r; nsamples = 200)  # exact posterior of the order: marginals and samples

# step 2: the commitment delay law, through the orders of step 1
smp = order_samples(post, r.fit.θ.η)
fit = fit_delay_law(smp, :both)               # a precise core and two tails (late, early)
inf = delay_law_inference(smp, fit)           # estimates with standard errors clustered by stem
prof = overtaking_profile(smp)                # q_d = P(site i + d commits before site i), read from the orders
prof.q, prof.se                               # the result to report; inf.profile is that of the fitted law
```

To compare two groups of plants, for instance a mutant and its wild type, compare their overtaking profiles:
`compare_profiles(prof_wt, prof_mutant)`. The profile is what the orders identify. The parameters of the delay law
(the precision of a typical commitment, the fraction of off-time organs) interpret it within the assumed shape of the
law. [§5.6](timing.md) and [§6](validation.md) explain which quantities the orders identify well.
