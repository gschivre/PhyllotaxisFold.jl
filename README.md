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
   later commits first (the *overtaking profile*), which is what the orders identify;
3. **tests hypotheses about timing**: any law of the commitment delay that can be written as a density is fitted
   and compared on the same reconstructed orders, in a few lines of code and without refitting step 1. The default law
   (a precise Gaussian core, plus organs late or early) tells how precise the timing of a typical organ is, and how
   often an organ is initiated out of its turn;
4. **compares groups of plants**, for instance a mutant and its wild type, with standard errors that treat the stem as
   the unit of replication.

## Input data

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

# the data: one vector of azimuths per stem (radians, bottom to top), and the stem length (mm) below each organ.
# here, 60 simulated stems of 45 organs
x = [collect(range(0.0, 200.0; length = 45)) for _ in 1:60]
law = GaussianExpTails(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)
angles = simulate_stems(law; plants = 60, organs = 45, x, logμκ = 1.8, torsion_mean = 0.2, torsion_sd = 0.15,
    rng = Xoshiro(1)).angles

# 1. reconstruct the orders (the twist of each stem is integrated out)
prior = TorsionPrior(mt = 0.0, st = 2.0, νt = 4.0, τt = 1.0)  # twist prior, in °/mm
m = PhyllotaxisModelAD(angles; torsion = prior, torsion_x = x)
r = fit_map(m)                                     # prints its convergence checks and parameter table
dec = decode_orders(r; full = true)                # the most probable order of each stem, with its twist and spiral
post = order_posterior(r; nsamples = 200)          # the full posterior of the order: probabilities and samples

# 2. the timing of initiation
smp = order_samples(post, r.θ.η)
fit = fit_delay_law(smp, GaussianExpTails(:both))  # a precise core, plus late and early organs
inf = delay_law_inference(smp, fit)                # estimates with standard errors
prof = overtaking_profile(smp)                     # how often an organ is overtaken by the one d places up, d = 1…4
prof.q, prof.se                                    # the result to report; inf.profile is that of the fitted law
```

## Simulating stems

The model is also a simulator: stems with a known order, angular imprecision, twist and aberrant angles, e.g. to test
a measurement pipeline or a statistic of divergence angles. `simulate_stems(η; organs, x, logμκ, σκ, α, outlier,
torsion_mean, torsion_sd)`, `divergence_angles(y; o)`; `simulate_stems(fit)` draws stems like the data of a fit (posterior
predictive checks). See the manual page "Simulating stems".

## Custom delay laws

The default law is a starting point. Another hypothesis on how the meristem times its organs can be written as a
delay law: a type and its log density, plus its free parameters to fit it. For instance, a Student t, the law of
Gaussian delays whose precision varies from site to site:

```julia
import PhyllotaxisFold: delay_logpdf, free_parameters, with_free_parameters

struct StudentDelay{T <: Real} <: DelayLaw
    s::T
    ν::T
end
delay_logpdf(f::StudentDelay, z) = -(f.ν + 1) / 2 * log1p((z / f.s)^2 / f.ν)   # up to a constant
free_parameters(f::StudentDelay) = [log(f.s), log(f.ν)]
with_free_parameters(::StudentDelay, v) = StudentDelay(exp(v[1]), exp(v[2]))

st = fit_delay_law(smp, StudentDelay(0.5, 5.0))
st.rel_ll - fit.rel_ll          # how much better (> 0) or worse than the default law
```

The other tools of the package then apply to it: standard errors, overtaking profile, simulation of stems, and joint
fits of groups once the law declares which of its parameters the groups share. The manual page
[custom delay laws](docs/src/custom_laws.md) describes the interface, with two examples.

## Comparing groups

Fit each group on its own (for a mutant, against its own co-segregant wild type). Then compare their **overtaking
profiles**, `compare_profiles(prof_wt, prof_mutant)`: how often an organ is overtaken by the one 1, 2 or 3 places
up. This needs no assumption on the shape of the delay law. A fitted law interprets the difference: for the default
law, the precision of a typical organ (`inf.estimate.σc`) and the fraction of organs off their turn
(`inf.estimate.off`). These are reliable only when off-time organs are rare; when one group has many, fitting the two
groups together with one shared core sd (`fit_delay_law([smp_wt, smp_mutant], GaussianExpTails(:both))`, tested by
`shared_core_test`) can recover them. For a custom law, the shared parameters are those the law declares
(`nshared_parameters`), for instance a scale; a law that declares none is fitted group by group. The manual explains
why, and shows how well each quantity is recovered on simulated stems.

## Learn more

The manual explains the method step by step, from the biology to the equations, with one simulated stem followed
throughout. It is written in [`docs/src`](docs/src):

- [a worked analysis](docs/src/workflow.md): a simulated wild type and mutant, analysed step by step, with the
  checks to make and how to read the results;
- [custom delay laws](docs/src/custom_laws.md): how to define, fit and compare other laws of the commitment
  delays, and why the default is a Gaussian core with exponential tails;
- [the problem](docs/src/problem.md): swaps, what is measured, and the two-step plan;
- [angles given an order](docs/src/angles.md), [summing over all orders](docs/src/orders.md) and
  [the propensity model](docs/src/propensity.md): how the orders are reconstructed;
- [commitment timing](docs/src/timing.md): how the delay law is estimated from the reconstructed orders;
- [validation on simulated stems](docs/src/validation.md): what is recovered, and how well;
- [assumptions and limits](docs/src/limits.md): the assumptions, and what would break them;
- [notation and references](docs/src/notation.md), and the [API](docs/src/api.md).

To build the manual as a website, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Authorship and AI assistance

PhyllotaxisFold.jl is developed by Geoffrey Schivre. The reconstruction of the orders from the angles (step 1: the
angle model, the propensity model and the exact sum over orders) is the author's work. It is the starting point of the
commitment-timing approach of step 2.

Claude (Opus 5.5, Anthropic), an AI model used as a coding and analysis agent through Claude Code, contributed under
the author's direction:

- the commitment-timing approach: scoring a delay law on the orders reconstructed in step 1, with the exact order
  likelihood and the re-weighting identity;
- its implementation (`CommitmentTiming`: the delay-law fit with exact derivatives, the shared-core option, the
  overtaking profile, and the `DelayLaw` interface for user-defined laws, at the author's request), its tests and its
  validation on simulated stems;
- the code review and cleanup of version 0.3, and the manual.

The author chose the questions, reviewed the results and the code, made the modelling decisions, and is responsible
for the package and for its use in publications.

## Licence

MIT, see [`LICENSE`](LICENSE).
