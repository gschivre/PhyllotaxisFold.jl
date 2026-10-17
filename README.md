# PhyllotaxisFold.jl

Statistical models of the **order of organs along a plant stem**, read from the absolute azimuths of the organs
(flowers, fruits). In *Arabidopsis* the organs follow a spiral with a divergence close to the golden angle, but two
organs initiated close in time often end up in the inverted order. These permutations are a readout of the timing
of organ commitment. The package decodes the orders from the angles, then estimates the law of the commitment delay
of each site.

The manual is in [`docs/src`](docs/src): the method from the biology to the equations and the code
([home](docs/src/index.md), [angles](docs/src/angles.md), [sum over orders](docs/src/orders.md),
[propensity model](docs/src/propensity.md), [commitment timing](docs/src/timing.md)), the
[validation on simulated stems](docs/src/validation.md), and the API. Build it with Documenter (below).

## Two steps

| step | what | main functions |
|---|---|---|
| **1. Propensity model** | the angles given an order (spiral offset, orientation, golden or Lucas divergence, per-stem precision, stem twist, optional outlier angles), summed over every order within a band with a free log-propensity η_d for each displacement d | `PhyllotaxisModelAD`, `fit_map`, `fit_map_torsion!`, `decode_orders`, `order_posterior`, `spiral_posterior` |
| **2. Fold-commit timing** | site i is ready at time i (plastochrons) and commits after a random delay eᵢ, so σ = sortperm(i + eᵢ). The delay law (a Gaussian core plus early and late tails) is fitted by re-weighting the propensity model's posterior orders with the exact order likelihood | `DelayLaw`, `order_samples`, `relative_loglik`, `fit_delay_law`, `delay_law_inference`, `tail_masses` |

Step 1 uses an exact sum over the orders (a banded permanent computed by dynamic programming) and is fitted by maximum
a posteriori with Newton's method (analytic gradient, Hessian from one dual-number pass). The stem twist is a random
effect per stem, integrated by adaptive quadrature. Step 2 rests on an identity: both models share the angle
likelihood, so

    p(y | f) / p(y | A) = E over the propensity model's posterior orders of P_f(σ) / P_A(σ).

Any delay law f is thereby scored against the propensity model in seconds, without refitting the angles.

The fold-commitment model (costs derived from the passage law of a switch near its fold), its variants (B-mix,
B-tempo) and the Level-1 simulator were removed in version 0.2. They are in the tag
[`with-model-B`](../../tree/with-model-B).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/gschivre/PhyllotaxisFold.jl")
```

## Quick start

```julia
using PhyllotaxisFold, Random

# angles: one vector of absolute azimuths (radians, bottom to top) per stem; the twist coordinate in mm per organ
# here: 60 simulated stems of 45 organs, 200 mm long
x = [collect(range(0.0, 200.0; length = 45)) for _ in 1:60]
law = DelayLaw(0.42; πL = 0.05, mL = 1.5, πE = 0.05, mE = 1.0)  # core sd and tails, in plastochrons
angles = simulate_stems(law; plants = 60, organs = 45, x, logμκ = 1.8, torsion_mean = 0.2, torsion_sd = 0.15,
    rng = Xoshiro(1)).angles

# 1. the propensity model, twist integrated out (TorsionPrior in degrees per unit of x)
m = PhyllotaxisModelAD(angles; symmetric = true, torsion_x = x)          # outlier = true: uniform outlier angles
r = fit_map_torsion!(m; prior = TorsionPrior(m0 = 0.0, sm = 2.0, νs = 4.0, τs = 1.0))
θ = r.fit.θ                                                             # η (2k + 1 costs), logμκ, σκ, α
r.m_ω, r.s_ω                                                             # twist population (degrees per mm)
dec = decode_orders(m, r; full = true)       # joint MAP per stem: σ, twist rate, ϕ, o, ψ, κ
post = order_posterior(m, r; nsamples = 200) # exact posterior marginals ρ[p, d + k + 1] and samples of σ

# 2. the commitment delay law through the posterior orders
smp = order_samples(post, θ.η)
fit = fit_delay_law(smp, :both)              # :core, :late or :both
fit.law, fit.rel_ll                          # the law, and log p(y | f) − log p(y | A)
inf = delay_law_inference(smp, fit)          # s, P(e > 1), P(e < −1), P(|e| > 1) with robust (stem-clustered) se
inf.estimate.off, inf.se.off                 # the fraction of organs more than one plastochron off their slot
```

Mutant against wild type: fit each group separately, then compare the off-time fraction `inf.estimate.off` with the
standard errors (independent groups), or the summed `relative_loglik` of separate and shared laws (a likelihood-ratio
test). The off-time fraction is the best identified quantity. When the tails carry a large share of the organs, the
core sd and the late/early split are not identified (manual §5.5 and §6).

## Tests, manual and figures

```julia
] test PhyllotaxisFold
```

The tests cover the dynamic programming against enumeration, the analytic gradients and Hessians, the twist
quadrature, the decoding and posterior sampling on simulated stems, the exact order likelihood against Monte Carlo,
the re-weighting identity, and the recovery of a delay law from angles alone.

The manual (Documenter; the `docs` environment points to this package):

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl            # then open docs/build/index.html
```

Figures (simulated data only, GLMakie in the `docs` environment, written to `docs/src/figures`):

- `docs/make_figures.jl`: the illustrations of the method;
- `docs/make_validation_figures.jl`: fitting, decoding and timing recovered on simulated stems (manual §6; about 1.5 h
  on 18 threads, the replicates are cached in `docs/validation.jls`).

## Licence

MIT, see [`LICENSE`](LICENSE).
