# Defining and fitting a law

The default delay law is one hypothesis about how the meristem times its organs. Any other hypothesis that can be
written as a law of the commitment delay can be fitted to the same stems and compared with it, or with any other law,
on the same scale. This is the main use of the package beyond the default analysis, and it takes a few lines
([Deriving a law: the fold switch](fold_law.md) derives one from a model of the commitment switch):

- **The orders are reconstructed once.** Step 1 does not depend on the delay law, so every law is scored on the same
  posterior orders ([§5.5](timing.md#5.5-Re-weighting-the-reconstructed-orders)), with nothing to refit.
- **Only the density is needed.** The exact probability of an order, the fit, the standard errors and the simulation
  of stems all follow from it, with exact derivatives.
- **All laws share one scale.** Each fit returns its log-likelihood relative to the same reference, so any two laws
  can be compared directly, nested or not, including with the default law of the [worked analysis](workflow.md).

## Example: a Student t law

**Hypothesis.** Nothing requires every site to commit with the same precision. Suppose that each site commits after
a Gaussian delay, but with its own precision, and that the precision varies between sites as a Gamma law. The delay
of a random site then follows a **Student t** law, with a scale ``s`` and ``\nu`` degrees of freedom that measure how
much the precision varies (the smaller ``\nu``, the heavier the tails; as ``\nu \to \infty``, a Gaussian). Under this
law, off-time organs are not a separate class: they come from the sites with the least precise commitment.

!!! details "Derivation and caveats"
    Take a precision ``\lambda \sim \mathrm{Gamma}(\nu/2, \nu/2)`` (shape and rate) and a delay
    ``\mathcal N(0, s^2/\lambda)``. Integrating ``\lambda`` out gives the density
    ``\propto (1 + z^2/(\nu s^2))^{-(\nu + 1)/2}``: a Student t of scale ``s``. The law is symmetric: it assumes that a
    less precise site is not also later on average. Where imprecision comes with lateness, as in the fold switch
    ([Deriving a law: the fold switch](fold_law.md), where a smaller margin both widens and lengthens the delay), the
    law should instead be skewed towards late sites.

**Definition.**

```@example laws
using PhyllotaxisFold, Random
using Distributions: TDist
import PhyllotaxisFold: delay_logpdf, free_parameters, with_free_parameters, reported_parameters, nshared_parameters, delay_grid

struct StudentDelay{T <: Real} <: DelayLaw
    s::T   # scale (plastochrons)
    ν::T   # degrees of freedom
end

delay_logpdf(f::StudentDelay, z) = -(f.ν + 1) / 2 * log1p((z / f.s)^2 / f.ν)  # up to a constant

free_parameters(f::StudentDelay) = [log(f.s), log(f.ν)]                       # what to fit, on an unconstrained scale
with_free_parameters(::StudentDelay, v) = StudentDelay(exp(v[1]), exp(v[2]))

# optional
reported_parameters(f::StudentDelay) = (; s = f.s, ν = f.ν)                   # reported with standard errors
nshared_parameters(::StudentDelay) = 1                                        # groups fitted jointly share the scale
Base.rand(rng::AbstractRNG, f::StudentDelay) = f.s * rand(rng, TDist(f.ν))    # a fast sampler
nothing # hide
```

**Usage.** The law is used like the default one, with the samples of step 1 of the [worked analysis](workflow.md):

```julia
st = fit_delay_law(smp, StudentDelay(0.5, 5.0))    # fit, starting from s = 0.5, ν = 5
inf = delay_law_inference(smp, st)                 # s, ν, tail masses, profile, with robust standard errors
both = fit_delay_law(smp, GaussianExpTails(:both)) # the default law
st.rel_ll - both.rel_ll                            # > 0: the Student law explains the orders better

fit_delay_law([smp_wt, smp_mutant], StudentDelay(0.5, 5.0))       # one scale, ν per group
shared_core_test([smp_wt, smp_mutant], StudentDelay(0.5, 5.0))    # is one scale enough?
simulate_stems(StudentDelay(0.4, 3.0); plants = 50, organs = 40)  # stems from this law, to test the method on it
```

**Joint fits of groups.** A joint fit shares the parameters that the law declares with `nshared_parameters`: the
first free parameters, here the scale ``s``. They must mean the same thing in every group, which only the author of
the law can decide; for the default law, it is the core sd. A law that declares none cannot be fitted jointly
(`fit_delay_law` and `shared_core_test` throw an `ArgumentError`), but its groups can still be fitted separately and
compared through their overtaking profiles.

**Interpreting a comparison.** The stems of the worked analysis were simulated from the default law. The comparison
below therefore shows how a comparison behaves when the law that generated the data is known; it says nothing about
real plants (each fit takes a few seconds):

| law (free parameters) | wild type | mutant |
|---|---|---|
| default, Gaussian core only (1) | −49.0 | −70.3 |
| Student t (2) | +12.0 | +13.6 |
| default, core and both tails (5) | +15.6 | +56.3 |

- **The score ranks the laws.** The family that generated the data comes first in both groups. In the wild type,
  whose tails are light, the Student t comes close (3.6 units behind, with three fewer parameters): from these stems
  alone, the two would be hard to tell apart.
- **The profile shows where a law fails.** In the mutant, the Student t reproduces the neighbour swaps
  (``q_1 = 0.149`` against 0.150 read from the orders) but not the longer overtakings (``q_2 = 0.050`` against
  ``0.066 \pm 0.005``): a single shape parameter cannot produce both at the mutant's rates. Comparing
  `overtaking(fit.law)` with `overtaking_profile(smp)` shows what a law misses.

On real stems the generating law is unknown, and the same two criteria, the score and the profile, are the basis for
choosing between laws.

## Example: a Gumbel law (first cell to commit)

**Hypothesis.** A site is a group of many cells, and the organ is founded as soon as the *first* of them commits. The
earliest of many independent Gaussian times follows a **Gumbel** law for minima (extreme-value theory): a delay
``e = -\beta G``, with ``G`` a standard Gumbel variable. One parameter, and a long early tail.

```@example laws
struct GumbelDelay{T <: Real} <: DelayLaw
    β::T
end
delay_logpdf(f::GumbelDelay, z) = z / f.β - exp(z / f.β)
free_parameters(f::GumbelDelay) = [log(f.β)]
with_free_parameters(::GumbelDelay, v) = GumbelDelay(exp(v[1]))
reported_parameters(f::GumbelDelay) = (; β = f.β)
Base.rand(rng::AbstractRNG, f::GumbelDelay) = f.β * log(-log(rand(rng)))
nothing # hide
```

**Comparison.** On the same simulated stems it scores ``-44.1`` (wild type) and ``-132.9`` (mutant), far behind the
default law, as expected since the stems were not drawn from it. The profile again shows the cause: the tail of a
Gumbel law decays on the same scale ``\beta`` as its core, so a law broad enough for the long overtakings produces too
many neighbour swaps (``q_1 = 0.111`` against 0.094 in the wild type, ``q_3 = 0.002`` against 0.010). Whether real
stems behave like this is a question for the data.

**Closed form.** For this law, the probability of an order has a closed form (a Plackett–Luce law).
[§6.8](validation.md#6.8-An-exact-check-of-the-re-weighting) uses the same `GumbelDelay` to check the re-weighting against an exact answer.

## Example: the fold switch

The two laws above are short hypotheses. [Deriving a law: the fold switch](fold_law.md) goes the whole way, from a model of
the commitment switch (a bistable network driven through a fold) to a delay law: an inverse Gaussian whose skew is
set by how far above the fold the cell commits. It shows how to simplify a mechanism until its delay law can be
written down, fitted and tested on the orders.

## The interface

| method | required? | what it does |
|---|---|---|
| `delay_logpdf(f, z)` | yes | log density at delay ``z`` (plastochrons), up to a constant |
| `free_parameters(f)` | to fit | the parameters to fit, as a vector of unconstrained reals |
| `with_free_parameters(f, v)` | to fit | the law of the same family at those parameters (`f` is a template) |
| `reported_parameters(f)` | no | a named tuple that `delay_law_inference` reports with standard errors |
| `nshared_parameters(f)` | no | how many leading free parameters groups share in a joint fit (default 0: no joint fit) |
| `delay_grid(f)` | no | the grid of delays the law is evaluated on (default `DELAY_GRID`: −30 to 30, step 0.05) |
| `rand(rng, f)` | no | one delay; without it, `simulate_stems` draws from the density, more slowly |

Notes:

- **No normalising constant is needed.** The density is renormalised on a grid of delays, even when the constant
  depends on the parameters. Return `-Inf` outside the support, if the law has one.
- **No location parameter.** Only differences between delays change the order ([§5.2](timing.md#5.2-The-law-of-the-order-of-a-paced-meristem)), so a shift of the
  whole law is not identified and its fit would drift. The scale is identified, because sites are ready one
  plastochron apart.
- **Generic number types.** The fits pass dual numbers through these methods to get exact derivatives: use one type
  parameter `T <: Real` for all the fields, as above, and no `Float64` annotations.

## The delay grid

**What it is for.** A delay law is used only through its values on a grid of delays, where it is renormalised
(hence no normalising constant): the exact probability of an order is computed on that grid, and so are the overtaking
profile and the tail masses. Why, and what it requires of the law, is in
[§5.4](timing.md#5.4-The-exact-probability-of-one-order). The default, `DELAY_GRID` (−30 to 30 plastochrons, step
0.05), suits the default law.

**When to change it.** The grid must hold the whole law and resolve its finest detail:

- **Long or heavy tails: a wider grid.** A law cut off by the grid scores better than it should. The fits warn when
  the density has not decayed at the edge of the grid.
- **A narrow core: a finer step.** The error is set by the step relative to the narrowest feature of the law.
- **Always a step that divides one plastochron** (0.05, 0.1, 0.025, …), so that the shifts between sites fall on
  the grid.

A wider or finer grid costs proportionally more time.

**How to change it.** Declare the grid once, with the law:

```julia
delay_grid(::MyLaw) = range(-60.0, 60.0; step = 0.05)   # MyLaw: your DelayLaw type
```

Every function then uses it by default: the fits and their standard errors (the fit result records it as
`fit.grid`, which `delay_law_inference` reuses), the scores, the profile, the tail masses and the fallback sampler of
`simulate_stems`. A `grid` keyword still overrides it for a single call, e.g. to check that a result does not change
on a finer grid.

## Checks

The checks of the worked analysis apply to any law:

- **The profile.** The fitted law's profile (`overtaking(fit.law)`) should agree with the one read from the orders
  (`overtaking_profile(smp)`). If it does not, the law's shape is distorting the data, whatever its score.
- **The score.** Count about one unit of log-likelihood per extra parameter (Akaike), and do not interpret gains
  below about 5 ([§6.4](validation.md#6.4-The-delay-law)). To know how far a gain can be trusted, simulate stems from both laws and
  refit.
- **The grid.** If a fit warns that the density has not decayed at the edge of the grid, give the law a wider one
  ([The delay grid](custom_laws.md#The-delay-grid)).
- **The parameters.** The orders identify the profile, not the parameters ([§5.6](timing.md#5.6-The-delay-law)). A parameter that only
  shapes the far tails comes out with a large standard error; fewer parameters, or sharing one between groups, help.

## Choice of the default law

`GaussianExpTails(σc; πL, mL, πE, mE)`: a Gaussian core of sd ``\sigma_c``, a fraction ``\pi_L`` of the sites late by
an exponential delay of mean ``m_L``, and a fraction ``\pi_E`` early by ``m_E`` ([§5.6](timing.md#5.6-The-delay-law)).

- **A Gaussian core** for the typical site, whose delay sums several small independent timing errors (readiness and
  commitment, [§5.2](timing.md#5.2-The-law-of-the-order-of-a-paced-meristem)). Its sd measures the precision of commitment, and it produces the neighbour swaps.
- **Tails** for the long displacements, which a Gaussian cannot produce without far too many neighbour swaps. The
  exponential is the simplest extra delay of a given mean, with no preferred duration; late and early are separate,
  since nothing requires the two sides to be alike.
- **Nested questions.** Removing a tail gives the simpler law, so the variants `GaussianExpTails(:core)`, `:late`,
  `:early` and `:both` answer, by likelihood ratios: are some organs off time, and on which side?
- **Exact and fast.** Its density has a closed form.

It is a starting point, not a claim about the plant. Its core sd and off-time fraction are readings of the overtaking
profile through its shape. If another law explains the orders as well with another shape, those readings change while
the profile does not; fitting that law is how to find out.

**Code.** `DelayLaw`, `delay_logpdf`, `free_parameters`, `with_free_parameters`, `reported_parameters`,
`nshared_parameters`; `GaussianExpTails`; `fit_delay_law(samples, f)`, `shared_core_test(groups, f)`.
