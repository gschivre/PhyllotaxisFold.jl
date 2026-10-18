# 5. Commitment timing

Step 1 is done. For every stem, the propensity model gives the exact posterior of its order, and we can draw as many
orders from it as we like ([§3.7](orders.md)). Step 2 asks the question of [§1.4](problem.md): *which delay law ``f``
explains these orders?* The page answers it in four moves.

1. Write down the law of the order that a delay law predicts, ``P_f(\sigma)`` ([§5.2](timing.md)).
2. See why the exact sum over orders of [§3](orders.md) cannot use it, and why this is only a small obstacle
   ([§5.3](timing.md)).
3. Score a single order exactly, by a recursion over the heights ([§5.4](timing.md)).
4. Turn these scores into the likelihood of ``f``, by re-weighting the orders of step 1 ([§5.5](timing.md)).

The delay law itself ([§5.6](timing.md)) and its fit ([§5.7](timing.md)) come last. But first, why should a delay be
random at all?

## 5.1 Why a random delay

In each cell of the peripheral zone of the meristem, a regulatory network downstream of auxin signalling reads an
input that rises as auxin accumulates at the site. *Commitment* is the switch of this network to the organ-founder
state. A switch of this kind often happens at a fold: the uncommitted state merges with a threshold and disappears as
the input rises. Near a fold, the whole network slows down onto one slow variable, ``dX = (\mu + X^2)\,ds + dW_s``
(figure 1). Even after the uncommitted state has disappeared, the dynamics stay slow near ``X = 0``. Fast molecular
noise then turns into a slow and *variable* commitment delay.

![The fold](figures/m0_fold.png)

*Figure 1. The speed of the switch variable, ``dX/ds = \mu + X^2``, for three margins. Left, before the fold: a stable
uncommitted state (blue) and a threshold (red circle). Middle: the two merge, at the fold. Right, after the fold: no
state is left, but the speed stays small near ``X = 0``. The time spent there sets the commitment delay.*

This picture motivates a random delay, but the package does not impose the delay law that a particular switch would
produce. It fits a flexible law ([§5.6](timing.md)) and lets the orders decide its shape.

## 5.2 The law of the order of a paced meristem

Recall the mechanism of [§1.4](problem.md): site ``i`` is ready at time ``i`` (in plastochrons), it commits at
``\tau_i = i + e_i`` with independent delays ``e_i \sim f``, and the order on the stem is the ranking of the
``\tau_i``. Only differences of delays matter, so ``f`` is defined up to a shift, and its core is centred at 0.

**Two sites first.** Sites 1 and 2 are swapped when site 2 commits first: ``2 + e_2 < 1 + e_1``, that is
``e_1 - e_2 > 1``. If the delays are Gaussian with sd ``s``, their difference is Gaussian with sd ``s\sqrt 2``, and

```math
P(2\,1) = \Phi\!\left(-\frac{1}{s\sqrt 2}\right).
```

With ``s = 0.42`` plastochron, this is 4.6 %: about one pair of neighbours in twenty is swapped. The same reasoning
shows why the Gaussian alone cannot be the whole story. An organ displaced by two positions needs
``e_i - e_{i+2} > 2``, which has probability ``\Phi(-2/(s\sqrt 2)) \approx 4 \times 10^{-4}``: one pair in 2600.
Real stems have many more ([§5.6](timing.md)).

**Any number of sites.** The organ at height ``p`` comes from site ``\sigma(p)``. So ``\sigma`` is the order on the
stem exactly when ``\tau_{\sigma(1)} < \tau_{\sigma(2)} < \dots < \tau_{\sigma(n)}``. Call
``t_p = \tau_{\sigma(p)}`` the commitment time of the organ at height ``p``. Its delay is ``t_p - \sigma(p)``, and the
probability of the order is the probability of this chain of inequalities:

!!! note "result — the order law of independent delays"
    ```math
    P_f(\sigma) = \int_{t_1 < t_2 < \dots < t_n} \prod_{p=1}^{n} f\big(t_p - \sigma(p)\big)\, dt_1 \cdots dt_n .
    ```

For two sites, this integral is the ``\Phi`` above.

**What the delay covers.** The delays are independent from site to site. This is less restrictive than it looks.
Suppose that the readiness of a site itself wanders, so that site ``i`` is ready at ``i + u_i`` and commits a delay
``c_i`` later. Then ``\tau_i = i + (u_i + c_i)``: the same model, with ``e_i = u_i + c_i``, as long as the ``u_i`` are
independent. So ``e_i`` is the deviation of site ``i`` from a regular clock, and ``f`` measures the total timing noise
around that clock. The orders cannot tell how much of it comes from readiness and how much from commitment
([§7](limits.md)).

What the model does not cover are timing errors that build on each other: a clock whose intervals vary, so that its
errors add up like a random walk, or a late site that also delays the next ones. Both make the delays of neighbouring
sites dependent ([§7](limits.md) lists the signs). [§6.7](validation.md) shows when a drifting clock starts to be read
as tails.

## 5.3 Not a product, but close to one

The integral has one factor per organ, as in the product form of [§3.1](orders.md). But the factors are tied together
by the chain ``t_1 < \dots < t_n``: where the organ of site ``j`` ends up depends on how many other sites committed
before it, so on all the other delays at once. Can a product ``\prod_p h_{p,\sigma(p)}`` still reproduce
``P_f``? There is a simple test.

**The test.** In a product form, a swap has a fixed price. Swapping the organs at heights 1 and 2 multiplies the
probability by the same factor, ``e^{\eta_{+1} + \eta_{-1}}``, whatever happens at heights 3 and 4 (their factors
cancel in the ratio):

```math
\frac{P(2\,1\,3\,4)}{P(1\,2\,3\,4)} = \frac{P(2\,1\,4\,3)}{P(1\,2\,4\,3)} .
```

So compute both ratios under ``P_f``. Here are the exact probabilities of the four orders, for Gaussian delays of sd
0.42 (they come from the recursion of [§5.4](timing.md); the box below has the code):

| order | 1 2 3 4 | 2 1 3 4 | 1 2 4 3 | 2 1 4 3 |
|---|---|---|---|---|
| ``P_f(\sigma)`` | 0.8635 | 0.0439 | 0.0439 | 0.00214 |

The odds of the first swap are ``0.0439 / 0.8635 = 0.0508`` when heights 3–4 are in order, but
``0.00214 / 0.0439 = 0.0487`` when they are swapped: **4 % lower**. No product form can do this.

Where does the coupling come from? From the boundary between the two pairs. If the organs at heights 2 and 3 did not
have to commit in order, the two pairs would be independent and the two odds would be equal. They must, though, and
*which* sites sit at heights 2 and 3 depends on both swaps: sites 2 and 3 in 1 2 3 4, sites 1 and 4 in 2 1 4 3. So the
probability of one swap depends a little on whether the other has happened.

The sign of this dependence is not fixed. With the delay law of the running example ([home page](index.md)), which
has late and early tails, the two odds are 0.0913 and 0.0941: **3.1 % higher** when heights 3–4 are swapped. A product form, with its one price
per swap, can follow neither effect.

!!! details "Reproducing the numbers"
    ```julia
    using PhyllotaxisFold
    P(σ, f) = exp(loglik_order_exact(σ, f))      # exact probability of an order under f (§5.4)
    odds(f) = (P([2, 1, 3, 4], f) / P([1, 2, 3, 4], f), P([2, 1, 4, 3], f) / P([1, 2, 4, 3], f))
    gauss = DelayLaw(0.42)                       # Gaussian delays, sd 0.42 plastochron
    ex = DelayLaw(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0) # the law of the running example
    P([2, 1], gauss)                             # 0.046 (§5.2)
    odds(gauss)                                  # (0.0508, 0.0487)
    odds(ex)                                     # (0.0913, 0.0941)
    P([3, 1, 2], gauss) + P([2, 3, 1], gauss), P([3, 1, 2], ex) + P([2, 3, 1], ex) # 0.0003, 0.024 (§5.6)
    ```

Two lessons, one bad and one good.

- **The bad one.** ``P_f`` is outside the family of laws for which the sum over orders of [§3](orders.md) is exact.
  So ``p(y \mid f) = \sum_\sigma P_f(\sigma)\, p(y \mid \sigma)`` cannot be computed that way.
- **The good one.** The deviation is a few percent. A product-form law cannot *be* ``P_f``, but it can come close.
  The propensity model is such a law, fitted to the same stems. The next two sections exploit this.

## 5.4 One order at a time is easy

We cannot sum ``P_f`` over all orders, but we can compute it exactly for *one* given order, fast. The idea is the same
as in the forward pass of [§3](orders.md): go up the stem one height at a time, and carry only what the next height
needs. Here, that is the distribution of the commitment time of the organ just placed.

!!! note "result — forward recursion over the heights"
    Let ``a_p(t)`` be the density of ``t_p`` jointly with the event ``t_1 < \dots < t_p``. Then
    ```math
    a_1(t) = f\big(t - \sigma(1)\big),\qquad
    a_p(t) = f\big(t - \sigma(p)\big) \int_{-\infty}^{t} a_{p-1}(u)\, du,\qquad
    P_f(\sigma) = \int a_n(t)\, dt .
    ```

In words: the organ at height ``p`` commits at ``t`` with density ``f(t - \sigma(p))``, and the organs below it must
all have committed before ``t``; the integral of ``a_{p-1}`` up to ``t`` is the probability of that. For two sites in
order, ``P(1\,2) = \int f(t - 2)\, F(t - 1)\, dt``, with ``F`` the cumulative of ``f``: the probability that site 1
commits before site 2, as it should.

!!! details "Derivation and numerics"
    Integrate the variables of the ``n``-fold integral in the order ``t_1, t_2, \dots``. The variable ``t_{p-1}``
    appears in the factor ``f(t_{p-1} - \sigma(p-1))``, in the constraints ``t_{p-2} < t_{p-1}`` (already absorbed into
    ``a_{p-1}``) and ``t_{p-1} < t_p``. Integrating it over ``(-\infty, t_p)`` gives the cumulative of ``a_{p-1}`` at
    ``t_p``. ∎

    The functions are represented on a grid of step 0.05 plastochron (`DELAY_GRID`, from −30 to 30), and each
    ``a_p`` is renormalised as it goes (the logs of the normalisers add up to ``\log P_f(\sigma)``). The delay
    density itself is renormalised on the grid, so the grid must hold all of it. A tail of mass ``\pi`` and mean
    ``m`` loses ``\pi e^{-30/m}`` beyond the edge: below ``10^{-5}`` for ``m \le 3`` plastochrons. A cut-off tail
    would not be harmless: renormalising gives the mass it lost back to every site, a bonus of about ``n`` times the
    lost mass per stem. `fit_delay_law` warns if a fitted tail is too long for the grid. The cumulative
    counts the cell that contains ``t`` for one half (midpoint rule). Counting it in full would treat two times in the
    same cell as ordered, which biases ``\log P_f(\sigma)`` upwards at first order in the step: by +0.28 for an order
    of 6 sites with three displacements. The midpoint rule leaves an error below 0.01. The tests check the two-site
    closed form ``\Phi(1/(s\sqrt2))``, that the probabilities of all 24 orders of 4 sites sum to 1, and the frequencies
    of the orders drawn from the delay law.

A stem of 45 organs is scored in a fraction of a millisecond. And step 1 gives us exactly the orders worth scoring:
the posterior samples of each stem. How do their scores become the likelihood of ``f``?

## 5.5 Re-weighting the reconstructed orders

The propensity model ``A`` and the delay law ``f`` describe the angles in the same way ([§2.5](angles.md)). They differ
only in the law of the order, ``P_A(\sigma)`` or ``P_f(\sigma)``. So write the likelihood of ``f`` as the sum over
orders, and use what step 1 already knows: under ``A``, Bayes' rule turns the angle term into the posterior of the
order.

!!! note "result — the re-weighting identity"
    For one stem,
    ```math
    \frac{p(y \mid f)}{p(y \mid A)} = \sum_\sigma p(\sigma \mid y, A)\, \frac{P_f(\sigma)}{P_A(\sigma)}
    = \mathbb E_{\sigma \sim p(\sigma \mid y, A)}\Big[ \frac{P_f(\sigma)}{P_A(\sigma)} \Big] .
    ```

!!! details "Derivation"
    ``p(y \mid f) = \sum_\sigma p(y \mid \sigma)\, P_f(\sigma)``, and under ``A``, Bayes' rule gives
    ``p(y \mid \sigma) = p(\sigma \mid y, A)\, p(y \mid A) / P_A(\sigma)``. Substitute and divide by ``p(y \mid A)``. ∎
    The identity requires ``P_A(\sigma) > 0`` wherever ``P_f(\sigma)\,p(y \mid \sigma) > 0``. Orders outside the
    band of ``A`` violate this; they are discussed under the limits below.

This is the intuition of [§1.5](problem.md), made exact. A delay law is good if it gives a high probability to the
orders that the angles support. The division by ``P_A(\sigma)`` removes the part of that support that came from
``A``'s own law of the order, so that ``A`` is not counted twice. ``P_A(\sigma)`` is computable: it is the product
form of [§4](propensity.md) divided by its permanent.

**In practice.** The expectation is an average over the posterior samples of the stem. With
``\sigma_1, \dots, \sigma_M`` the distinct sampled orders and ``w_j`` their frequencies,

```math
\Delta\ell_s(f) = \log \sum_{j} w_j\, \frac{P_f(\sigma_j)}{P_A(\sigma_j)},\qquad
\Delta\ell(f) = \sum_{\text{stems}} \Delta\ell_s(f) = \log p(y \mid f) - \log p(y \mid A) .
```

``\Delta\ell > 0`` means that the timing law explains the angles better than the free costs of the propensity model.
Two delay laws are compared by the difference of their ``\Delta\ell``, in which ``A`` cancels. Scoring a new law takes
seconds, because neither the angles nor the permanents are recomputed.

![Re-weighting the orders of the example stem](figures/e5_example_reweighting.png)

*Figure 2. The example stem. (a) Its distinct posterior orders, with their frequencies among 200 samples. (b) The
log-weight ``\log P_f(\sigma) - \log P_A(\sigma)`` of each, under the true delay law: a ratio of two probabilities,
not the probability of the order under ``f``. The true order is in orange.
(c) ``\Delta\ell`` summed over the 40 stems (black) and the term of the example stem (orange, right axis), as a
function of the core sd ``s``, the tails kept at their true values. The dashed line is the true ``s``.*

Figure 2 shows the identity at work on the example stem. One order, the true one, carries 90 % of its posterior
samples, and seven others share the rest (a). The true delay law finds the true order about six times more probable
than the propensity model does (log-weight +1.8), and most of the alternatives less probable (b). So
``\Delta\ell_s > 0``: on this stem, the timing law explains the angles better than ``A``.

Order 4 has a log-weight as high as the true order's. This does not mean that ``f`` prefers it: ``f`` finds it 16
times *less* probable than the true order (``\log P_f`` = −18.0 against −15.2). Its weight is high because ``A``
finds it less probable still. Where the true order has two neighbours each displaced by one (−1, +1), order 4 has them
displaced by two (+2, −2): a pattern that the product form penalises a little more than a law with tails does
([§5.3](timing.md)). The weight corrects for this. Order 4 holds 0.5 % of the samples, so it adds 0.03 to the sum,
against 5.3 for the true order. Repeated for each value
of the core sd and summed over the stems, ``\Delta\ell`` peaks close to the true value (c).

**When can the average be trusted?** An average over samples is precise when the weights
``r_j = P_f(\sigma_j)/P_A(\sigma_j)`` are of similar size. Their spread is measured by the *effective sample size*
over the ``N`` draws, ``(\sum r)^2 / \sum r^2``: it equals ``N`` when all the weights are equal, and 1 when a single
draw carries everything. The weights are even when ``P_A`` is close to ``P_f``, and [§5.3](timing.md) showed that a
product-form law can come close to ``P_f``. In the validation, the effective sample size is typically 175–200
out of 200 draws ([§6.6](validation.md)).

**Limits.** The angle nuisances (``\kappa``, spiral, twist) are those of the propensity model's fit. Orders outside its
band (``|d| > k``) are never sampled, so a law that puts mass on them is slightly penalised. A stem whose effective
sample size is small is scored imprecisely; more posterior samples help.

## 5.6 The delay law

**assumption** — the delay is a precise Gaussian core, plus two exponential tails for sites committed late or early:

```math
f = (1 - \pi_L - \pi_E)\, \mathcal N(0, s^2)
+ \pi_L\, \big(\mathcal N(0, s^2) \ast \mathrm{Exp}(m_L)\big)
+ \pi_E\, \big(\mathcal N(0, s^2) \ast (-\mathrm{Exp}(m_E))\big).
```

The core sd ``s`` is the precision of a typical commitment. A fraction ``\pi_L`` of the sites is late by an extra
exponential delay of mean ``m_L``, and a fraction ``\pi_E`` is early by ``m_E``. Why tails? A Gaussian core of sd 0.42
makes neighbour swaps common but displacements by two positions very rare ([§5.2](timing.md)). A tail produces exactly
the pattern a Gaussian cannot: one organ displaced by two or three positions while its neighbours stay in order. For
three sites, the two orders with a displacement by two (3 1 2 and 2 3 1) have a probability of 0.03 % under the
Gaussian core alone, and of 2.4 % under the law of the running example, whose tails hold 12 % of the sites.

Three nested variants are fitted: `:core` (``s``), `:late` (``s, \pi_L, m_L``) and `:both` (all five). Comparing their
``\Delta\ell`` tells whether the orders call for tails, and on which side.

!!! details "The density"
    A Gaussian of sd ``s`` convolved with an exponential of rate ``\lambda = 1/m`` (an exponentially modified
    Gaussian) has the density
    ``\lambda \exp\!\big(\lambda(\lambda s^2/2 - x)\big)\, \Phi\big((x - \lambda s^2)/s\big)``, with ``x = z`` for the
    late tail and ``x = -z`` for the early one. The mixture is evaluated in log space on `DELAY_GRID` and
    renormalised there.

**What the orders can identify.** The five parameters of ``f`` are not what the orders measure. The orders do not
see the delay of a site, only whom it overtakes. What they identify is the **overtaking profile**:

```math
q_d = P(\text{site } i + d \text{ commits before site } i) = P(e_i - e_{i+d} > d), \qquad d = 1, 2, 3, \dots
```

- **The profile is read directly off the stems.** With independent delays, whether the organs of sites ``i`` and
  ``i + d`` are inverted on the stem depends only on their own two delays. So ``q_d`` is the probability that any two
  sites ``d`` apart are inverted, whatever the other sites do, and the fraction of inverted pairs estimates it. Every
  delay law that predicts the same orders has the same profile. ``q_1`` is the rate of neighbour swaps; ``q_2`` and
  ``q_3`` are the longer displacements, which are where tails show.
- **It has no direction.** ``q_d`` depends on ``f`` only through the law of a *difference* of two delays, which is
  the same whether the off-time sites are late or early. The late/early split rests on rarer patterns of three
  organs (2 3 1 against 3 1 2), and is only weakly identified.
- **The parameters of ``f`` are readings of its shape.** The core sd ``s``, the tail masses ``\pi`` and ``m``, and
  properties of ``f`` itself such as the **off-time fraction** ``P(|e| > 1)`` (`tail_masses`) are not seen directly.
  They follow from the profile only through the assumed shape (a Gaussian core with exponential tails). When that
  shape is unambiguous they are well determined: for light tails, the core and the off-time fraction are recovered.
  When it is not, they drift with no change in the orders. Without tails, a short "tail" holding most of the sites
  only reshapes the core. With heavy tails, a narrower core plus more tail mass predicts the same orders, and the core
  collapses. In both cases the profile does not move.

So the profile is the result to report and to compare between groups (`overtaking_profile` reads it from the
posterior orders, with no delay law; `delay_law_inference` gives that of the fitted law). The fitted law interprets
it: how precise a typical commitment is, and how many sites are off time, within the assumed shape.

The validation ([§6.4](validation.md)) measures each of these statements on simulated stems.

## 5.7 Fit, uncertainty and comparisons

- **Fit.** `fit_delay_law(samples, variant)` maximises ``\Delta\ell`` by Newton's method with a trust region, on
  unconstrained parameters: ``\log s``; for `:both`, ``\operatorname{logit}(\pi_L + \pi_E)``, the logit of the late
  share ``\pi_L/(\pi_L + \pi_E)``, ``\log m_L`` and ``\log m_E``. The tail mass and the late share are separate
  coordinates because the first is better determined than the second. The grid is fixed, so ``\Delta\ell``
  is a smooth function of the parameters. Its gradient and Hessian are exact: forward-mode dual numbers pass through
  the delay density and the recursion of [§5.4](timing.md).
- **Robust standard errors.** The stem is the unit of replication: each stem contributes its own term
  ``\Delta\ell_s``, and stems may differ from each other more than one shared law says. `delay_law_inference`
  therefore measures the uncertainty from the spread of the per-stem slopes ``g_s`` of these terms, not from the
  curvature ``H`` of their sum alone. This is the sandwich covariance
  ``H^{-1}\big(\sum_s g_s g_s^\top\big)H^{-1}``, with exact derivatives. The delta method then gives the standard
  errors of ``s``, of the tail masses and of the profile ``q_1, \dots, q_4`` (`overtaking`).
- **The profile without a delay law.** `overtaking_profile(samples)` averages, over each stem's posterior samples,
  the fraction of pairs ``d`` apart that are inverted, and pools the stems. Its standard errors are clustered by
  stem. It needs no delay law, so it is also a check of the fitted one: the two profiles should agree.
- **Two groups** (a mutant and its co-segregant wild type). Compare their **profiles**: `compare_profiles(a, b)`
  gives the differences ``q_d`` by lag with standard errors, and a Wald test over ``q_1, q_2, q_3``. The profiles can
  come from `overtaking_profile` (no delay law) or from each group's fitted law. Separate laws can also be compared
  with one shared law by the difference of their summed ``\Delta\ell`` (a likelihood ratio). The terms of ``A``
  cancel in these differences, because each stem's term is relative to its own group's fit of ``A``.
- **A shared core.** To read the off-time fraction of a group with heavy tails, whose own core sd is not identified
  (§5.6), the groups can be
  fitted together with **one core sd and their own tails**: `fit_delay_law([smp_wt, smp_mutant], :both)`. This states
  a biological hypothesis, *the mutant changes the off-time organs, not the precision of a typical commitment*, and
  the group with light tails carries the core. `shared_core_test` tests it, by the likelihood ratio of one core
  against one per group (χ² with one degree of freedom per extra group; a permutation of the group labels among the
  stems calibrates it when the tails are heavy). `delay_law_inference` on the joint fit gives each group's off-time
  fraction and their difference, with standard errors that account for the shared core. Check that the shared core
  stays close to the light-tailed group's own: if it collapses instead, that group does not hold the core, and the
  separate fits should be reported ([§6.5](validation.md)).
- **Delays in hours.** The delays are in plastochrons. The `Plastochron` module estimates the plastochron at each
  flower rank from daily counts of open flowers, to convert them.

**Code.** `DelayLaw(s; πL, mL, πE, mE)`, `rand(rng, f)`, `delay_logdensity(f)`, `tail_masses(f; a = 1)`;
`loglik_order_exact(σ, f)` (or `(σ, logg, grid, γ)`); `order_samples(order_posterior(m, fit; nsamples), η)`,
`stem_terms(samples, f)`, `relative_loglik(samples, f)` (`stem_terms(samples, logP)` scores any law of the order given
as ``\sigma \mapsto \log P(\sigma)``); `fit_delay_law`, `delay_law_inference`, `shared_core_test`; the profile:
`overtaking(f)`, `overtaking_profile(samples)`, `compare_profiles(a, b)`;
`simulate_stems(f::DelayLaw; ...)`;
`flowering_rate`, `fit_flowering`, `plastochron_by_rank`, `plant_plastochron`, `relative_plastochron`.
