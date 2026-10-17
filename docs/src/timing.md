# 5. Commitment timing

The propensity model reconstructs the orders of the stems. This page turns to the question of [§1.4](problem.md): what
do these orders say about the *timing* of organ initiation? It fits the law of the commitment delay, using the orders
of [§4](propensity.md) as an intermediate step.

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
produce. It fits a flexible law ([§5.3](timing.md)) and lets the orders decide its shape.

## 5.2 The law of the order of a paced meristem

Recall the mechanism of [§1.4](problem.md): site ``i`` is ready at time ``i`` (plastochrons), it commits at
``\tau_i = i + e_i`` with independent delays ``e_i \sim f``, and the order on the stem is the ranking of the
``\tau_i``. Only differences of delays matter, so the law ``f`` is defined up to a shift; its core is centred at 0.

What is the probability of a given order ``\sigma``? The organ at height ``p`` comes from site ``\sigma(p)``. So
``\sigma`` is the order on the stem exactly when ``\tau_{\sigma(1)} < \tau_{\sigma(2)} < \dots < \tau_{\sigma(n)}``.
Call ``t_p = \tau_{\sigma(p)}`` the commitment time of the organ at height ``p``. Its delay is ``t_p - \sigma(p)``, and

!!! note "result — the Level-0 order law"
    ```math
    P_f(\sigma) = \int_{t_1 < t_2 < \dots < t_n} \prod_{p=1}^{n} f\big(t_p - \sigma(p)\big)\, dt_1 \cdots dt_n .
    ```

("Level 0": the sites commit independently of each other. A paced meristem where a late site also delays the next
ones would be a "Level 1".)

This integral looks like the product form of [§3.1](orders.md), one factor per organ, depending on ``p`` and
``\sigma(p)``. But the factors are tied together by the ordering constraint ``t_1 < \dots < t_n``. Whether the organ
of site ``j`` sits at height ``p`` depends on how many other sites committed before it: on all the other delays at
once. A product ``\prod_p h_{p,\sigma(p)}`` cannot express this.

Here is one consequence that can be checked. In product form, the odds of a swap at heights 1–2 do not depend on
whether heights 3–4 are swapped:

```math
\frac{P(2\,1\,4\,3)}{P(1\,2\,4\,3)} = \frac{P(2\,1\,3\,4)}{P(1\,2\,3\,4)} ,
```

because both sides equal ``e^{\eta_{+1} + \eta_{-1}}`` (the factors of heights 3 and 4 cancel). Under Level 0 the two
ratios differ. With Gaussian delays of sd 0.42 they are 0.0487 and 0.0508 (−4 %). With a WT-like law that has tails
([§6](validation.md)) they are 0.089 and 0.087 (+3 %). A swap makes the next swap slightly less likely in the first
case, and slightly more likely in the second.

So the order law of a paced meristem is outside the family of laws for which the sum over orders of [§3](orders.md)
is exact, and ``p(y \mid f) = \sum_\sigma P_f(\sigma)\,p(y \mid \sigma)`` cannot be computed by that sum. Note,
though, how small the deviation is. A product-form law cannot be ``P_f``, but it can come close. This will matter
below.

## 5.3 One order at a time is easy

There is, however, a fast and exact way to compute ``P_f(\sigma)`` for *one* given order. Integrate the times one height
at a time, from the bottom:

!!! note "result — forward recursion over the heights"
    Let ``a_p(t)`` be the density of ``t_p`` jointly with the event ``t_1 < \dots < t_p``. Then
    ```math
    a_1(t) = f\big(t - \sigma(1)\big),\qquad
    a_p(t) = f\big(t - \sigma(p)\big) \int_{-\infty}^{t} a_{p-1}(u)\, du,\qquad
    P_f(\sigma) = \int a_n(t)\, dt .
    ```

!!! details "Derivation and numerics"
    Integrate the variables of the ``n``-fold integral in the order ``t_1, t_2, \dots``. The variable ``t_{p-1}``
    appears in the factor ``f(t_{p-1} - \sigma(p-1))``, in the constraints ``t_{p-2} < t_{p-1}`` (already absorbed into
    ``a_{p-1}``) and ``t_{p-1} < t_p``. Integrating it over ``(-\infty, t_p)`` gives the cumulative of ``a_{p-1}`` at
    ``t_p``. ∎

    The functions are represented on a grid of step 0.05 plastochron (`DELAY_GRID`, from −12 to 30), and each
    ``a_p`` is renormalised as it goes (the logs of the normalisers add up to ``\log P_f(\sigma)``). The cumulative
    counts the cell that contains ``t`` for one half (midpoint rule). Counting it in full would treat two times in the
    same cell as ordered, which biases ``\log P_f(\sigma)`` upwards at first order in the step: by +0.28 for an order
    of 6 sites with three displacements. The midpoint rule leaves an error below 0.01. The tests check the two-site
    closed form ``\Phi(1/(s\sqrt2))``, that the probabilities of all 24 orders of 4 sites sum to 1, and the frequencies
    of the orders drawn from the delay law.

We cannot sum over all orders, but we can score any single order exactly, in a fraction of a millisecond. Step 1 gives
us orders: the exact posterior samples of each stem under the propensity model ([§3.7](orders.md)). It remains to see
how scoring these samples gives the likelihood of ``f``.

## 5.4 Re-weighting the reconstructed orders

The propensity model ``A`` and the delay law ``f`` describe the angles in the same way ([§2.5](angles.md)). They differ
only in the law of the order, ``P_A(\sigma)`` or ``P_f(\sigma)``. Write the likelihood of ``f`` with the sum over
orders, and use Bayes' rule under ``A`` for the angle term.

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

The intuition of [§1.5](problem.md) is exact: a delay law is good if it gives a high probability to the orders that
the angles support. The correction ``1/P_A(\sigma)`` removes the part of that support that came from ``A``'s own law of
the order. ``P_A(\sigma)`` is computable, since it is the product form of [§4](propensity.md) divided by its
permanent.

The expectation is estimated with the posterior samples of each stem. With ``\sigma_1, \dots, \sigma_M`` the distinct
sampled orders and ``w_j`` their frequencies,

```math
\Delta\ell_s(f) = \log \sum_{j} w_j\, \frac{P_f(\sigma_j)}{P_A(\sigma_j)},\qquad
\Delta\ell(f) = \sum_{\text{stems}} \Delta\ell_s(f) = \log p(y \mid f) - \log p(y \mid A) .
```

``\Delta\ell > 0`` means that the timing law explains the angles better than the free costs of the propensity model.
Two delay laws are compared by the difference of their ``\Delta\ell``. Scoring a new law takes seconds, because neither
the angles nor the permanents are recomputed.

![Re-weighting the orders of the example stem](figures/e5_example_reweighting.png)

*Figure 2. The example stem. (a) Its distinct posterior orders, with their frequencies among 200 samples. (b) The
log-weight ``\log P_f(\sigma) - \log P_A(\sigma)`` of each, under the true delay law; the true order is in orange.
(c) ``\Delta\ell`` summed over the 40 stems (black) and the term of the example stem (orange, right axis), as a
function of the core sd ``s``, the tails kept at their true values. The dashed line is the true ``s``.*

Why does this work well in practice, and not only in principle? An average over samples is precise when the
weights ``P_f(\sigma_j)/P_A(\sigma_j)`` are of similar size, so that no single sample dominates. This is the case
when ``P_A`` is close to ``P_f``. We saw in [§5.2](timing.md) that a product-form law can come close to a Level-0 law,
and the propensity model is fitted to the same orders that ``f`` must explain. Its reconstructed orders are therefore
a good basis for re-weighting. The validation measures this: the effective sample size is typically 175–200 out of
200 samples ([§6](validation.md)).

**Limits.** The angle nuisances (``\kappa``, spiral, twist) are those of the propensity model's fit. Orders outside its
band (``|d| > k``) are never sampled, so a law that puts mass on them is slightly penalised. The estimate is reliable
when the weights are spread over many samples. Its *effective sample size* is reported in the validation
([§6](validation.md)).

## 5.5 The delay law

**assumption** — the delay is a precise Gaussian core, plus two exponential tails for sites committed late or early:

```math
f = (1 - \pi_L - \pi_E)\, \mathcal N(0, s^2)
+ \pi_L\, \big(\mathcal N(0, s^2) \ast \mathrm{Exp}(m_L)\big)
+ \pi_E\, \big(\mathcal N(0, s^2) \ast (-\mathrm{Exp}(m_E))\big).
```

The core sd ``s`` is the precision of a typical commitment. A fraction ``\pi_L`` of the sites is late by an extra
exponential delay of mean ``m_L``, and a fraction ``\pi_E`` is early by ``m_E``. The tails are what a single Gaussian
cannot produce: one organ displaced by two or three positions while its neighbours are in order. Three nested variants
are fitted: `:core` (``s``), `:late` (``s, \pi_L, m_L``) and `:both` (all five). Their comparison tells whether the
orders call for tails, and on which side.

!!! details "The density"
    A Gaussian of sd ``s`` convolved with an exponential of rate ``\lambda = 1/m`` (an exponentially modified
    Gaussian) has the density
    ``\lambda \exp\!\big(\lambda(\lambda s^2/2 - x)\big)\, \Phi\big((x - \lambda s^2)/s\big)``, with ``x = z`` for the
    late tail and ``x = -z`` for the early one. The mixture is evaluated in log space on `DELAY_GRID` and
    renormalised there.

**result** (from the validation, [§6](validation.md)) — what the orders identify. A frequent short tail and a rarer
long one predict nearly the same orders, so ``\pi`` and ``m`` are poorly identified one by one. The quantities to
report are the **tail masses** ``P(e > 1)``, ``P(e < -1)`` and their sum ``P(|e| > 1)``: the fractions of sites
committed more than one plastochron late, early, or either. The off-time fraction ``P(|e| > 1)`` is the best
identified. When the tails are light, the core sd ``s`` and the split between late and early are identified too. When
they carry a large share of the sites (tens of percent at about one plastochron), a narrower core plus more tail mass
predicts nearly the same orders. ``s`` and the late/early split are then no longer identified, even though their
standard errors may look small.

## 5.6 Fit, uncertainty and comparisons

- **Fit.** `fit_delay_law(samples, variant)` maximises ``\Delta\ell`` by Nelder–Mead on unconstrained parameters:
  ``\log s``; for `:both`, ``\operatorname{logit}(\pi_L + \pi_E)``, the logit of the late share
  ``\pi_L/(\pi_L + \pi_E)``, ``\log m_L`` and ``\log m_E``.
- **Robust standard errors.** The stem is the unit of replication. `delay_law_inference` computes the sandwich
  covariance ``H^{-1}\big(\sum_s g_s g_s^\top\big)H^{-1}`` from the per-stem scores ``g_s`` and the Hessian ``H`` of
  ``\Delta\ell`` (central differences). It stays valid when the stems differ from each other more than the model says.
  The delta method then gives the standard errors of ``s`` and of the tail masses.
- **Two groups** (a mutant and its co-segregant wild type). Fit each group separately, then compare the off-time
  fractions with their standard errors (independent groups), or the summed ``\Delta\ell`` of separate laws and of one
  shared law (a likelihood ratio). The terms of ``A`` cancel in such differences, because each stem's term is relative
  to its own group's fit of ``A``.
- **Delays in hours.** The delays are in plastochrons. The `Plastochron` module estimates the plastochron at each
  flower rank from daily counts of open flowers, to convert them.

**Code.** `DelayLaw(s; πL, mL, πE, mE)`, `rand(rng, f)`, `delay_logdensity(f)`, `tail_masses(f; a = 1)`;
`loglik_order_exact(σ, f)` (or `(σ, logg, grid, γ)`); `order_samples(order_posterior(m, fit; nsamples), η)`,
`stem_terms(samples, f)`, `relative_loglik(samples, f)` (`stem_terms(samples, logP)` scores any law of the order given
as ``\sigma \mapsto \log P(\sigma)``); `fit_delay_law`, `delay_law_inference`; `simulate_stems(f::DelayLaw; ...)`;
`flowering_rate`, `fit_flowering`, `plastochron_by_rank`, `plant_plastochron`, `relative_plastochron`.
