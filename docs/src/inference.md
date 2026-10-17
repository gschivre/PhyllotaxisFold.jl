# 7–8. Inference and limits

## 7. Inference

### 7.1 Fitting the propensity model

- **Parameters on the real line.** Positive parameters go through a log, probabilities through a logit, and the costs
  ``\eta`` through the softplus map of their gaps ([§4.3](propensity.md)). The log-Jacobian of these maps is part of
  the log density.
- **Maximum a posteriori by Newton's method with a trust region.** The value, the gradient and the Hessian are
  computed in one call, and the Hessian only when the optimiser asks for it. The gradient is analytic
  ([§3.6](orders.md)). The Hessian is one pass of forward-mode dual numbers through the analytic gradient. A
  non-finite evaluation counts as a rejected step.
- **Uncertainty.** Laplace approximation: ``-H^{-1}`` at the maximum is the covariance of the unconstrained
  parameters.
- **Model comparison.** The integrated log-likelihood at the maximum (`loglik` of `fit_map_torsion!`, priors
  excluded). Timing laws are compared with the propensity model and with each other by ``\Delta\ell``
  ([§5.4](timing.md)).

### 7.2 Integrating the twist out

Each stem has its own twist rate ``\omega_s \sim \mathcal N(m_\omega, s_\omega^2)`` ([§2.3](angles.md)), and its
likelihood must be integrated over it. This integral is harder than it looks. The likelihood of one stem is very sharp
in ``\omega_s`` (about 0.1°/mm wide). It is also often multimodal: a different twist combined with a few permutations
can explain the same angles. A fixed grid of nodes would need to be very fine. Gauss–Hermite nodes placed on the
population law would miss the peaks.

The quadrature is therefore *adaptive*, built on each stem's own likelihood.

1. Scan the rate, find the local maxima within 8 log-units of the best, and measure the curvature of each.
2. Place Gauss–Hermite nodes on each mode (a mixture of Gaussians ``q_s``).
3. Weight each node so that the rule integrates *any* population law: ``\int L_s(\omega)\, p(\omega)\, d\omega
   \approx \sum_j w_j\, L_s(\omega_j)\, p(\omega_j)/q_s(\omega_j)``.

The nodes stay fixed during a Newton fit, so the objective is a smooth function of all the parameters, including
``(m_\omega, \log s_\omega)``. `fit_map_torsion!` alternates the two:

1. fit without twist;
2. place the nodes;
3. fit with each stem at its best mode;
4. place the nodes again;
5. fit the integrated objective.

It then optionally places the nodes once more and refits.

!!! details "Why the weights are right"
    For component ``r`` of the mixture (weight ``\pi_r``, centre ``c_r``, sd ``h_r``), the Gauss–Hermite rule gives
    ``\mathbb E_{\mathcal N(c_r, h_r^2)}[g] \approx \sum_j \gamma_j\, g(c_r + h_r z_j)``. Writing
    ``\int L p = \int (L p / q)\, q = \sum_r \pi_r\, \mathbb E_{\mathcal N_r}[L p / q]`` and applying the rule to each
    component gives node weights ``\pi_r \gamma_j / q(\omega_{rj})``. This is multiple importance sampling with the
    balance heuristic. It is exact for any ``p`` in expectation, and accurate when ``L_s`` is close to Gaussian around
    each mode and ``p`` varies slowly on the scale of the mode.

### 7.3 Cost and accuracy

- **Cost.** One stem of about 45 organs takes about 50 ms (2048 nuisance nodes). A fit of 50–90 stems with the twist
  integrated takes 10–15 minutes on 18 threads. The cost is proportional to `ngrid_ϕ` × `ngrid_κ`.
- **Accuracy.** The band ``k = 4`` is checked by the posterior mass at ``|d| = k``. 32 ``\phi`` nodes give converged
  estimates; use 64 for fine comparisons of log-likelihoods. 16 ``\kappa`` nodes are accurate while
  ``\sigma_\kappa \lesssim 0.5``.

## 8. Assumptions and limits

Every assumption of the manual, and what would show that it fails:

| assumption | where | what would break it, and how to check |
|---|---|---|
| organs sit on a regular spiral; noise acts on timing, not on positions | [§2.1](angles.md) | missed organs are not modelled; without `outlier = true`, a grossly wrong angle appears as a long displacement |
| von Mises angular noise, one precision per stem | [§2.2](angles.md) | a broad but not uniform error (e.g. angles off by a fraction of the divergence) |
| one twist rate per stem, Normal across stems | [§2.3](angles.md) | a tapering stem makes the rate drift along the stem |
| displacements bounded by ``k = 4`` | [§3.2](orders.md) | posterior mass at ``\lvert d\rvert = k``; refit with a larger ``k`` |
| sites ready at a regular pace commit independently, with i.i.d. delays (Level 0) | [§5.2](timing.md) | cascades of permutations, a drifting timeline; stems that differ in their off-time rate |
| the same delay law all along the stem | [§5.2](timing.md) | an off-time rate that changes along the stem (e.g. at the base) |
| the delay law is a Gaussian core with exponential tails | [§5.5](timing.md) | discrete slot errors (organs exactly one plastochron off) are fitted as short tails |
| the timing law is scored at the propensity model's angle nuisances and band | [§5.4](timing.md) | a small effective sample size of the weights; refit with a larger ``k`` |
| flowers open at the pace they were initiated; piecewise-linear opening rate | `Plastochron` | a maturation delay that changes during flowering; a curved rise or decline |

Some questions are beyond the reach of the angles alone:
- they confound angular noise with permutations (a lower ``\kappa`` or more permutations);
- they identify the shape of the delay law only through the tail masses ([§5.5](timing.md));
- they cannot separate a noisier readiness from a noisier commitment, since only their sum enters the order.

Timing data (internodes, imaging of primordia) are needed for those questions.
