# 5. Commitment timing

Step 1 gives, for every stem, the exact posterior of its order and as many samples of it as we like
([§3.7](orders.md)). Step 2 asks the question of [§1.4](problem.md): *which delay law ``f`` explains these orders?* In
four moves:

1. the law of the order that a delay law predicts, ``P_f(\sigma)`` ([§5.2](timing.md));
2. why the exact sum of [§3](orders.md) cannot use it, and why this is a small obstacle ([§5.3](timing.md));
3. the exact score of a single order ([§5.4](timing.md));
4. the likelihood of ``f``, by re-weighting the orders of step 1 ([§5.5](timing.md)).

The delay law itself ([§5.6](timing.md)) and its fit ([§5.7](timing.md)) come last. But first, why should a delay be
random at all?

## 5.1 Why a random delay

**The switch.** In each cell of the peripheral zone of the meristem, a regulatory network downstream of auxin
signalling reads an *input*: the strength of auxin signalling in that cell. The input rises as auxin accumulates at
the site, and it is lowered near young primordia, which inhibit the emergence of new ones around them. *Commitment* is
the switch of the network to the organ-founder state, a switch that is not undone when the input falls back. In
practice, a site has committed once removing auxin no longer prevents the organ from forming. The visible signs of the
organ (a marker turning on, a bulge) come later, but that extra lag does no harm: its common part shifts every site
alike, which leaves the order unchanged ([§5.2](timing.md)), and its jitter simply adds to ``f``.

**Why a fold.** The **assumption** is that the network has two stable states over a range of inputs, uncommitted and
committed, and that the rising input makes the uncommitted state disappear. Positive feedback readily builds such a
switch, as in the all-or-none maturation of *Xenopus* oocytes (Ferrell 2012). Nothing else is assumed: not the genes,
nor their number, nor their kinetics. When a smoothly rising input makes a stable state disappear, the generic way is a
*fold*: the state merges with the threshold that separates it from the committed state, and both vanish (figure 1). The
other ways need special conditions, a symmetry or a finely tuned network (Sáez et al. 2022). The fold also explains why
commitment lasts: the committed state is still there when the input falls back, so the cell stays in it (Ferrell 2012).

!!! details "Why a fold is generic"
    Write the network as ``d\mathbf y/dt = \mathbf F(\mathbf y; u, \boldsymbol\theta)``, with ``\mathbf y`` its
    concentrations, ``u`` the input and ``\boldsymbol\theta`` its gains, cooperativities, turnover rates and growth
    dilution. A stable state disappears (rather than just losing stability) when it collides with an unstable one, at
    a zero eigenvalue of the Jacobian ``\partial_{\mathbf y}\mathbf F``. With a single parameter varying, the input,
    the generic collision is a saddle-node at ``u = u_{\rm on}``: it has codimension one and needs no symmetry. A cusp,
    where bistability itself is born, needs a second parameter tuned; a pitchfork needs a symmetry of
    ``\mathbf F``. The fold also survives changes of the network: if a second parameter varies too (between cells, or
    in a mutant), the folds form whole lines in the plane of the two parameters, while the other bifurcations sit at
    single points. A path through that plane crosses fold lines without any tuning, but hits one of those points only
    by chance (Sáez et al. 2022). The fold is assumed non-degenerate: its curvature is not zero, and raising the input
    moves the system through the fold rather than just touching it.

**One slow variable.** This is a **result**: near a fold, the whole network behaves as a single slow variable. All
directions but one relax quickly, and along the remaining one the dynamics take the same form for every network,

```math
dX = (\mu + X^2)\, ds + dW_s .
```

``X`` is the position along the slow direction, ``dW_s`` is the molecular noise felt along it, and ``\mu`` is the
*margin*: how far the input is above the fold. ``X``, ``\mu`` and the time ``s`` are measured in units set by the
noise (box). Below the fold (``\mu < 0``), the speed ``\mu + X^2`` vanishes at two points: the uncommitted state and
the threshold. At the fold they merge, and above it (``\mu > 0``) they are gone (figure 1). Commitment is the moment
``X`` passes the *point of no return*, beyond which the cell reaches the committed state even if the input falls back.

!!! details "Derivation"
    At the fold, ``\partial_{\mathbf y}\mathbf F`` has a single zero eigenvalue. The other directions relax on their
    own time scales, and the dynamics collapse onto a one-dimensional centre manifold. With ``x`` the coordinate
    along it (projected on the left null vector, centred at the fold) and ``\Delta = (u - u_{\rm on})/u_s`` the
    margin, a Taylor expansion gives
    ``dx/dt = k_m(\Delta + a_f x^2 + b_1 x\Delta + b_3 x^3 + \dots)``, where ``k_m`` is the rate of the slow mode and
    ``a_f > 0`` the curvature of the fold. The two leading terms are the same for every network that satisfies the
    assumption; the coefficients ``b`` are specific to each network, and to leading order they only add a constant
    to the commitment time. Noise in every component, and fluctuations of the input, project onto ``x`` with an
    intensity ``S_0``: ``dx = k_m(\Delta + a_f x^2)\,dt + \sqrt{k_m S_0}\,dW_t``. With the noise units
    ``x_S = (S_0/a_f)^{1/3}``, ``t_S = (a_f^2 S_0)^{-1/3}/k_m`` and ``\Delta_S = (a_f S_0^2)^{1/3}``, the variables
    ``X = x/x_S``, ``s = t/t_S`` and ``\mu = \Delta/\Delta_S`` obey ``dX = (\mu + X^2)\,ds + dW_s``. This is an
    asymptotic statement, exact as ``\Delta`` and ``x`` go to 0.

![The fold](figures/m0_fold.png)

*Figure 1. The speed of the switch variable, ``dX/ds = \mu + X^2``, for three margins. Left, before the fold: a stable
uncommitted state (blue) and a threshold (red circle). Middle: the two merge, at the fold. Right, after the fold: no
state is left, but the speed stays small near ``X = 0``. The time spent there sets the commitment delay.*

**Slow from fast.** Another **result**: after the fold, no state is left, but the speed stays small near ``X = 0``,
where the uncommitted state used to be. The cell lingers there before it escapes to the committed state, and the
smaller the margin, the longer it lingers: the passage takes a time proportional to ``1/\sqrt\mu``. This is how a
network whose reactions are all fast can produce a slow commitment.

The same slowness shows up just *before* a fold: a system knocked off its state returns ever more slowly as the
threshold nears. This *critical slowing down* is used to anticipate tipping points (Scheffer et al. 2009), and it has
been measured in a living system: a cyanobacterial population pushed towards a fold (Veraart et al. 2012).

!!! details "Derivation"
    Without noise and at a constant margin ``\Delta > 0``, the time from the uncommitted state ``x_0 < 0`` to the
    point of no return ``x_c > 0`` is
    ``T = \frac{1}{k_m}\int_{x_0}^{x_c} \frac{dx}{\Delta + a_f x^2} \simeq \frac{1}{k_m}\Big[\frac{\pi}{\sqrt{a_f\Delta}} - C\Big]``,
    with ``C \simeq (1/|x_0| + 1/x_c)/a_f`` plus network-specific corrections. In noise units, the first term is
    ``\pi/\sqrt\mu``. The time is spent near ``x = 0``, the "ghost" of the vanished state, where the flow almost
    stops.

**Noise makes the delay random.** A third **result**: near the ghost the speed is small, so a small push from the
molecular noise moves the moment of escape a lot. The closer to the fold, the longer the passage, and its spread grows
even faster than its mean. In noise units, the relative spread of the delay depends on the margin ``\mu`` alone, and
increases as ``\mu`` shrinks. The randomness of the delay is thus a property of the switch itself, not an
imperfection added to it.

!!! details "Derivation"
    For a margin well above the noise scale (``\mu \gg 1``), follow the deterministic path: a push ``\delta x`` at a
    point where the speed is ``v`` shifts the passage by ``\delta x / v``. Adding up the noise along the path,
    ``\mathrm{Var}(T) \simeq \int k_m S_0\, v(x)^{-3}\, dx`` with ``v = k_m(\Delta + a_f x^2)``, so
    ``\mathrm{sd}(T) \simeq \frac{\sqrt{S_0}}{k_m}\sqrt{\tfrac{3\pi}{8}}\, a_f^{-1/4}\Delta^{-5/4}``. The mean grows
    as ``\Delta^{-1/2}`` and the sd as ``\Delta^{-5/4}``; in noise units, the coefficient of variation is
    ``\mathrm{sd}/\text{mean} \simeq \sqrt{3/(8\pi)}\,\mu^{-3/4}``. Fluctuations of the input itself (for
    instance bursts of transcription upstream of the network) act the same way, as long as they are fast compared with
    the passage: they simply add to the noise ``S_0``.

**Why near the fold.** The fold makes the delay slow and variable only if cells commit close to it, at a small margin.
The switch alone does not force this, so it is an **assumption**, but a natural one. Auxin accumulates at a site much
more slowly than the network in each cell responds, and a switch driven this slowly lags just behind its input: the cell
commits shortly after the uncommitted state has disappeared, while the margin is still small (**result**, box below).
Such slow passages through a bifurcation are known to shape cell fates in morphogen gradients, where *when* a cell
crosses, not only how far the input goes, decides the outcome (Wedgwood & Ashwin 2022).

The meristem points the same way. A new primordium inhibits the emergence of primordia around it, which keeps the input
of the nearby sites from rising far above threshold. And cells of the meristem integrate auxin over time before changing
fate: the transcriptional response is out of phase with the build-up of auxin, and longer auxin treatments give stronger
responses (Galvan-Ampudia et al. 2020). A switch near its fold behaves this way: a brief rise above threshold is not
enough, since the cell must cross the slow region before the input falls back. With noise, this becomes a matter of
odds: the longer the input stays above threshold, the more likely the cell commits.

!!! details "Derivation"
    Without noise, let the margin rise through the fold, ``\Delta(t) = \dot\Delta\, t``, in
    ``dx/dt = k_m(\Delta + a_f x^2)``. Rescaling time by ``T_0 = (k_m^2 a_f \dot\Delta)^{-1/3}`` gives
    ``dX/d\tau = \tau + X^2``, whose solution from the uncommitted state escapes at a fixed ``\tau`` (a zero of the
    Airy function). The cell therefore commits a time of order ``T_0`` after the fold, at a margin of order
    ``(\dot\Delta/k_m)^{2/3}``, which is small whenever the input rises slowly compared with the switch: the classical
    delay of a slow passage through a fold. Noise spreads the passage around this time; strong enough noise can even
    trigger the escape a little before the fold, but it never makes the margin larger (Berglund & Gentz 2002).

**Delays in the network.** Some steps take time by themselves (a transcript must be elongated and matured; the
feedback that locks the switch may lag). A delay upstream of the switch shifts every commitment equally and leaves the
order unchanged. A delay inside the feedback slows the switch: the commitment delay gets longer and, relative to its
mean, a little less variable. Either way, the picture of a slow and random delay stands.

!!! details "Derivation"
    A pure lag of the input, ``u(t) \to u(t - \tau_d)``, shifts commitment by ``\tau_d``. A distributed delay acts,
    to first order, as a lag equal to its mean; it also smooths the rise of the input, which changes the shape of the
    delay law only when ``\tau_d`` is not short compared with the passage. A delay ``\tau_d`` inside the positive
    feedback does not move the fold but slows the switch, ``k_m \to k_m/(1 + c)`` with ``c = k_{\rm fb}\tau_d`` and
    ``k_{\rm fb}`` the slope of the feedback at the fold. To first order, the mean delay is multiplied by ``1 + c``
    and its sd by ``\sqrt{1 + c}``, so the coefficient of variation decreases. These are first-order results
    (``k_{\rm fb}\tau_d \ll 1``); long delays can change the switch itself.

From this picture the package keeps only what [§1.4](problem.md) assumed: each site commits after a random delay,
with the same law for every site. A fold is not the only source of such a delay (a long cascade of steps, or a precise
switch reading a jittery input, give one too), so the package does not impose the delay law of a fold: it fits a
flexible law ([§5.6](timing.md)) and lets the orders decide its shape. Any other law, a fold's included, can be put in
its place ([Custom delay laws](custom_laws.md)).

## 5.2 The law of the order of a paced meristem

Recall the mechanism of [§1.4](problem.md): site ``i`` is ready at time ``i`` (in plastochrons), it commits at
``\tau_i = i + e_i`` with independent delays ``e_i \sim f``, and the order on the stem is the ranking of the
``\tau_i``. Only differences of delays matter, so ``f`` is defined up to a shift, and its core is centred at 0.
Ranking items by independent noisy times is a classical model: Thurstone's law of comparative judgment
(Thurstone 1927), here with commitment times as the noisy values.

**Two sites first.** Sites 1 and 2 are swapped when site 2 commits first: ``2 + e_2 < 1 + e_1``, that is
``e_1 - e_2 > 1``. If the delays are Gaussian with sd ``\sigma_c``, their difference is Gaussian with sd ``\sigma_c\sqrt 2``, and

```math
P(2\,1) = \Phi\!\left(-\frac{1}{\sigma_c\sqrt 2}\right).
```

With ``\sigma_c = 0.42`` plastochron, this is 4.6 %: about one pair of neighbours in twenty. The same reasoning shows why a
Gaussian alone is not the whole story: a displacement by two needs ``e_i - e_{i+2} > 2``, of probability
``\Phi(-2/(\sigma_c\sqrt 2)) \approx 4 \times 10^{-4}``, one pair in 2600. Real stems have many more ([§5.6](timing.md)).

**Any number of sites.** The organ at height ``p`` comes from site ``\sigma(p)``. So ``\sigma`` is the order on the
stem exactly when ``\tau_{\sigma(1)} < \tau_{\sigma(2)} < \dots < \tau_{\sigma(n)}``. Call
``t_p = \tau_{\sigma(p)}`` the commitment time of the organ at height ``p``. Its delay is ``t_p - \sigma(p)``, and the
probability of the order is the probability of this chain of inequalities:

!!! note "result — the order law of independent delays"
    ```math
    P_f(\sigma) = \int_{t_1 < t_2 < \dots < t_n} \prod_{p=1}^{n} f\big(t_p - \sigma(p)\big)\, dt_1 \cdots dt_n .
    ```

For two sites, this integral is the ``\Phi`` above.

**What the delay covers.** Independent delays are less restrictive than they look. If the readiness of site ``i``
wanders to ``i + u_i`` and the site commits ``c_i`` later, then ``\tau_i = i + (u_i + c_i)``: the same model with
``e_i = u_i + c_i``, as long as the ``u_i`` are independent. So ``e_i`` is the deviation of site ``i`` from a regular
clock, and ``f`` the total timing noise around it; the orders cannot split it between readiness and commitment
([§7](limits.md)).

What the model does not cover are timing errors that build on each other: a clock whose intervals vary, so that its
errors add up like a random walk, or a late site that also delays the next ones. Both make the delays of neighbouring
sites dependent ([§7](limits.md) lists the signs). [§6.7](validation.md) shows when a drifting clock starts to be read
as tails.

## 5.3 Not a product, but close to one

The integral has one factor per organ, as in the product form of [§3.1](orders.md), but the chain
``t_1 < \dots < t_n`` ties them together: where the organ of site ``j`` ends up depends on all the other delays at
once. Can a product ``\prod_p h_{p,\sigma(p)}`` still reproduce ``P_f``? A simple test says no.

**The test.** In a product form, a swap has a fixed price. Swapping the organs at heights 1 and 2 multiplies the
probability by the same factor, ``e^{\eta_{+1} + \eta_{-1}}``, whatever happens at heights 3 and 4 (their factors
cancel in the ratio):

```math
\frac{P(2\,1\,3\,4)}{P(1\,2\,3\,4)} = \frac{P(2\,1\,4\,3)}{P(1\,2\,4\,3)} .
```

Under ``P_f``, with Gaussian delays of sd 0.42, the four orders have exact probabilities (by the recursion of
[§5.4](timing.md); code in the box below):

| order | 1 2 3 4 | 2 1 3 4 | 1 2 4 3 | 2 1 4 3 |
|---|---|---|---|---|
| ``P_f(\sigma)`` | 0.8635 | 0.0439 | 0.0439 | 0.00214 |

The odds of the first swap are ``0.0439 / 0.8635 = 0.0508`` when heights 3–4 are in order, but
``0.00214 / 0.0439 = 0.0487`` when they are swapped: **4 % lower**. No product form can do this.

The coupling comes from the boundary between the pairs. The organs at heights 2 and 3 must commit in order, and
*which* sites they hold depends on both swaps (sites 2 and 3 in 1 2 3 4, sites 1 and 4 in 2 1 4 3). So one swap
depends a little on the other. The sign is not fixed: with the tailed law of the running example
([home page](index.md)), the two odds are 0.0913 and 0.0941, **3.1 % higher** when heights 3–4 are swapped. A product
form, with one price per swap, can follow neither effect.

!!! details "Reproducing the numbers"
    ```julia
    using PhyllotaxisFold
    P(σ, f) = exp(loglik_order_exact(σ, f))                                # exact probability of an order under f (§5.4)
    odds(f) = (P([2, 1, 3, 4], f) / P([1, 2, 3, 4], f), P([2, 1, 4, 3], f) / P([1, 2, 4, 3], f))
    gauss = GaussianExpTails(0.42)                                         # Gaussian delays, sd 0.42 plastochron
    ex = GaussianExpTails(0.42; πL = 0.06, mL = 1.5, πE = 0.06, mE = 1.0)  # the law of the running example
    P([2, 1], gauss)                                                       # 0.046 (§5.2)
    odds(gauss)                                                            # (0.0508, 0.0487)
    odds(ex)                                                               # (0.0913, 0.0941)
    # the orders 3 1 2 and 2 3 1 (§5.6): 0.0003 under gauss, 0.024 under ex
    P([3, 1, 2], gauss) + P([2, 3, 1], gauss), P([3, 1, 2], ex) + P([2, 3, 1], ex)
    ```

Two consequences:

- **a limitation**: ``P_f`` is outside the family for which the sum of [§3](orders.md) is exact, so
  ``p(y \mid f) = \sum_\sigma P_f(\sigma)\, p(y \mid \sigma)`` cannot be computed that way;
- **an opportunity**: the deviation is a few percent. A product-form law cannot *be* ``P_f``, but it can come close, and
  the propensity model, fitted to the same stems, is such a law. The next two sections exploit this.

## 5.4 The exact probability of one order

``P_f`` cannot be summed over all orders, but it is exact and fast for *one* order. As in [§3](orders.md), go up the
stem one height at a time, carrying only what the next height needs: here, the distribution of the commitment time of
the organ just placed.

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
    closed form ``\Phi(1/(\sigma_c\sqrt2))``, that the probabilities of all 24 orders of 4 sites sum to 1, and the frequencies
    of the orders drawn from the delay law.

A stem of 45 organs is scored in a fraction of a millisecond, and step 1 provides exactly the orders worth scoring:
the posterior samples of each stem. How do their scores become the likelihood of ``f``?

## 5.5 Re-weighting the reconstructed orders

The propensity model ``A`` and the delay law ``f`` describe the angles in the same way ([§2.5](angles.md)) and differ
only in the law of the order. So write the likelihood of ``f`` as a sum over orders, and let Bayes' rule under ``A``
turn the angle term into the posterior of step 1.

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

This is the intuition of [§1.5](problem.md), made exact: a good delay law gives a high probability to the orders the
angles support. Dividing by ``P_A(\sigma)`` (the product form of [§4](propensity.md) over its permanent) removes the
part of that support that came from ``A``'s own law of the order, so that it is not counted twice.

**In practice.** The expectation is an average over the posterior samples of the stem. With
``\sigma_1, \dots, \sigma_M`` the distinct sampled orders and ``w_j`` their frequencies,

```math
\Delta\ell_s(f) = \log \sum_{j} w_j\, \frac{P_f(\sigma_j)}{P_A(\sigma_j)},\qquad
\Delta\ell(f) = \sum_{\text{stems}} \Delta\ell_s(f) = \log p(y \mid f) - \log p(y \mid A) .
```

``\Delta\ell > 0`` means that the timing law explains the angles better than the free costs of the propensity model.
Two delay laws are compared by the difference of their ``\Delta\ell``, in which ``A`` cancels. Scoring a new law takes
seconds: neither the angles nor the permanents are recomputed.

![Re-weighting the orders of the example stem](figures/e5_example_reweighting.png)

*Figure 2. The example stem; its true order is in orange. (a) Its distinct posterior orders and their frequencies
among 200 samples. (b) The log-weight ``\log P_f(\sigma) - \log P_A(\sigma)`` of each order under the true delay law
(a ratio of two probabilities, not the probability of the order). (c) ``\Delta\ell`` against the core sd
``\sigma_c``, tails at their true values: summed over the 40 stems (black) and for the example stem (orange, right
axis). Dashed: the true ``\sigma_c``.*

Figure 2 shows the identity at work. On the example stem, the true order carries 90 % of the posterior samples and
seven others share the rest (a). The true delay law finds the true order about six times more probable than the
propensity model does (log-weight +1.8), and most alternatives less probable (b): ``\Delta\ell_s > 0``.

Order 4 is an instructive contrast. Its log-weight is as high as the true order's, yet ``f`` finds it 16 times *less*
probable (``\log P_f`` = −18.0 against −15.2): ``A`` simply finds it even less probable. Where the true order displaces
two neighbours by one (−1, +1), order 4 displaces them by two (+2, −2), a pattern the product form penalises a little
more than a tailed law does ([§5.3](timing.md)); the weight corrects for this. With 0.5 % of the samples, order 4 adds
only 0.03 to the sum, against 5.3 for the true order. Summed over the stems, ``\Delta\ell`` peaks close to the true core
sd (c).

**When can the average be trusted?** When the weights ``r_j = P_f(\sigma_j)/P_A(\sigma_j)`` are of similar size. The
*effective sample size* over the ``N`` draws, ``(\sum r)^2 / \sum r^2``, measures this: ``N`` when all weights are
equal, 1 when a single draw carries everything. The weights are even when ``P_A`` is close to ``P_f``, as
[§5.3](timing.md) leads us to expect; in the validation the effective sample size is typically 175–200 out of 200
([§6.6](validation.md)).

**Limits.** The angle nuisances (``\kappa``, spiral, twist) are those of the propensity model's fit. Orders outside its
band (``|d| > k``) are never sampled, so a law that puts mass on them is slightly penalised. A stem whose effective
sample size is small is scored imprecisely; more posterior samples help.

## 5.6 The delay law

**assumption** — the delay is a precise Gaussian core, plus two exponential tails for sites committed late or early
(the package's default law, `GaussianExpTails`):

```math
f = (1 - \pi_L - \pi_E)\, \mathcal N(0, \sigma_c^2)
+ \pi_L\, \big(\mathcal N(0, \sigma_c^2) \ast \mathrm{Exp}(m_L)\big)
+ \pi_E\, \big(\mathcal N(0, \sigma_c^2) \ast (-\mathrm{Exp}(m_E))\big).
```

The core sd ``\sigma_c`` is the precision of a typical commitment. A fraction ``\pi_L`` of the sites is late by an extra
exponential delay of mean ``m_L``, and a fraction ``\pi_E`` early by ``m_E``. Why tails? A Gaussian core of sd 0.42
makes neighbour swaps common but displacements by two very rare ([§5.2](timing.md)). A tail gives exactly what a
Gaussian cannot: one organ displaced by two or three positions while its neighbours stay in order. For three sites,
the orders 3 1 2 and 2 3 1 have a probability of 0.03 % under the Gaussian core alone, and 2.4 % under the law of the
running example, whose tails hold 12 % of the sites.

Its variants are named by the tails they fit: `:core` (``\sigma_c``), `:late` (``\sigma_c, \pi_L, m_L``), `:early`
(``\sigma_c, \pi_E, m_E``) and `:both` (all five); `GaussianExpTails(:late)` is the starting law of the `:late` fit.
The nested sequence `:core`, `:late`, `:both` is the usual one: comparing their ``\Delta\ell`` tells whether the
orders call for tails, and on which side. This law is a default, not a requirement: everything below works for any
delay law, and [Custom delay laws](custom_laws.md) shows how to define one and why this one is the default.

!!! details "The density"
    A Gaussian of sd ``\sigma_c`` convolved with an exponential of rate ``\lambda = 1/m`` (an exponentially modified
    Gaussian) has the density
    ``\lambda \exp\!\big(\lambda(\lambda \sigma_c^2/2 - x)\big)\, \Phi\big((x - \lambda \sigma_c^2)/\sigma_c\big)``, with ``x = z`` for the
    late tail and ``x = -z`` for the early one. The mixture is evaluated in log space on `DELAY_GRID` and
    renormalised there.

**What the orders can identify.** The orders do not see the delay of a site, only whom it overtakes. What they
identify is not the five parameters of ``f`` but the **overtaking profile**:

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
- **The parameters of ``f`` are readings of its shape.** The core sd ``\sigma_c``, the tails ``\pi`` and ``m``, and the
  **off-time fraction** ``P(|e| > 1)`` (`tail_masses`) are not seen directly: they follow from the profile through the
  assumed shape, a Gaussian core with exponential tails. When the shape is clear-cut, they are well determined (for
  light tails, the core and the off-time fraction are recovered). When it is not, they can drift while the orders stay
  the same: without tails, a short "tail" holding most of the sites just reshapes the core; with heavy tails, a
  narrower core plus more tail mass predicts the same orders, and the core collapses. Either way, the profile does not
  move.

So the profile is the result to report and to compare between groups (`overtaking_profile` reads it from the
posterior orders, with no delay law; `delay_law_inference` gives that of the fitted law). The fitted law interprets
it, within its assumed shape: how precise a typical commitment is, and how many sites are off time.
[§6.4](validation.md) measures each of these statements on simulated stems.

## 5.7 Fit, uncertainty and comparisons

- **Fit.** `fit_delay_law(samples, f)` maximises ``\Delta\ell`` by Newton's method with a trust region, starting
  from the law ``f``, on the unconstrained free parameters that ``f`` declares ([Custom delay laws](custom_laws.md)).
  For the default law, ``f`` is a variant such as `GaussianExpTails(:both)`, or `GaussianExpTails(:both, g)` to
  start from the values of a law ``g``. Its free parameters: ``\log \sigma_c``; for `:both`,
  ``\operatorname{logit}(\pi_L + \pi_E)``, the logit of the late
  share ``\pi_L/(\pi_L + \pi_E)``, ``\log m_L`` and ``\log m_E``. The tail mass and the late share are separate
  coordinates because the first is better determined than the second. The grid is fixed, so ``\Delta\ell``
  is a smooth function of the parameters. Its gradient and Hessian are exact: forward-mode dual numbers pass through
  the delay density and the recursion of [§5.4](timing.md).
- **Robust standard errors.** The stem is the unit of replication: each stem contributes its own term
  ``\Delta\ell_s``, and stems may differ from each other more than one shared law says. `delay_law_inference`
  therefore measures the uncertainty from the spread of the per-stem slopes ``g_s`` of these terms, not from the
  curvature ``H`` of their sum alone. This is the sandwich covariance
  ``H^{-1}\big(\sum_s g_s g_s^\top\big)H^{-1}``, with exact derivatives. The delta method then gives the standard
  errors of ``\sigma_c``, of the tail masses and of the profile ``q_1, \dots, q_4`` (`overtaking`).
- **The profile without a delay law.** `overtaking_profile(samples)` averages, over each stem's posterior samples,
  the fraction of pairs ``d`` apart that are inverted, and pools the stems. Its standard errors are clustered by
  stem. It needs no delay law, so it is also a check of the fitted one: the two profiles should agree.
- **Two groups** (a mutant and its co-segregant wild type). Compare their **profiles**: `compare_profiles(a, b)`
  gives the differences ``q_d`` by lag with standard errors, and a Wald test over ``q_1, q_2, q_3``. The profiles can
  come from `overtaking_profile` (no delay law) or from each group's fitted law. Separate laws can also be compared
  with one shared law by the difference of their summed ``\Delta\ell`` (a likelihood ratio). The terms of ``A``
  cancel in these differences, because each stem's term is relative to its own group's fit of ``A``.
- **A shared core.** A group with heavy tails has no identified core sd of its own ([§5.6](timing.md)). To still read
  its off-time fraction, fit the groups together with **one core sd and their own tails**:
  `fit_delay_law([smp_wt, smp_mutant], GaussianExpTails(:both))`. This states a biological hypothesis, *the mutant
  changes the off-time organs, not the precision of a typical commitment*, with the light-tailed group carrying the
  core.
  `shared_core_test` tests it by the likelihood ratio of one core against one per group (χ² with one degree of freedom
  per extra group; with heavy tails, permuting the group labels among the stems calibrates it). `delay_law_inference`
  on the joint fit gives each group's off-time fraction and their difference, with standard errors that account for
  the shared core. Check that the shared core stays close to the light-tailed group's own; if it collapses, report the
  separate fits instead ([§6.5](validation.md)).
- **Delays in hours.** The delays are in plastochrons. The `Plastochron` module estimates the plastochron at each
  flower rank from daily counts of open flowers (box below), and converts a delay to hours with the mean plastochron
  of the measured organs (`mean_plastochron`, `in_hours`; [worked analysis](workflow.md), step 5).

!!! details "The flowering model"
    **assumption** — flowers open at the pace at which they were initiated, so that the rate of opening is the inverse
    of the plastochron. Plant ``i`` opens flowers at a piecewise-linear rate (per day), with ``t`` in days:
    ```math
    \lambda_i(t) = \begin{cases} b + a t & t \le \tau_i \\ \rho_i - d\,(t - \tau_i) & \tau_i < t \le A_i \\ 0 & t > A_i \end{cases}
    \qquad \rho_i = b + a\tau_i,\quad A_i = \tau_i + \rho_i / d ,
    ```
    a rise, then a decline from a change point ``\tau_i`` until the meristem arrests at ``A_i``. The change points vary
    between plants, ``\log\tau_i \sim \mathcal N(\mu_\tau, \sigma_\tau^2)``, and are integrated out. The new flowers of
    a day are Poisson with mean ``\int\lambda_i`` over that day (over the whole gap after days without record), and
    the five parameters are fitted by maximum likelihood (`fit_flowering`).

    Flower ``k`` opens when the cumulative ``F_i(t) = \int_0^t \lambda_i`` reaches ``k - \tfrac12``, and its plastochron
    is ``T_P(k) = 1/\lambda_i(t_k)``. Since ``\lambda_i`` is linear on each phase, ``\lambda_i^2`` is linear in ``F_i``
    (``b^2 + 2aF`` while rising), which gives ``T_P(k)`` in closed form (`plastochron_by_rank`). A delay ``x`` in
    plastochrons is ``x\,\bar T`` in hours, with ``\bar T`` the mean of ``T_P`` over the ranks of the measured organs; the
    standard error of ``\bar T`` is the delta method through the covariance of the fit, and that of ``x\,\bar T``
    combines both, as independent: ``\sqrt{(\bar T\,\mathrm{se}_x)^2 + (x\,\mathrm{se}_{\bar T})^2}``.

**Code.** `GaussianExpTails(σc; πL, mL, πE, mE)`, `GaussianExpTails(variant)`, any `DelayLaw` (`delay_logpdf`, `free_parameters`,
`with_free_parameters`, `reported_parameters`, `nshared_parameters`), `rand([rng,] f)`, `delay_logdensity(f)`, `tail_masses(f; a = 1)`;
`loglik_order_exact(σ, f)` (or `(σ, logg, grid, γ)`); `order_samples(order_posterior(m, fit; nsamples), η)`,
`stem_terms(samples, f)`, `relative_loglik(samples, f)` (`stem_terms(samples, logP)` scores any law of the order given
as ``\sigma \mapsto \log P(\sigma)``); `fit_delay_law`, `delay_law_inference`, `shared_core_test`; the profile:
`overtaking(f)`, `overtaking_profile(samples)`, `compare_profiles(a, b)`;
`simulate_stems(f::DelayLaw; ...)`;
`flowering_rate`, `fit_flowering`, `plastochron_by_rank`, `plant_plastochron`, `relative_plastochron`,
`mean_plastochron`, `in_hours`.
