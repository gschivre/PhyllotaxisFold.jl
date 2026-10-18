# 2. Angles given an order

Suppose for a moment that we know the order ``\sigma`` of a stem: the organ at height ``p`` comes from site
``\sigma(p)``. How should its angle ``y_p`` be distributed? This page builds the answer one term at a time, from the
three difficulties of [§1.3](problem.md).

## 2.1 The spiral

**assumption** — the sites sit on a regular spiral. Site ``j`` is at the angle

```math
\phi + o\,(j - 1)\,\psi ,
```

where ``\phi`` is the angle of the first site, ``o = \pm 1`` the orientation of the spiral and ``\psi`` the
divergence. The divergence is the golden angle (``\approx 137.5^\circ``) with probability ``\alpha``, and the Lucas
angle (``\approx 99.5^\circ``) otherwise. ``\phi`` is uniform on the circle, and each orientation has probability ½.
These three quantities are specific to each stem, unknown, and integrated out ([§2.6](angles.md)).

A permutation moves *organs* between sites. It does not move the sites. The organ at height ``p`` is therefore expected
at ``\phi + o\,(\sigma(p) - 1)\,\psi``. This separation, *where the sites are* versus *in which order their organs
appear*, is the core of every model of the package.

## 2.2 The noise

**assumption** — the angle of an organ is its site's angle plus an independent error with a von Mises law, the circular
analogue of the Gaussian:

```math
\mathrm{vM}(y \mid m, \kappa) = \frac{e^{\kappa \cos(y - m)}}{2\pi I_0(\kappa)} .
```

The concentration ``\kappa`` plays the role of an inverse variance: for large ``\kappa`` the error has an sd of about
``1/\sqrt{\kappa}`` radians. ``I_0`` is the modified Bessel function of order 0.

**assumption** — each stem has its own concentration, ``\log\kappa_s \sim \mathcal N(\log\mu_\kappa, \sigma_\kappa^2)``,
integrated out of its likelihood ([§2.6](angles.md)).

Why one ``\kappa`` per stem? A poorly measured stem must be allowed to be noisy. With a single ``\kappa`` for all stems,
the model could only explain its scattered angles by many permutations, and these spurious permutations would bias
everything downstream.

## 2.3 The twist

Many stems twist as they elongate (Landrein et al. 2013). If the surface of a stem of radius ``r_s`` is inclined by a
small angle ``\alpha_s``, an internode of length ``\ell`` rotates everything above it by ``\ell \tan\alpha_s / r_s``.
The rotations add up along the stem. The organ at height ``p`` is then rotated by

```math
\omega_s\, x_p, \qquad x_p = \ell_1 + \dots + \ell_{p-1} ,
```

where ``x_p`` is the length of stem below the organ and ``\omega_s`` a rate (degrees per mm) specific to each stem.
When the internodes are not measured, ``x_p = p - 1`` and the rate is per organ.

**assumption** — the rates are a random effect, ``\omega_s \sim \mathcal N(m_\omega, s_\omega^2)``, integrated out of
each stem's likelihood ([§2.6](angles.md)).

Could a twist be mistaken for a slightly different divergence? Both rotate the organs more and more going up the
stem. A divergence ``\psi + \delta`` instead of ``\psi`` moves the organ of site ``j`` by ``o\,(j - 1)\,\delta``:
*with the sign of the orientation*. A twist rotates the organ at height ``p`` by ``\omega_s x_p``, whatever the
orientation of the spiral. So the two are told apart by comparing clockwise and counter-clockwise stems. A shifted
divergence pushes them in opposite directions, a twist in the same direction. A twist also follows the length of the
stem (long internodes rotate more), and a divergence follows the organ count.

![The twist of the example stem](figures/e2_example_twist.png)

*Figure 2. The example stem. (a) The error of each angle with respect to the site of its true organ, against the
length of stem below it: the errors drift linearly, at the stem's twist rate. (b) The same errors once the twist is
removed: what remains is the von Mises noise.*

## 2.4 Outlier angles (option)

**assumption** (with `outlier = true`) — with a small probability ``\pi_o``, an angle is not related to its site at
all, and is uniform on the circle:

```math
y_p \sim (1 - \pi_o)\,\mathrm{vM}\big(\cdot \mid \phi + o\,(\sigma(p) - 1)\,\psi + \omega_s x_p,\ \kappa_s\big) + \frac{\pi_o}{2\pi} .
```

``\pi_o`` is fitted (prior Beta(1, 49), mean 2 %). A grossly mis-measured angle is then explained as an outlier
rather than as a long displacement of its organ.

## 2.5 The likelihood of one stem

Given the order, the angles are independent. The probability of the angles of a stem is the sum, over everything that
is not observed, of the probability of the angles:

```math
p(y) = \sum_{o,\,\psi} p(o)\,p(\psi) \int_0^{2\pi} \frac{d\phi}{2\pi} \int p(\kappa)\,d\kappa
\int \mathcal N(\omega \mid m_\omega, s_\omega^2)\,d\omega
\;\sum_{\sigma} P(\sigma) \prod_{p=1}^{n} \mathrm{vM}\big(y_p \mid \phi + o\,(\sigma(p) - 1)\,\psi + \omega\, x_p,\ \kappa\big).
```

Everything in this formula is fixed by this page except ``P(\sigma)``, the law of the order. The models of the
package differ *only* in ``P(\sigma)``: the propensity model ([§4](propensity.md)) and the timing laws
([§5](timing.md)). That is what makes the comparison of [§1.5](problem.md) possible.

Two parts of the formula remain hard. The sum over ``\sigma`` is the subject of [§3](orders.md). The integrals over
the unknowns of each stem are the subject of the next section.

## 2.6 Integrating out the unknowns of each stem

The spiral (``o``, ``\psi``, ``\phi``), the precision ``\kappa`` and the twist ``\omega`` of a stem are not known.
Why integrate over them, rather than fix each at its best value? Because fixing them would make the orders look more
certain than they are. A different offset or twist, combined with a few permutations, can explain the same angles;
a precision fixed too high leaves no room for noise, and turns it into permutations. Integrating keeps every
explanation, in proportion to its probability.

The orientation and the divergence take four combinations, summed exactly. The three other integrals are
one-dimensional, and each is computed by a rule suited to the shape of its integrand:

- **the offset** ``\phi``, uniform on the circle: the rectangle rule on `ngrid_ϕ` equally spaced nodes (32 by
  default), which is very efficient for periodic functions;
- **the precision** ``\log\kappa``, Gaussian across stems: a Gauss–Hermite rule on that law (`ngrid_κ`, 16 nodes);
- **the twist** ``\omega``: the likelihood of one stem is very sharp in ``\omega`` (about 0.1°/mm wide) and often
  multimodal, so a rule placed on the population law would miss its peaks. The nodes are placed on the peaks of each
  stem's own likelihood instead (5 per peak).

All the terms are computed in log space, and ``\log I_0`` with a dedicated routine (`logbesseli`) that also accepts
dual numbers for the derivatives. The cost is proportional to `ngrid_ϕ` × `ngrid_κ` × `nnodes` (the twist nodes per
peak, set by `fit_map_torsion!(m; nodes_kw = (; nnodes))`).
The boxes below derive each rule and measure its accuracy on the 40 stems of the running example.

!!! details "Integrating out the spiral: derivation and accuracy"
    **Orientation and divergence.** Four combinations, summed exactly with their probabilities (½ for each
    orientation, ``\alpha`` and ``1 - \alpha`` for the two divergences).

    **The offset ``\phi``.** It is uniform on the circle, and integrated by the rectangle rule on ``N`` equally spaced
    nodes (`ngrid_ϕ`, 32 by default). Why this rule, and how accurate is it? For a given order and precision, the
    angle terms of a stem add up to one cosine in ``\phi``:
    ``\sum_p \kappa \cos(y_p - \phi - c_p) = a \cos(\phi - \bar\phi)``, with ``c_p`` the rest of the expected
    angle and ``a = \kappa \lvert \sum_p e^{i(y_p - c_p)} \rvert \le \kappa n``. So the integrand is a von Mises peak
    of concentration ``a``, whose Fourier series is
    ```math
    e^{a \cos\theta} = I_0(a) + 2 \sum_{m \ge 1} I_m(a) \cos m\theta .
    ```
    The rectangle rule on ``N`` nodes integrates every Fourier mode exactly except the multiples of ``N``. Its relative
    error is therefore ``2 \sum_{j \ge 1} I_{jN}(a)/I_0(a) \approx 2\, I_N(a)/I_0(a) \approx 2\, e^{-N^2/(2a)}`` (for
    ``N`` well below ``a``): it falls very fast with ``N``, but only once the nodes are closer than the width of the
    peak, ``1/\sqrt a``. For a well-measured stem of 45 organs (``\kappa \approx 6.4``), ``a \approx 290``: the peak is
    about 3.4° wide, while 32 nodes are 11.25° apart. The sum over orders and precisions mixes such peaks, and the
    sharpest ones dominate the error.

    **Accuracy.** On the 40 stems of the running example, at their fitted parameters, each stem's log-likelihood was
    compared with a reference computed on 256 × 64 nodes (``\phi`` × ``\kappa``), for the fitted ``\sigma_\kappa``
    and three larger ones (0.3, 0.5, 0.8):

    | ``\phi`` nodes | median error per stem | worst stem |
    |---|---|---|
    | 16 | 1.1–1.4 | 3.6–4.9 |
    | 32 (default) | 0.17–0.19 | 0.30–0.43 |
    | 64 | 0.0002 | 0.003 |
    | 128 | ``10^{-9}`` | ``10^{-7}`` |

    (in log-likelihood units; the 64- and 128-node rows at the fitted ``\sigma_\kappa``). As predicted, each
    doubling squares the error factor ``e^{-N^2/(2a)}``. The error of the default depends on where each stem's peak
    falls between the nodes; see the last box of this section.

!!! details "Integrating out the precision: derivation and accuracy"
    **The rule.** ``\log\kappa_s`` is Gaussian, so its integral is a Gauss–Hermite rule on that law: with ``K`` nodes
    ``z_k`` and weights ``\gamma_k`` (`ngrid_κ`, 16 by default),
    ```math
    \int p(y \mid \kappa)\, p(\kappa)\, d\kappa \approx \sum_k \gamma_k\, p\big(y \mid e^{\log\mu_\kappa + \sigma_\kappa z_k}\big).
    ```
    The nodes move with ``(\log\mu_\kappa, \sigma_\kappa)``, so the approximation is a smooth function of them, as
    the fit needs.

    **When it is accurate.** The rule is exact when the likelihood, as a function of ``z``, is a polynomial of degree
    below ``2K``. In ``\log\kappa`` the likelihood of a stem is a peak whose width is about ``\sqrt{2/n}`` (the
    information about ``\log\kappa`` carried by ``n`` concentrated angles is about ``n/2``), that is about 0.2 for 45
    organs. In ``z`` its width is ``\sqrt{2/n}/\sigma_\kappa``. The rule is accurate while this width is not small
    compared with the spacing of the nodes: while ``\sigma_\kappa`` is at most a few times ``\sqrt{2/n}``. Unlike
    the twist, the precision of the stems varies little, and nodes placed on the population law are enough.

    **Accuracy.** Same stems and reference as for ``\phi`` (with 128 ``\phi`` nodes, so that only the error of the
    ``\kappa`` rule remains); median and worst stem, in log-likelihood units:

    | ``\sigma_\kappa`` | 8 nodes | 16 nodes (default) | 32 nodes |
    |---|---|---|---|
    | 0.21 (fitted) | 0.00003, 0.0003 | ``< 10^{-6}`` | ``< 10^{-9}`` |
    | 0.3 | 0.002, 0.013 | ``< 10^{-4}`` | ``< 10^{-7}`` |
    | 0.5 | 0.05, 0.12 | 0.004, 0.013 | 0.0002 |
    | 0.8 | 0.31, 0.75 | 0.08, 0.17 | 0.009, 0.02 |

    The default is accurate up to ``\sigma_\kappa \approx 0.5``. Beyond, use 32 nodes.

!!! details "Integrating out the twist: derivation and accuracy"
    **The integral.** Write ``L_s(\omega)`` for the likelihood of stem ``s`` at twist rate ``\omega``: the formula of
    [§2.5](angles.md) without its integral over ``\omega``. What is needed is
    ``\int L_s(\omega)\, p(\omega)\, d\omega`` with ``p = \mathcal N(m_\omega, s_\omega^2)``.

    **Nodes.** Scan the rate, find the local maxima of ``\log L_s`` within 8 log-units of the best (at most four),
    refine each by golden section, and measure its curvature. Each mode ``r`` becomes a Gaussian component (centre
    ``c_r``, sd ``h_r`` from the curvature, widened by 25 %), with a weight ``\pi_r`` proportional to its Laplace mass.
    Their mixture ``q_s`` is a proposal that covers every peak. Five Gauss–Hermite nodes ``z_j`` (weights
    ``\gamma_j``) are placed on each component: ``\omega_{rj} = c_r + h_r z_j``.

    **Weights.** Write the integral against the proposal and apply the Gauss–Hermite rule to each component:
    ```math
    \int L_s\, p = \int \frac{L_s\, p}{q_s}\, q_s = \sum_r \pi_r\, \mathbb E_{\mathcal N(c_r, h_r^2)}\Big[\frac{L_s\, p}{q_s}\Big]
    \approx \sum_{r,j} \frac{\pi_r \gamma_j}{q_s(\omega_{rj})}\, L_s(\omega_{rj})\, p(\omega_{rj}) .
    ```
    The node weights ``\pi_r \gamma_j / q_s(\omega_{rj})`` do not depend on the population law ``p``. So the same
    nodes integrate any ``(m_\omega, s_\omega)``, and the integrated likelihood is a smooth function of them, which the
    fit needs ([§4.5](propensity.md)). This is multiple importance sampling with the balance heuristic. It is exact
    when ``L_s\, p / q_s`` is a polynomial of low degree on each component, that is, when each peak of ``L_s`` is close
    to Gaussian and ``p`` varies slowly across it.

    **Accuracy.** On the 40 stems of the running example, at their fitted parameters, each stem's integral was
    compared with a brute-force sum over a fine grid of rates (step 0.005 °/mm), for the fitted population law and three
    others (sd doubled, sd halved, mean shifted by 0.5 °/mm). 38 stems have one mode, one has two and one has four.

    | rule | median error per stem | worst stem |
    |---|---|---|
    | adaptive, 5 nodes per mode (default) | 0.02 | 0.23–0.29 |
    | adaptive, 9 nodes per mode | | 0.08 |
    | adaptive, 15 nodes per mode | | 0.01 |
    | Gauss–Hermite on the population law, 32 nodes | 0.2–2.5 | 0.6–11 |
    | Gauss–Hermite on the population law, 128 nodes | 0.1–0.8 | 0.4–3.4 |

    Errors are in log-likelihood units (the 9- and 15-node rows on the six worst stems). The adaptive rule is equally
    accurate for the four population laws, as the derivation says. Its residual error comes from peaks that are not
    quite Gaussian, and more nodes per mode remove it. A rule placed on the population law misses the sharp peaks
    even with 128 nodes. For fine comparisons of integrated log-likelihoods, use 9 nodes per mode
    (`fit_map_torsion!(m; nodes_kw = (; nnodes = 9))`), at about twice the cost.

**How accurate is accurate enough?** The default settings are chosen for *estimation*. An integration error that is
nearly constant as the parameters move shifts the log-likelihood without moving its maximum or its curvature, so it
leaves the estimates and their standard errors alone. The errors here are of that kind: they depend mostly on the
data of each stem (where its peak in ``\phi`` falls between the nodes, how far its peak in ``\omega`` is from a
Gaussian), and little on the parameters.

They do not cancel when the integrated log-likelihoods of two *different* fitted models are compared, because the two
fits can sit at parameters where the error differs. Such comparisons call for finer settings: 64 ``\phi`` nodes, 32
``\kappa`` nodes if ``\sigma_\kappa`` exceeds 0.5, and 9 twist nodes per peak
(`PhyllotaxisModelAD(angles; ngrid_ϕ = 64)`, `fit_map_torsion!(m; nodes_kw = (; nnodes = 9))`). The timing laws of
[§5](timing.md) are compared on the orders, not through these integrals, and do not need them.

**Code.** `AngleLikelihood.stem_loglik!` (one stem, all grids, and its gradient), `stem_loglik`, `angle_loglikelihood`;
the twist: `torsion_coordinate(ℓ)`, `corrected_angles`; simulation: `simulate_stems` (both laws of the order).
