# 1. The problem

## 1.1 Organs on a spiral

At the tip of a growing shoot, the meristem produces new organs one after the other. Each new organ (a *primordium*)
appears where the auxin field allows it: away from the youngest organs, which inhibit their surroundings (Reinhardt
et al. 2003; Shi & Vernoux 2019). This rule is enough to produce a regular pattern. In *Arabidopsis* the flowers
follow a spiral, and two successive flowers are separated by the **divergence angle** ``\psi \approx 137.5^\circ``,
the golden angle. Some plants follow a rarer spiral of ``\approx 99.5^\circ`` (the Lucas angle). The spiral turns
clockwise or counter-clockwise, depending on the plant.

The stem keeps a record of this process. Once the organs are initiated, they are carried upwards by the growth of the
stem, and they stay in place. Going up a mature stem is like reading the history of the meristem, one organ per
*plastochron* (the time between two initiations).

## 1.2 A swap and its signature

The record is often disturbed: two organs initiated close in time can end up in the inverted order along the stem
(Besnard et al. 2014; Refahi et al. 2016). Their *positions* on the spiral are correct, but the organ of the later site
sits below the organ of the earlier one. What does this look like in the angles?

Number the sites of the spiral ``j = 1, 2, \dots`` in the order of the spiral (the *canonical* order). Site ``j`` sits
at the angle ``(j - 1)\psi``. In a perfect stem the organ at height ``p`` comes from site ``p``, and every divergence
is ``\psi``. Now swap the organs of sites 6 and 7. The organs at heights 5 to 8 come from sites 5, 7, 6, 8, and the
three divergences around the swap become

```math
(7 - 5)\psi = 2\psi,\qquad (6 - 7)\psi = -\psi,\qquad (8 - 6)\psi = 2\psi
```

instead of ``\psi, \psi, \psi``. A swap leaves a signature that spans three divergences (figure 1c). Every
permutation of the organs has such a signature. If organ ``p`` comes from site ``\sigma(p)``, the divergence between
heights ``p`` and ``p + 1`` is ``(\sigma(p + 1) - \sigma(p))\,\psi``.

![A swap in the timeline, the internodes and the angles](figures/m9_swap_signature.png)

*Figure 1. How a timing event becomes a permutation. (a) Twelve sites: site ``j`` becomes ready at time ``j`` (circles,
in plastochrons) and commits after a random delay (stars). Sites 6 and 7 (orange) commit in inverted order. (b) The
internodes around the inverted pair, if the stem grows at a steady pace: about two plastochrons, nearly zero, just
under one. (c) The divergence angles: the inversion replaces three divergences ``\psi`` by ``2\psi``, ``-\psi``,
``2\psi``.*

## 1.3 Why the order cannot simply be read off

If the angles were exact, the order could be read directly from the divergences. They are not:

- **Noise.** The measured azimuth of an organ is off by several degrees, more on some stems than on others.
- **Unknown spiral.** The angle of the first site, the orientation of the spiral and the divergence (golden or
  Lucas) differ from stem to stem, and are not measured.
- **Twist.** Many stems twist as they elongate. The twist slowly rotates the organs higher up, and it accumulates
  along the stem.

So a divergence of ``170^\circ`` may be a noisy ``\psi``, or part of a swap seen through noise. The angles of one organ
cannot decide this alone. Only the *whole* stem can: an order is plausible if it explains all the angles at once, with
one spiral, one level of noise and one twist. Look at the example stem on the [home page](index.md): which organs are
swapped? Some swaps are obvious, others are not.

A careful answer therefore weighs every possible order of the stem by how well it explains the angles. There are many:
a stem of 45 organs can be ordered in ``45! \approx 10^{56}`` ways, more than the number of atoms in the Earth.

## 1.4 The question behind the order

We do not want the order for its own sake. We want what it says about the meristem. Here is the simplest mechanism
that produces swaps:

- **assumption** — the sites become *ready* at a regular pace: site ``i`` at time ``i``, in plastochrons;
- **assumption** — each site then *commits* to become an organ after a random delay ``e_i``, independently of the
  others, with the same law ``f`` for every site;
- **assumption** — the organs appear on the stem in the order of commitment.

The commitment time of site ``i`` is ``\tau_i = i + e_i``. Two sites are swapped when the later one commits first. The
order on the stem is the ranking of the commitment times (figure 1a):

```math
\sigma = \operatorname{sortperm}(\tau_1, \dots, \tau_n),\qquad \tau_i = i + e_i,\qquad e_i \overset{\text{iid}}{\sim} f .
```

Everything about the mechanism is in the **delay law** ``f``. A narrow ``f`` (sd well below one plastochron) gives rare
swaps between neighbours. A wide ``f`` gives frequent and longer-range permutations. An ``f`` with a tail, a few sites
committed much later than the others, gives a distinctive pattern: one organ displaced by two or three positions. The
goal of the package is to **estimate ``f`` from the angles**.

## 1.5 The plan

Write the likelihood of a delay law. The angles depend on ``f`` only through the order:

```math
p(y \mid f) = \sum_{\sigma} P_f(\sigma)\, p(y \mid \sigma),
```

where ``P_f(\sigma)`` is the probability of the order under the delay law, and ``p(y \mid \sigma)`` the probability of
the angles given the order (with the unknown spiral, noise and twist integrated out). Three questions follow, and each
gets a page:

1. What is ``p(y \mid \sigma)``? A model of the angles given the order ([§2](angles.md)).
2. How can a sum over ``10^{56}`` orders be computed? The sum turns out to be easy, *provided* ``P(\sigma)`` has a
   particular form: a product of one term per organ, each depending only on the organ's position and its site
   ([§3](orders.md)).
3. Does ``P_f`` have that form? Not exactly. Whether site ``i`` comes before site ``j`` depends on both of their
   delays, so the event "``\sigma(p) = j``" is tied to all the other positions. A product of one term per organ can
   approach this, but not reproduce it ([§5.3](timing.md)).

So we cannot compute ``p(y \mid f)`` by the easy sum. But there is a way around. Suppose we fit an order law ``A``
that *does* have the product form, and is flexible enough to follow the data. Then we can compute exactly which orders
are plausible for each stem, and draw samples of them. If ``f`` is a good description of the plant, it should give a
high probability to those orders.

This intuition is exact. Since ``A`` and ``f`` describe the angles in the same way, and differ only in the law of the
order, the likelihood ratio of ``f`` against ``A`` is an average over the orders reconstructed under ``A``
([§5.5](timing.md)):

```math
\frac{p(y \mid f)}{p(y \mid A)} = \mathbb{E}_{\sigma \sim p(\sigma \mid y, A)}\left[\frac{P_f(\sigma)}{P_A(\sigma)}\right].
```

This is the two-step plan of the package:

| step | model | what it gives | page |
|---|---|---|---|
| 1 | the **propensity model** ``A``: a free cost for each displacement, in product form | the exact posterior of the order of each stem, with the spiral, the noise and the twist | [§2–4](propensity.md) |
| 2 | **commitment timing**: the delay law ``f`` | ``\log p(y \mid f) - \log p(y \mid A)``, so a fit of ``f`` and tests between laws | [§5](timing.md) |

**Data.** For each stem, from the bottom to the top (positions ``p = 1, \dots, n``):

| measurement | symbol | used for |
|---|---|---|
| absolute azimuth of each organ (radians) | ``y_p`` | the likelihood |
| internode lengths (mm), optional | ``\ell_p`` | the coordinate of the stem twist ([§2](angles.md)) |
| daily counts of open flowers, optional | ``c(\text{day})`` | the plastochron in hours (`Plastochron`) |

The models use absolute azimuths rather than divergence angles ``y_{p+1} - y_p``. A single misplaced organ then
changes one observation instead of two, and the observations given the order are independent.
