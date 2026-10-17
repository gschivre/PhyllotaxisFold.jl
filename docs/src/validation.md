# 6. Validation on simulated stems

The whole pipeline is run on stems simulated with a known truth, and each step is compared with it: the angle model
and the twist ([§2](angles.md)), the decoding of the orders ([§3](orders.md)), and the fitted delay law through the
re-weighting ([§5](timing.md)). The figures are made by `docs/make_validation_figures.jl` (simulated data only).

**Design [B].** It is close to the *Arabidopsis* data:

- 60 stems of 45 organs per data set;
- internodes from about 9 mm at the base to 3 mm at the top, with log-normal noise, so the twist coordinate ``x_p`` is
  in mm;
- an angular precision ``\kappa_s`` log-normal around 6.4 (``\log\mu_\kappa = 1.85``, ``\sigma_\kappa = 0.15``);
- 95 % golden spirals;
- a stem twist of ``1.0 \pm 0.7`` °/mm.

The orders come from Level-0 fold commitment ([§5.2](timing.md)) with three delay laws (plastochrons):

| scenario | core sd ``s`` | late tail | early tail | ``P(\lvert e\rvert > 1)`` |
|---|---|---|---|---|
| no tails | 0.42 | — | — | 1.7 % |
| WT-like | 0.42 | 5 % (mean 1.5) | 5.4 % (mean 1.0) | 6.4 % |
| mutant-like | 0.39 | 20 % (mean 1.1) | 27 % (mean 1.0) | 19.7 % |

The WT-like and mutant-like laws are close to the laws fitted to a wild type and to a mutant with more off-time organs.
Three replicates per scenario. Each replicate runs the steps of the [quick start](index.md):

1. `fit_map_torsion!` (propensity model, ``k = 4``, twist integrated out);
2. `decode_orders` (joint MAP);
3. `order_posterior` with 200 exact samples per stem;
4. the `:core`, `:late` and `:both` delay laws by `fit_delay_law`;
5. `delay_law_inference`.

## 6.1 The angle model and the twist

![Fitting](figures/v1_fitting.png)

*Figure 1. The angle model, all 9 replicates (colour: delay law of the replicate). (a) The twist rate of each stem,
posterior mean against truth. (b) The twist population: mean and sd, truth in black. (c) The population law of the
angular precision. (d) The fraction of stems whose orientation and divergence (golden or Lucas) are decoded
correctly.*

The twist of each stem is recovered (correlation 0.99), as are its population (mean 0.85–1.04 °/mm for a truth of 1.0;
sd 0.62–0.80 for 0.7) and the precision of the angles (``\log\mu_\kappa`` 1.81–1.90 for 1.85). Every spiral is
decoded with its correct orientation and divergence. One stem in 540 gets a twist at a secondary mode of its likelihood
(figure 1a, the point far from the diagonal): a reminder that the twist likelihood of a stem can be multimodal, which
is why it is integrated rather than fixed ([§7.2](inference.md)).

## 6.2 The orders

![Decoding](figures/v2_decoding.png)

*Figure 2. The orders. (a) Calibration of the posterior: every pair (height ``p``, displacement ``d``) is grouped by
its posterior probability ``\rho_{p,d}``, and the frequency with which ``d`` is the true displacement is plotted
against that probability (± 2 se). (b) The most probable displacement against the true one: percentage of the organs of each true displacement.
(c) The fraction of displaced organs per stem, decoded against true.*

- **The posterior is calibrated** (figure 2a). The displacements to which it gives a probability of 0.7 are the true
  ones about 70 % of the time. This is what makes the posterior samples a sound input for [§5](timing.md).
- **The most probable order is right for 98 % of the positions.** Displacements of ±1 and ±2 are recovered in 87–96 %
  of cases, and the rare displacements of ±3 and ±4 in 64–70 %. When the single best order is wrong, the posterior
  usually shares its probability with the true order, which is why counting is done on the posterior and not on the
  best order.

## 6.3 The delay law

![Timing recovery](figures/v3_timing_recovery.png)

*Figure 3. The delay law fitted through the posterior orders ([§5.4](timing.md)). (a–d) The core sd and the three tail
masses: estimate ± 2 robust se for each replicate, truth in black. (e) The fitted densities (colour) and the true ones
(dashed). (f) Log-likelihood gains: what the two tails add to a core-only law (circles), and how much the fitted law
beats the true one (triangles, the optimism of the fit).*

| scenario | ``s``: truth → fits | ``P(e > 1)`` | ``P(e < -1)`` | ``P(\lvert e\rvert > 1)`` | tails gain (``\Delta\ell``) |
|---|---|---|---|---|---|
| no tails | 0.42 → 0.28, 0.43, 0.42 | 0.009 → 0.022, 0.010, 0.009 | 0.009 → 0.003, 0.010, 0.009 | 0.017 → 0.025, 0.019, 0.018 | 1.6, 0.0, 0.0 |
| WT-like | 0.42 → 0.41, 0.41, 0.38 | 0.035 → 0.043, 0.045, 0.043 | 0.030 → 0.018, 0.021, 0.026 | 0.065 → 0.061, 0.065, 0.068 | 132, 126, 99 |
| mutant-like | 0.39 → 0.00, 0.01, 0.01 | 0.086 → 0.105, 0.108, 0.113 | 0.110 → 0.110, 0.110, 0.104 | 0.197 → 0.216, 0.218, 0.217 | 74, 58, 76 |

What the validation shows, from the most to the least robust:

1. **No false tails.** When the truth has none, the tails add 0 to 1.6 units of log-likelihood, and the off-time
   fraction stays at the core's own (0.017).
2. **Real tails are found without doubt.** For both tailed laws the tails add 58 to 132 units.
3. **The off-time fraction ``P(|e| > 1)`` is the best estimated quantity.** It is recovered within its standard error
   for the WT-like law. For the mutant-like law it is about 0.02 too high in all three replicates (about 2 se): a
   10 % overestimate when a fifth of the organs are off time.
4. **The core sd is recovered when the tails are light** (WT-like: 0.38–0.41 for 0.42), **not when they are heavy.**
   For the mutant-like law (47 % of the sites in a tail) the fitted core shrinks to zero, with a standard error that
   is misleadingly small. A narrow core plus more tail mass predicts nearly the same orders ([§5.5](timing.md)).
5. **The split between late and early leans late.** For the WT-like law ``P(e > 1)`` is overestimated (0.043–0.045
   for 0.035) and ``P(e < -1)`` underestimated (0.018–0.026 for 0.030). Some patterns have two readings. The order
   2 3 1 is either one site committed two plastochrons late, or two sites committed one plastochron early, and the
   fit tends to prefer the single late site.
6. **Optimism.** The fitted law beats the true one by 0 to 5 units. Differences of ``\Delta\ell`` below about 5
   between two laws are not meaningful on their own.

In short, the off-time fraction is the quantity to compare between groups. The core sd and the late/early split are
informative for laws with light tails, and should be read with the caveats of points 4 and 5.

## 6.4 The exact order likelihood and the weights

![Exact likelihood and weights](figures/v4_exact_likelihood_and_weights.png)

*Figure 4. (a) The exact Level-0 probability of each order of 7 sites ([§5.3](timing.md)), against its frequency
among a million orders drawn from the three delay laws. (b) The effective sample size of the re-weighting at the
fitted law, per stem, out of 200 posterior samples.*

The recursion of [§5.3](timing.md) matches the frequencies of simulated orders over five orders of magnitude. The
effective sample size of the weights is close to 200 for most stems (median 175–200), and above 100 for nearly all.
The posterior of the propensity model covers the orders that the delay laws favour, as anticipated in
[§5.4](timing.md).
