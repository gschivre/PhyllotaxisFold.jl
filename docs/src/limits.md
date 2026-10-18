# 7. Assumptions and limits

Every **assumption** of the manual is listed below, with the page that introduces it and the signs that it fails.
Most can be checked on the data with the package itself: a posterior mass at the edge of the band, a small effective
sample size, per-stem terms that disagree with each other.

| assumption | where | what would break it, and how to check |
|---|---|---|
| organs sit on a regular spiral; noise acts on timing, not on positions | [§2.1](angles.md) | missed organs are not modelled; without `outlier = true`, a grossly wrong angle appears as a long displacement |
| von Mises angular noise, one precision per stem | [§2.2](angles.md) | a broad but not uniform error (e.g. angles off by a fraction of the divergence) |
| one twist rate per stem, Normal across stems | [§2.3](angles.md) | a tapering stem makes the rate drift along the stem |
| displacements bounded by ``k = 4`` | [§3.2](orders.md) | posterior mass at ``\lvert d\rvert = k``; refit with a larger ``k`` |
| the deviations of the sites from a regular clock are i.i.d. (independent readiness jitter is absorbed into ``f``) | [§5.2](timing.md) | errors that add up (a clock with variable intervals, read as tails beyond about 30 % interval variation, [§6.7](validation.md)), cascades of permutations; stems that differ in their off-time rate |
| the same delay law all along the stem | [§5.2](timing.md) | an off-time rate that changes along the stem (e.g. at the base) |
| the delay law is a Gaussian core with exponential tails | [§5.6](timing.md) | only the readings of the law depend on it (core sd, off-time fraction), not the overtaking profile; check that the fitted law's profile matches `overtaking_profile`; discrete slot errors (organs exactly one plastochron off) are fitted as short tails |
| (shared-core fit) the groups share one core sd | [§5.7](timing.md) | cores that truly differ: `shared_core_test` rejects the shared core |
| the timing law is scored at the propensity model's angle nuisances and band | [§5.5](timing.md) | a small effective sample size of the weights; refit with a larger ``k`` |
| flowers open at the pace they were initiated; piecewise-linear opening rate | `Plastochron` | a maturation delay that changes during flowering; a curved rise or decline |

Some questions are beyond the reach of the angles alone, whatever the model:

- they confound angular noise with permutations (a lower ``\kappa`` or more permutations);
- they identify the delay law only through its overtaking profile, which carries no late/early direction
  ([§5.6](timing.md));
- they cannot separate a noisier readiness from a noisier commitment, since only their sum enters the order
  ([§5.2](timing.md)).

Timing data (internodes, imaging of primordia) are needed for those questions.
