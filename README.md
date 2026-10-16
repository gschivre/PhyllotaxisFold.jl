# PhyllotaxisFold.jl

Statistical models of the **order of organs along a plant stem**, read from the absolute azimuths of the organs
(flowers, fruits). In *Arabidopsis* the organs follow a spiral with a divergence close to the golden angle, but two
organs initiated close in time often end up in the inverted order. These permutations are a readout of the timing
noise of organ initiation. The package estimates that timing noise, and interprets it through a cellular switch that
commits a group of cells to become an organ.

The method is described in [`docs/method.md`](docs/method.md), from the biology to the equations and the code.

## Models

| model | constructor | how the order of organs is described | timing parameters |
|---|---|---|---|
| **A** | `PhyllotaxisModelAD` | free log-propensity η_d of each displacement d | η_1 … η_k |
| **B** | `FoldModel` | η derived from a commitment switch near a fold (saddle-node) bifurcation, with sites made ready at a regular pace | shape of the delay law, precision γ = plastochron / sd(delay) |
| **B with tempo** | `FoldModel(...; tempo = r)` | as B, with a precision that follows the plastochron along the stem, γ_p = γ r_p^β | shape, γ, β |

All models share the same angle likelihood: an exact sum over every order of the organs within a band (a banded
permanent computed by dynamic programming), over the unknown spiral offset, orientation and divergence (golden or
Lucas), over a per-stem angular precision, and over a per-stem twist of the stem (integrated out). Estimation is by
maximum a posteriori with Newton's method (analytic gradient, Hessian from one dual-number pass) and a Laplace
approximation.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/gschivre/PhyllotaxisFold.jl")
```

## Quick start

```julia
using PhyllotaxisFold

# precomputed passage-time and propensity tables (plateau family; tables_ramp_k4.jls for the ramp family)
(; tab, ptab) = load_tables(joinpath(pkgdir(PhyllotaxisFold), "tables", "tables_k4.jls"))

# angles: one vector of absolute azimuths (radians, bottom to top) per stem; internodes in mm (for the twist)
sim = simulate_stems(tab, 2.0, 1.7; plants = 50, organs = 45)      # 50 simulated stems
angles = sim.angles
internodes = [fill(6.0, 44) for _ in angles]

# model B, stem twist integrated out
m = FoldModel(angles, ptab; torsion_x = [torsion_coordinate(ℓ) for ℓ in internodes])
r = fit_map_torsion(m)
r.fit.θ            # (μ = shape, logγ, logμκ, σκ, α, β)
r.m_ω, r.s_ω       # twist population (degrees per mm)

# model A on the same stems
prob = TransformedLogDensityProblem(PhyllotaxisModelAD(angles, PhyllotaxisPrior(); torsion_x = m.torsion.x))
rA = fit_map_torsion(prob)

# model B with a tempo profile from daily open-flower counts (cumulative counts per plant, NaN = day not recorded)
# fl = fit_flowering(counts)                  # piecewise-linear rate, per-plant change point, finite arrest
# TP = plastochron_by_rank(fl, 80)             # typical plant, by flower rank (days); or, one curve per stem:
# TP = [plant_plastochron(fl, counts_of_plant(s), 80) for s in stems]
# known ranks: the organ at stem position p is flower p + offset
# mt = FoldModel(angles, ptab; torsion_x = m.torsion.x, plastochron = TP, offsets = 1)
# uncertain ranks (0 to 2 basal organs missed): the likelihood of each stem is summed over the offsets
# mt = FoldModel(angles, ptab; torsion_x = m.torsion.x, plastochron = TP, offsets = 0:2)
# fit_map_torsion(mt).fit.θ.β
```

Tables can be rebuilt with `build_tables` and `build_ramp_tables` (about a minute on 18 threads).

## Tests and figures

```julia
] test PhyllotaxisFold
```

The figures of the method description are made from simulations only: `julia --project=docs docs/make_figures.jl`.

## Licence

MIT, see [`LICENSE`](LICENSE).
