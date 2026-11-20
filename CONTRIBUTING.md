# Developing PhyllotaxisFold.jl

## Tests

```julia
] test PhyllotaxisFold
```

The tests check the numerical core against independent references (brute-force enumeration, Monte Carlo, finite
differences), the recovery of known parameters on simulated stems, and code quality with
[Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) and [ExplicitImports.jl](https://github.com/JuliaTesting/ExplicitImports.jl).
They are grouped by module, step 1 then step 2. The delay laws of the manual page "Custom delay laws"
(`StudentDelay`, `GumbelDelay`) are defined at the top of `test/runtests.jl` as on the page (the Student law without
its sampler, to test the fallback): keep the two in sync.

The test suite also guards the hot paths against regressions: `@inferred` on the per-stem likelihood and the
summed log posterior (Float64 and the Hessian duals, with and without outliers), and `@allocated` on the stem kernel
and the permanent.

## Performance

`benchmark/` is a separate environment with the tools (BenchmarkTools, JET, AllocCheck, Profile); it develops the
package from `..`. Start Julia with several threads (`julia --project=benchmark -t auto`):

- `benchmark/setup.jl`: the fixtures (40 simulated stems of 40 organs, default grids; a twisted set), included by the
  others.
- `benchmark/fits.jl`: timings and allocations of the log posterior, its gradient and Hessian, and of the two fits.
- `benchmark/checks.jl`: JET (`@report_opt`) over the log posterior and its derivatives, with and without twist,
  printing every runtime dispatch; AllocCheck (`check_allocs`) on the kernels of the permanent and of the weight
  tables. The only expected dispatch is the function barrier in front of each Hessian pass (the dual width is the
  number of parameters, known at run time).

For a time profile, `Profile.@profile` on a call of `AngleLikelihood._stem_ll!` then
`Profile.print(format = :flat, sortedby = :count, mincount = 30)`. `benchmark/README.md` records the reference timings.

## The manual

The pages are in `docs/src`, and the `docs` environment points to this package (`[sources]`). The build commands are
at the top of `docs/make.jl`; the manual is written to `docs/build/index.html`. The repository is private, so the
manual is built locally; `docs/make.jl` holds the commented `deploydocs` call for later. The equations are rendered by
MathJax, which needs an internet connection.

The references are in `docs/src/refs.bib`, cited in the pages by key (`[ferrell2012bistability](@cite)`, or
`(@citet)` for "Ferrell (2012)") with [DocumenterCitations.jl](https://github.com/JuliaDocs/DocumenterCitations.jl).
The list on the notation page is generated from the cited entries, and an unknown key fails the build.

CI (`.github/workflows/CI.yml`) runs the tests on Julia 1.10 and the latest release, and builds the manual headless
on every push to `main`. The built manual is kept as the run's `manual` artifact.

## Figures

The figures are written to `docs/src/figures` and committed, so building the manual does not remake them. They use
simulated data only. Each script says at its top how to run it.

- `docs/make_figures.jl`: the illustrations of the method, and the running example (`fig_examples()`).
- `docs/make_validation_figures.jl`: the validation on simulated stems (manual §6). `run_validation(; refit_timing =
  true)` refits only the delay laws, from the cached replicates.
- `docs/validation_site_walk.jl`: the wandering-sites simulation of manual §6.2 (table only, no figure).
- `docs/validation_clock_drift.jl`: the drifting-clock simulation of manual §6.8.
- `docs/validation_plackett_luce.jl`: the exact check of the re-weighting of manual §6.9.
- `docs/make_workflow.jl`: the figures of the worked analysis.
- `docs/make_simulation_figures.jl`: the figures of the page "Simulating stems" (simulation only, seconds).
- `docs/make_plate_diagram.jl`: the plate diagram of manual §2.5, from `docs/plate_diagram.tex` (TikZ, `bayesnet`).
  It installs its own TeX engine in a temporary environment; no LaTeX installation is needed.

The slow fits are cached (`docs/*.jls`) and reused on the next run; delete a cache to recompute it. The caches and
`docs/build` are not committed.

## Conventions

- Julia code: generic in the number type wherever a function can see ForwardDiff duals. Optimisation is done in
  unconstrained parameters, by Newton's method with a trust region (value, gradient and Hessian in one call): an
  `only_fgh!` callback where the gradient is analytic (the angle model), `TwiceDifferentiable` with
  `AutoForwardDiff()` otherwise (the delay law, the flowering rate).
- Formatting: [Runic.jl](https://github.com/fredrikekre/Runic.jl) (`runic --inplace src test docs/*.jl`).

## History: the tag `with-model-B`

Before version 0.2 (October 2026), the package held a second angle model. Its name, "model B" (the propensity model
was "model A"; "fold-commitment model" in the paper drafts), says nothing about what it did; this section describes
it. The tag `with-model-B` (commit b5c2127, still in the history of `main`) is its last version, and reproduces the
numbers of the supplementary negative result.

**What it did.** Like the propensity model, it summed the angles over all orders with the banded permanent. But its
displacement costs η were not free parameters: they were computed from a mechanism. Each site commits after the
first-passage time of a noisy switch through its fold (`PassageTime`: Fokker–Planck tables, with a constant or a
rising margin), with the sites ready at a regular spacing γ. The exact law of the order this produces is not of
product form, so it was projected (Kullback–Leibler) onto the costs η of a banded permanent with the same displacement
marginals (`OrderStatistics`), tabulated over the shape of the passage law and γ, and fitted to the angles
(`FoldAngleModel`, `FoldModel`). Variants: a margin that varies between sites ("B-mix", `MixturePassage`: a mixture of
passage laws, with a heavier late tail), and a readiness spacing that follows the plastochron along the stem ("tempo",
one exponent β, from `Plastochron`). The tag also holds a simulator of commitment-paced readiness (`Level1`: a site
becomes ready only once the older primordia that frame it have committed, so timing errors propagate), with the
statistics that would tell it from independent delays.

**Why it was removed** (2026-10-17). The two-step approach of the current package does the same job more generally:
the orders are reconstructed once by the propensity model, and any delay law, a fold's passage law included, is then
scored with its exact order law, without the projection, which cost the fold law up to 28 units of log-likelihood. On
the real stems, no fold passage law, single or mixed, came near a precise core with separate late and early tails
(gaps of 2 to 42 units across groups). The fold remains the interpretation of the delay ([manual §5.1](docs/src/timing.md)).
To test a fold passage law today, write it as a `DelayLaw` ([manual, "Custom delay laws"](docs/src/custom_laws.md)).

To work with the old code: `git checkout with-model-B` (in a separate worktree, e.g.
`git worktree add ../PhyllotaxisFold-modelB with-model-B`).
