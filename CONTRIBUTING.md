# Developing PhyllotaxisFold.jl

## Tests

```julia
] test PhyllotaxisFold
```

The tests check the numerical core against independent references (brute-force enumeration, Monte Carlo, finite
differences), the recovery of known parameters on simulated stems, and code quality with
[Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) and [ExplicitImports.jl](https://github.com/JuliaTesting/ExplicitImports.jl).

## The manual

The pages are in `docs/src`, and the `docs` environment points to this package (`[sources]`). The build commands are
at the top of `docs/make.jl`; the manual is written to `docs/build/index.html`. The repository is private, so the
manual is built locally; `docs/make.jl` holds the commented `deploydocs` call for later. The equations are rendered by
MathJax, which needs an internet connection.

CI (`.github/workflows/CI.yml`) runs the tests on Julia 1.10 and the latest release, and builds the manual headless
on every push to `main`. The built manual is kept as the run's `manual` artifact.

## Figures

The figures are written to `docs/src/figures` and committed, so building the manual does not remake them. They use
simulated data only. Each script says at its top how to run it.

- `docs/make_figures.jl`: the illustrations of the method, and the running example (`fig_examples()`).
- `docs/make_validation_figures.jl`: the validation on simulated stems (manual §6). `run_validation(; refit_timing =
  true)` refits only the delay laws, from the cached replicates.
- `docs/validation_clock_drift.jl`: the drifting-clock simulation of manual §6.7.
- `docs/make_workflow.jl`: the figures of the worked analysis.
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
- The earlier fold-commitment model and its variants are in the tag `with-model-B`.
