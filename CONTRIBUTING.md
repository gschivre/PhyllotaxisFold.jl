# Developing PhyllotaxisFold.jl

## Tests

```julia
] test PhyllotaxisFold
```

The tests take about 4 minutes. They cover:

- the dynamic programming of the banded permanent against brute-force enumeration;
- the analytic gradients and Hessians, and the Bessel functions with their dual derivatives;
- the twist quadrature;
- the decoding and the posterior sampling on simulated stems;
- the exact order likelihood against Monte Carlo;
- the re-weighting identity;
- the exact derivatives of the delay-law fit, and the shared-core fit and test;
- the overtaking profile (closed form, simulation, model-free estimate, comparison of groups);
- the recovery of a delay law from the angles alone;
- code quality: [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) (method ambiguities, unbound type parameters,
  undefined exports, stale dependencies, compat entries, type piracy) and no unused explicit imports
  ([ExplicitImports.jl](https://github.com/JuliaTesting/ExplicitImports.jl)).

## The manual

The pages are in `docs/src`, and the `docs` environment points to this package (`[sources]`). To build the manual:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=docs docs/make.jl                        # then open docs/build/index.html
```

The repository is private, so the manual is built locally; `docs/make.jl` holds the commented `deploydocs` call for
later. The equations are rendered by MathJax, which needs an internet connection.

CI (`.github/workflows/CI.yml`) runs the tests on Julia 1.10 and the latest release, and builds the manual headless
on every push to `main`. The built manual is kept as the run's `manual` artifact.

## Figures

All figures use simulated data only (CairoMakie, in the `docs` environment) and are written to `docs/src/figures`.

- `docs/make_figures.jl`: the illustrations of the method, and the running example (`fig_examples()`; its fit,
  about 5 minutes on 18 threads, is cached in `docs/example.jls`).
- `docs/make_validation_figures.jl`: the validation on simulated stems (manual §6; about 2 h on 18 threads, the
  replicates are cached in `docs/validation.jls` with their posterior order samples, and
  `run_validation(; refit_timing = true)` refits only the delay laws, in about 15 minutes).
- `docs/validation_clock_drift.jl`: the drifting-clock simulation of manual §6.7 (about 2 minutes on 18 threads,
  cached in `docs/clock_drift.jls`).

The caches (`*.jls`) and `docs/build` are not committed.

## Conventions

- No data and no analysis of real plants in this repository.
- Julia code: generic in the number type wherever a function can see ForwardDiff duals. Optimisation is done in
  unconstrained parameters, by Newton's method with a trust region (value, gradient and Hessian in one call): an
  `only_fgh!` callback where the gradient is analytic (the angle model), `TwiceDifferentiable` with
  `AutoForwardDiff()` otherwise (the delay law, the flowering rate).
- Formatting: [Runic.jl](https://github.com/fredrikekre/Runic.jl) (`runic --inplace src test docs/*.jl`).
- The earlier fold-commitment model and its variants are in the tag `with-model-B`.
