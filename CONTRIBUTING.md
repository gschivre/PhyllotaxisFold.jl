# Contributing to PhyllotaxisFold.jl

Issues and pull requests are welcome. For anything beyond a small fix, please open an issue first to discuss it.

## Tests

```julia
] test PhyllotaxisFold
```

The suite checks the numerical core against independent references (brute-force enumeration, ForwardDiff or finite
differences for every analytic gradient and Hessian), recovery of known parameters on simulated stems, type stability
and allocations of the hot paths (`@inferred`, `@allocated`), the code examples of the manual and the README, and code
quality ([Aqua.jl](https://github.com/JuliaTesting/Aqua.jl), [ExplicitImports.jl](https://github.com/JuliaTesting/ExplicitImports.jl)).
CI runs it on Julia 1.10 and the latest release.

A change to the likelihood should come with a test against an independent reference, not only a regression value.

## Conventions

- **Generic number types** wherever a function can see ForwardDiff duals: the Hessian is one dual pass over the
  analytic gradient.
- **Fits** are done in unconstrained parameters by Newton's method with a trust region.
- **Hot paths** stay type-stable and non-allocating; check with the tools in `benchmark/` (see below).
- **Formatting:** [Runic.jl](https://github.com/fredrikekre/Runic.jl) (`runic --inplace src test docs/*.jl`).
- **Comments** explain why, not what; the method itself is documented in the manual.

## Performance

`benchmark/` is a separate environment (BenchmarkTools, JET, AllocCheck) that develops the package from `..`; run it
with threads (`julia --project=benchmark -t auto`). `benchmark/fits.jl` times the log posterior, its derivatives and
the fits; `benchmark/checks.jl` reports runtime dispatch (JET) and allocations (AllocCheck). Reference timings are in
`benchmark/README.md`.

## The manual

The pages are in `docs/src`; build them with the commands at the top of `docs/make.jl`
(output in `docs/build/index.html`). References go in `docs/src/refs.bib` and are cited by key with
[DocumenterCitations.jl](https://github.com/JuliaDocs/DocumenterCitations.jl).

Figures use simulated data only, are committed in `docs/src/figures`, and are made by the `docs/make_*.jl` and
`docs/validation_*.jl` scripts (each says at its top how to run it). Slow fits are cached in `docs/*.jls`, which are
not committed: delete a cache to recompute it.
