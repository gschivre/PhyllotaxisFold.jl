# Performance

A separate environment with the tools, developing the package from `..`:

```
julia --project=benchmark -t auto -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark -t auto benchmark/fits.jl     # timings
julia --project=benchmark -t auto benchmark/checks.jl   # JET and AllocCheck
```

`setup.jl` holds the fixtures: 40 simulated stems of 40 organs with the default grids (32 offsets × 16 κ nodes) and
outlier angles, and a second set with a twist (16 × 8 grid) for the twist fit.

## Reference timings

2026-11-01, Intel Core Ultra 5 135H (4 performance + 10 efficiency cores, 18 threads), Julia 1.13.3, `-t 18`. Median
of the benchmark samples; "before" is commit 23b537e (the API before the `PhyllotaxisFit` rework), same stems, same
point (the prior mean), same machine.

| at the prior mean, 40 stems × 40 organs | before | after |
|------------------------------------------|--------|-------|
| log posterior                            | 282 ms, 9.5 MB, 6842 allocations | 143 ms, 0.10 MB, 549 |
| + gradient (analytic)                    | 353 ms, 9.8 MB | 181 ms, 0.13 MB |
| + Hessian (dual pass over the gradient)  | 940 ms, 24 MB | 462 ms, 0.43 MB |
| one stem, value (one thread)             | 46 ms | 26 ms |
| `fit_map`, no twist (8 Newton steps)     | 9.4 s | 3.6 s |
| `fit_map` with the twist (`fit_map_torsion!` before) | 125 s | 68 s |

Both versions give the same MAP and the same integrated log-likelihood (−1796.2923130126) to 1e-14. At the MAP, a
Hessian costs 242 ms against 148 ms for a value (4 % of the grid points active in the dual pass).

## What was changed, and why

- **Type stability.** The parameter bijector was a `Stacked` over a `Vector{Any}` of bijectors: `binv(u)` was inferred
  as `Any`, and every call that followed it (η, the nuisance parameters, the stem likelihood itself, the chain rule)
  went through dynamic dispatch. It is now a `Stacked` over a tuple. Whether angles can be outliers is a type
  parameter of the model, so `π_o` is either a number or `nothing` at compile time and the branches on it fold away.
  ForwardDiff chooses its chunk size at run time, so the Jacobian of the bijector and the derivatives of the prior are
  type-asserted; the dual Hessian passes sit behind a function barrier. JET (`checks.jl`) now finds one runtime
  dispatch per entry point, that barrier.
- **Allocations.** The log posterior allocated 9.5 MB per call: a workspace (with its permutation graph, built from a
  `Dict`) per chunk of stems and per call, the angle grid per stem, the corrected angles per stem, and the Jacobian of
  the bijector per stem (`ForwardDiff.jacobian` once per stem, and once more per twist node). Now the model keeps a
  pool of workspaces (one per element type) and one graph; the per-stem gradients are summed in the constrained space
  and the chain rule is applied once per evaluation; `logbesseli` no longer allocates. AllocCheck proves the kernels
  (`forward!`, `forward_linear!`, `backward!`, the weight and cosine tables) cannot allocate.
- **Threads.** One task per stem, longest first, instead of 2 × nthreads fixed chunks.
- **The permanent.** The scaled forward pass is written in pull form (each state sums its incoming edges: one store
  per state instead of scattered updates) and, with the weights of the columns outside 1..n set to 0, the inner loops
  of the scaled forward and backward passes need no bounds test: forward 17.6 → 11.1 µs, backward 18.9 → 14.1 µs on a
  stem of 40 organs, identical to the log-space pass to 1e-13.
- **Outlier weights.** With outliers, each edge weight was `η_d + logaddexp(log(1 − π) + κ (cos − 1), log π + log I₀(κ) − κ)`,
  then exponentiated by the scaled pass: an exp, a log1p and an exp per entry. The uniform term keeps the mixture
  weight away from underflow, so it is now filled in linear form and passed to `forward_linear!` (one exp per entry);
  the posterior responsibility of the von Mises term in the gradient is `logistic(a − b)` (one exp instead of three).
- **The Hessian pass.** The dual pass (9 partials) ran on all 2048 grid points of each stem. A Float64 pass at the same
  point now marks the points within e^-40 of the best (at most 2048 e^-40 ≈ 1e-14 of the stem's likelihood is left
  out); at the prior mean 30 % of the points remain, at the MAP 4 %.

Not changed: pruning the forward passes by an upper bound (perm(A) ≤ ∏ row sums) would skip only 12 % of the grid
points at the prior mean with outliers, and was left out. DispatchDoctor's `@stable` was not adopted: it would make
the package depend on it, and the `@inferred` tests (test/runtests.jl, "type stability and allocations") guard the
same functions.
