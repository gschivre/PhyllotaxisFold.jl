# Timings of the log posterior, its derivatives and the two fits on the stems of setup.jl:
#   julia --project=benchmark -t auto benchmark/fits.jl
# (reference, 2026-11-01, Intel Core Ultra 5 135H, 18 threads, Julia 1.13: see benchmark/README.md)
include(joinpath(@__DIR__, "setup.jl"))

# compile everything on a small model first
let m = PhyllotaxisModelAD(SIM.angles[1:3]; ngrid_ϕ = 4, ngrid_κ = 3, torsion = TorsionPrior(st = 5.0))
    fit_map(m; recenter = 0)
    fit_map(PhyllotaxisModelAD(SIM.angles[1:3]; ngrid_ϕ = 4, ngrid_κ = 3))
end

results = (;
    value = summ(@benchmark AL.angle_vgh($MB, $UB, nothing, 0) samples = 10 evals = 1),
    gradient = summ(@benchmark AL.angle_vgh($MB, $UB, nothing, 1) samples = 10 evals = 1),
    hessian = summ(@benchmark AL.angle_vgh($MB, $UB, nothing, 2) samples = 5 evals = 1),
    fit_no_twist_s = (@elapsed fb = fit_map(MB)),
    fit_twist_s = (@elapsed ft = fit_map(MT)),
)
display(results)
display(fb)
display(ft)
