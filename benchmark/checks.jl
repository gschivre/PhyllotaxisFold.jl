# Static checks of the hot paths:
#   julia --project=benchmark -t auto benchmark/checks.jl
# JET: every runtime dispatch in the call trees of the log posterior and its derivatives, with and without twist.
# AllocCheck: proof that the kernels of the banded permanent and of the weights cannot allocate.
include(joinpath(@__DIR__, "setup.jl"))
using JET, AllocCheck

# the runtime dispatches of a JET report, one line per site (its caller and the call)
function dispatches(r)
    lines = split(sprint(show, r), '\n')
    out = String[]
    for (i, l) in enumerate(lines)
        occursin("runtime dispatch", l) || occursin("captured variable", l) || continue
        caller = i > 1 ? strip(replace(lines[i - 1], r"^[│┌└ ]+" => "")) : ""
        push!(out, first(caller, 120) * "\n      " * first(strip(replace(l, r"^[│┌└ ]+" => "")), 160))
    end
    return unique(out)
end

# the twist model with its nodes (placed by a fit at coarse settings)
fit_map(MT; recenter = 0, nodes_kw = (; nnodes = 3))
const NODES = MT.nodes[]
const VT = vcat(prior_mean_parameters(MT)[1:AL.nmodel(MT)], 1.0, 0.0)

jet = (;
    value = dispatches(JET.@report_opt target_modules = (PF,) AL.angle_vgh(MB, UB, nothing, 0)),
    gradient = dispatches(JET.@report_opt target_modules = (PF,) AL.angle_vgh(MB, UB, nothing, 1)),
    hessian = dispatches(JET.@report_opt target_modules = (PF,) AL.angle_vgh(MB, UB, nothing, 2)),
    twist_gradient = dispatches(JET.@report_opt target_modules = (PF,) AL.twist_vgh(MT, NODES, VT, 1)),
    twist_hessian = dispatches(JET.@report_opt target_modules = (PF,) AL.twist_vgh(MT, NODES, VT, 2)),
)
for (k, v) in pairs(jet)
    println("JET $(k): $(length(v)) runtime dispatch site(s)")
    foreach(x -> println("    ", x), v)
end

const WS = typeof(FWS.perm)
allocs = (;
    forward = check_allocs(PE.forward!, (WS, Int)),
    forward_linear = check_allocs(PE.forward_linear!, (WS, Int)),
    backward = check_allocs(PE.backward!, (WS, Int)),
    fill_weights = check_allocs(AL._fill_weights!, (Matrix{Float64}, Vector{Float64}, Array{Float64, 3}, Int, Float64, Int, Int, Tuple{Float64, Float64})),
    fill_weights_no_outlier = check_allocs(AL._fill_weights!, (Matrix{Float64}, Vector{Float64}, Array{Float64, 3}, Int, Float64, Int, Int, Nothing)),
    fill_linear = check_allocs(AL._fill_linear!, (Matrix{Float64}, Vector{Float64}, Array{Float64, 3}, Int, Float64, Int, Int, Tuple{Float64, Float64})),
    fill_cosines = check_allocs(AL._fill_cosines!, (typeof(FWS), typeof(MB.grid), SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}, Float64)),
)
for (k, v) in pairs(allocs)
    println("AllocCheck $(k): ", isempty(v) ? "cannot allocate" : "$(length(v)) possible allocation(s)")
    foreach(a -> println("    ", sprint(show, a)[1:min(end, 300)]), v)
end
