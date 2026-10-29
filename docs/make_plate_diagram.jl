# Render the plate diagram of manual §2.5 (docs/plate_diagram.tex, TikZ + bayesnet) to docs/src/figures:
#   julia docs/make_plate_diagram.jl
# Tectonic (TeX engine, fetches the LaTeX packages it needs) and Poppler (pdftocairo, PDF to SVG) are installed in a
# temporary environment, so neither LaTeX nor the docs environment is needed. The PDF is kept next to the SVG, for
# reuse outside the manual.

using Pkg
Pkg.activate(; temp = true)
Pkg.add(["Tectonic", "Poppler_jll"]; io = devnull)
using Tectonic, Poppler_jll

const SRC = joinpath(@__DIR__, "plate_diagram.tex")
const OUT = joinpath(@__DIR__, "src", "figures")

mktempdir() do dir
    tectonic(bin -> run(`$bin --chatter minimal --outdir $dir $SRC`))
    pdf = joinpath(dir, "plate_diagram.pdf")
    cp(pdf, joinpath(OUT, "m3_plate_diagram.pdf"); force = true)
    run(`$(pdftocairo()) -svg $pdf $(joinpath(OUT, "m3_plate_diagram.svg"))`)
end
