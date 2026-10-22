# Build the manual:
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'   (once; PhyllotaxisFold is a [sources] path dependency)
#   julia --project=docs docs/make.jl
# The pages are in docs/src; the figures in docs/src/figures are made by docs/make_figures.jl (illustrations) and
# docs/make_validation_figures.jl (§6, simulated stems).

using Documenter, PhyllotaxisFold

DocMeta.setdocmeta!(PhyllotaxisFold, :DocTestSetup, :(using PhyllotaxisFold); recursive = true)

makedocs(;
    sitename = "PhyllotaxisFold.jl",
    modules = [PhyllotaxisFold],
    authors = "Geoffrey Schivre",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true", mathengine = Documenter.MathJax3(),
        size_threshold_ignore = ["api.md"]
    ),
    pages = [
        "Home" => "index.md",
        "Method" => [
            "The problem" => "problem.md",
            "Angles given an order" => "angles.md",
            "Summing over all orders" => "orders.md",
            "The propensity model" => "propensity.md",
            "Commitment timing" => "timing.md",
        ],
        "Validation on simulated stems" => "validation.md",
        "Assumptions and limits" => "limits.md",
        "Notation and references" => "notation.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

# The repository is private: the manual is built locally (docs/build/index.html). To publish it on GitHub Pages:
# deploydocs(; repo = "github.com/gschivre/PhyllotaxisFold.jl.git", devbranch = "main")
