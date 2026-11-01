# Build the manual:
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'   (once; PhyllotaxisFold is a [sources] path dependency)
#   julia --project=docs docs/make.jl
# The pages are in docs/src; the figures in docs/src/figures are made by docs/make_figures.jl (illustrations) and
# docs/make_validation_figures.jl (§6, simulated stems), those of the worked analysis by docs/make_workflow.jl.

using Documenter, DocumenterCitations, PhyllotaxisFold

DocMeta.setdocmeta!(PhyllotaxisFold, :DocTestSetup, :(using PhyllotaxisFold); recursive = true)

# the references: docs/src/refs.bib, cited in the pages as [key](@cite) and listed on the notation page, in the
# author–year style of DocumenterCitations with the authors written "Last F." and separated by commas (the package's
# own style writes "Last, F." and separates them by semicolons)
const DC = DocumenterCitations

function _authors_comma(entry)
    names = map(entry.authors) do a
        words = filter(!isempty, split("$(a.first) $(a.middle)")) # each word its own initial ("C. A.")
        initials = join(DC._initial.(words), " ")
        name = join(filter(!isempty, [a.particle, a.last, initials, a.junior]), " ")
        return name
    end
    return strip(DC.tex_to_markdown(join(names, ", ", " and ")))
end

# in the text, the author's suffix is dropped: "(Ferrell, 2012)", not "(Ferrell Jr., 2012)"
function DC.format_citation(::Val{:authoryear_comma}, cit, entries, citations)
    md = DC.format_citation(Val(:authoryear), cit, entries, citations)
    for key in cit.keys, a in (haskey(entries, key) ? entries[key].authors : ())
        isempty(a.junior) || (md = replace(md, "$(a.last) $(a.junior)" => a.last))
    end
    return md
end
function DC.format_bibliography_reference(::Val{:authoryear_comma}, entry)
    ref = DC.format_bibliography_reference(Val(:authoryear), entry)
    authors = DC.format_names(entry; names = :lastfirst) # how the author–year style begins the reference
    startswith(ref, authors) || return ref
    return _authors_comma(entry) * ref[(ncodeunits(authors) + 1):end]
end
DC.bib_html_list_style(::Val{:authoryear_comma}) = :ul
DC.bib_sorting(::Val{:authoryear_comma}) = :nyt

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear_comma)

makedocs(;
    sitename = "PhyllotaxisFold.jl",
    modules = [PhyllotaxisFold],
    authors = "Geoffrey Schivre",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true", mathengine = Documenter.MathJax3(),
        size_threshold_ignore = ["api.md"], assets = ["assets/custom.css"]
    ),
    pages = [
        "Home" => "index.md",
        "A worked analysis" => "workflow.md",
        "Custom delay laws" => "custom_laws.md",
        "Simulating stems" => "simulation.md",
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
    plugins = [bib],
)

# The repository is private: the manual is built locally (docs/build/index.html). To publish it on GitHub Pages:
# deploydocs(; repo = "github.com/gschivre/PhyllotaxisFold.jl.git", devbranch = "main")
