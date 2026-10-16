"""
    PropensityPriors

Prior on free log-propensities η_1 > η_2 > … > η_k (all negative, decreasing in |d|), built from positive log-normal gaps:

    η_d = −(Δ_1 + … + Δ_d),   Δ_j ~ LogNormal(μΔ_j, σΔ_j)

`LogNormalSoftPlus` (d > 0) and `ReverseLogNormalSoftPlus` (the mirror, for η_{−k} < … < η_{−1} in the asymmetric
model), their bijectors to ℝ^k (θ_j = softplus⁻¹(Δ_j)), and `η_prior` (log-normal location and scale from the mean and
variance of each gap). Used by the free-η angle model (`PropensityModel`).
"""
module PropensityPriors

using LinearAlgebra, Random, Statistics
using Distributions, LogExpFunctions
using Bijectors: Bijectors

export LogNormalSoftPlus, ReverseLogNormalSoftPlus, η_prior, marginal_quantiles

struct LogNormalSoftPlus{V <: AbstractVector} <: ContinuousMultivariateDistribution
    μΔ::V
    σΔ::V
end
Base.length(d::LogNormalSoftPlus) = length(d.μΔ)
function Distributions._rand!(rng::AbstractRNG, d::LogNormalSoftPlus, x::AbstractVector{T}) where {T <: Real}
    Δ = rand(rng, MvLogNormal(d.μΔ, Diagonal(d.σΔ .^ 2)))
    cumsum!(x, Δ)
    x .*= -1
    return x
end
function Distributions._logpdf(d::LogNormalSoftPlus, η::AbstractVector{T}) where {T <: Real}
    k = length(d)
    @assert length(η) == k
    Δ₁ = -η[1]
    (Δ₁ > 0) || return -Inf
    logp = logpdf(LogNormal(d.μΔ[1], d.σΔ[1]), Δ₁)
    for j in 2:k
        Δⱼ = η[j - 1] - η[j]
        (Δⱼ > 0) || return -Inf
        logp += logpdf(LogNormal(d.μΔ[j], d.σΔ[j]), Δⱼ)
    end
    return logp # the linear map Δ ↦ η has |det| = 1
end
function Distributions.mean(d::LogNormalSoftPlus)
    Δm = exp.(d.μΔ .+ (d.σΔ .^ 2) ./ 2)
    return -cumsum(Δm)
end

# the same in reverse order: η = (η_{−k}, …, η_{−1}), gaps Δ_j = η_{j+1} − η_j and Δ_k = −η_{−1}
struct ReverseLogNormalSoftPlus{V <: AbstractVector} <: ContinuousMultivariateDistribution
    μΔ::V
    σΔ::V
end
Base.length(d::ReverseLogNormalSoftPlus) = length(d.μΔ)
function Distributions._rand!(rng::AbstractRNG, d::ReverseLogNormalSoftPlus, x::AbstractVector{<:Real})
    Δ = rand(rng, MvLogNormal(d.μΔ, Diagonal(d.σΔ .^ 2)))
    reverse!(Δ)
    cumsum!(x, Δ)
    x .*= -1
    reverse!(x)
    return x
end
function Distributions._logpdf(d::ReverseLogNormalSoftPlus, η::AbstractVector{<:Real})
    k = length(d)
    @assert length(η) == k
    Δₖ = -η[k]
    (Δₖ > 0) || return -Inf
    logp = logpdf(LogNormal(d.μΔ[k], d.σΔ[k]), Δₖ)
    for j in 1:(k - 1)
        Δⱼ = η[j + 1] - η[j]
        (Δⱼ > 0) || return -Inf
        logp += logpdf(LogNormal(d.μΔ[j], d.σΔ[j]), Δⱼ)
    end
    return logp
end
function Distributions.mean(d::ReverseLogNormalSoftPlus)
    Δm = exp.(d.μΔ .+ (d.σΔ .^ 2) ./ 2)
    return reverse!(-cumsum(reverse(Δm)))
end

"marginal quantiles of each η_d by Monte Carlo"
function marginal_quantiles(d::Union{LogNormalSoftPlus, ReverseLogNormalSoftPlus}, α::Real; N::Int = 100_000)
    samples = rand(d, N)
    return [quantile(samples[i, :], α) for i in 1:length(d)]
end

"""
    η_prior(m, v)

Log-normal (location μΔ, scale σΔ) of gaps with mean m and variance v (elementwise): σΔ² = log(1 + v / m²),
μΔ = log m − σΔ² / 2.
"""
function η_prior(m, v)
    σΔ = sqrt.(log1p.(v ./ (m .^ 2)))
    μΔ = log.(m) .- (σΔ .^ 2) ./ 2
    return (μΔ, σΔ)
end

# bijectors: constrained η ↦ unconstrained θ (θ_j = softplus⁻¹(Δ_j)) and back (η = −cumsum(softplus(θ)))
struct LogNormalSoftPlusBijector <: Bijectors.Bijector end
function (b::LogNormalSoftPlusBijector)(η::AbstractVector)
    k = length(η)
    θ = similar(η)
    θ[1] = invsoftplus(-η[1])
    for j in 2:k
        θ[j] = invsoftplus(η[j - 1] - η[j])
    end
    return θ
end
function Bijectors.with_logabsdet_jacobian(b::LogNormalSoftPlusBijector, η::AbstractVector)
    θ = b(η)
    return (θ, sum(x -> softplus(-x), θ))
end
struct ReverseLogNormalSoftPlusBijector <: Bijectors.Bijector end
function (b::ReverseLogNormalSoftPlusBijector)(η::AbstractVector)
    k = length(η)
    θ = similar(η)
    for j in 1:(k - 1)
        θ[j] = invsoftplus(η[j + 1] - η[j])
    end
    θ[k] = invsoftplus(-η[k])
    return θ
end
function Bijectors.with_logabsdet_jacobian(b::ReverseLogNormalSoftPlusBijector, η::AbstractVector)
    θ = b(η)
    return (θ, sum(x -> softplus(-x), θ))
end

# the Jacobian of θ ↦ η is triangular with diagonal −logistic(θ_j): log|J| = Σ log logistic(θ_j)
struct InverseLogNormalSoftPlusBijector <: Bijectors.Bijector end
(ib::InverseLogNormalSoftPlusBijector)(θ::AbstractVector) = -cumsum(softplus.(θ))
function Bijectors.with_logabsdet_jacobian(ib::InverseLogNormalSoftPlusBijector, θ::AbstractVector)
    return (ib(θ), sum(loglogistic, θ))
end
struct ReverseInverseLogNormalSoftPlusBijector <: Bijectors.Bijector end
function (ib::ReverseInverseLogNormalSoftPlusBijector)(θ::AbstractVector)
    Δ = softplus.(θ)
    return reverse!(.-cumsum!(reverse(Δ), reverse(Δ)))
end
function Bijectors.with_logabsdet_jacobian(ib::ReverseInverseLogNormalSoftPlusBijector, θ::AbstractVector)
    return (ib(θ), sum(loglogistic, θ))
end

Bijectors.inverse(::LogNormalSoftPlusBijector) = InverseLogNormalSoftPlusBijector()
Bijectors.inverse(::InverseLogNormalSoftPlusBijector) = LogNormalSoftPlusBijector()
Bijectors.inverse(::ReverseLogNormalSoftPlusBijector) = ReverseInverseLogNormalSoftPlusBijector()
Bijectors.inverse(::ReverseInverseLogNormalSoftPlusBijector) = ReverseLogNormalSoftPlusBijector()

Bijectors.bijector(::LogNormalSoftPlus) = LogNormalSoftPlusBijector()
Bijectors.bijector(::ReverseLogNormalSoftPlus) = ReverseLogNormalSoftPlusBijector()

end # module PropensityPriors
