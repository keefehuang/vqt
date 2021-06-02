using LinearAlgebra
using SparseArrays

"""
Compute `-log(Z)`, with `Z` the partition function.
"""
thermal_logZ(βH::AbstractMatrix) = -log(tr(exp(-Matrix(βH))))

"""
Compute the thermal state `exp(-βH) / Z`, with `Z` the partition function.
"""
function thermal_state(βH::AbstractMatrix)
    σ = exp(-Matrix(βH))
    return σ / tr(σ)
end

"""
    binary_entropy(p::Number)

Compute the binary entropy of `p` (natural logarithm).
"""
binary_entropy(p::Number) = -(xlogx(p) + xlogx(1 - p))


"""
Compute the von Neumann entropy of a density matrix `ρ`.
"""
neumann_entropy(ρ::AbstractMatrix) = -sum(xlogx.(real(eigvals(Matrix(ρ)))))


"""
Trace distance to target density matrix `σ`.
"""
trdistance(θlist, σ) = 0.5 * opnorm(matrix(apply(latent_density(θlist), cgc)) - σ, 1)
