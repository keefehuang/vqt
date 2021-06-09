using Base.Iterators
using Zygote

"""
    xlogx(x::Number)

Compute `x * log(x)`, returning zero if `x` is zero. (Copied from StatsFuns.jl package.)
"""
function xlogx(x::Number)
    result = x * log(x)
    ifelse(iszero(x), zero(result), result)
end

"""
Compute the von Neumann entropy of a density matrix `ρ`.
"""
neumann_entropy(ρ::AbstractMatrix) = -sum(xlogx.(real(eigvals(Matrix(ρ)))))

"""
    binary_entropy(p::Number)

Compute the binary entropy of `p` (natural logarithm).
"""
binary_entropy(p::Number) = -(xlogx(p) + xlogx(1 - p))


"""
Compute `-log(Z)`, with `Z` the partition function.
"""
thermal_logZ(βH::AbstractMatrix) = -log(tr(exp(-Matrix(βH))))


"""
Trace distance to target density matrix `σ`.
"""
trdistance(θlist, σ, cgc) = 0.5 * opnorm(real.(matrix(apply(latent_density(θlist), cgc))) - σ, 1)

"""
Use representation of density matrix in terms of Pauli matrices; cos(θ) is Bloch vector coefficient of Pauli-Z for each qubit
"""
latent_density(θlist) = DensityMatrix(kron([[1, 0, 0, cos(θ)] for θ in θlist]...), length(θlist))
# """
# Use representation of density matrix in terms of Pauli matrices; cos(θ) is Bloch vector coefficient of Pauli-Z for each qubit
# """
# function latent_density2(θlist)
#     ρ = kron([[θ1, 0, 0, θ2] for (θ1,θ2) in partition(θlist,2)]...)
#     return DensityMatrix(ρ ./ ρ[1], length(θlist)÷2)
# end

# Zygote.@adjoint function partition(θlist, step::Int)
#     return partition(θlist, step), θ->(collect(Base.Iterators.flatten(θ)), 0)
# end

function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
        if eltype(x) <: Complex
            # derivative with respect to imaginary part
            x[i] = tmp - im*δ/2
            y1 = f(xs...)
            x[i] = tmp + im*δ/2
            y2 = f(xs...)
            x[i] = tmp
            Δ[i] += im*(y2-y1)/δ
        end
    end
    return grads
end
