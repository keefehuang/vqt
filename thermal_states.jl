using LinearAlgebra
using SparseArrays

"""
Compute the thermal state `exp(-βH) / Z`, with `Z` the partition function.
"""
function thermal_state(βH::AbstractMatrix)
    σ = exp(-Matrix(βH))
    return σ / tr(σ)
end


