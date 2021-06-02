using LinearAlgebra
using SparseArrays

"""
Construct a Heisenberg-type Hamiltonian as sparse matrix using site-dependent interaction strength, with nearest-neighbor interactions according to adjacency.
"""
function construct_hamiltonian(J::Array{<:Real,3})
    L = size(J, 3)
    @assert(size(J) == (3, L, L))

    # spin operators (Pauli matrices divided by 2)
    sigma12 = (0.5*sparse([0.  1.; 1.  0.]),
               0.5*sparse([0. -im; im  0.]),
               0.5*sparse([1.  0.; 0. -1.]))

    H = spzeros(Float64, 2^L, 2^L)

    # interaction terms
    for i in 1:L
        for j in i+1:L
            for k in 1:3
                # considering only entries in J for i < j
                if J[k, i, j] ≠ 0
                    H -= J[k, i, j] * real(kron(sparse_identity(2^(L-j)), sigma12[k], sparse_identity(2^(j-i-1)), sigma12[k], sparse_identity(2^(i-1))))
                end
            end
        end
    end

    return H
end
