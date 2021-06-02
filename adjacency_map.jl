using Qaintessent
using Qaintellect
using LinearAlgebra
using SparseArrays
using Flux
using IterTools: ncycle
# visualization
using Plots
using LaTeXStrings

"""
Construct adjacency map for a cartesian Nx × Ny lattice.
"""
function lattice_adjacency_map(Nx::Integer, Ny::Integer; pbc=true)
    L = Nx * Ny
    adjacency = zeros(Int, (L, L))
    for j in 0:Ny-1
        j_next = (j+1) % Ny
        for i in 0:Nx-1
            i_next = (i+1) % Nx
            # nearest neighbors
            if (pbc || i_next > 0) adjacency[j*Nx + i + 1, j*Nx + i_next + 1] = 1; end
            if (pbc || j_next > 0) adjacency[j*Nx + i + 1, j_next*Nx + i + 1] = 1; end
        end
    end
    adjacency = adjacency + transpose(adjacency)
    # only 0 or 1 entries
    return (adjacency .≠ 0)
end

"""
Construct horizontal adjacency map for a cartesian Nx × Ny lattice.
"""
function lattice_adjacency_map_horz(Nx::Integer, Ny::Integer; pbc=true)
    L = Nx * Ny
    adjacency = zeros(Int, (L, L))
    for j in 0:Ny-1
        for i in 0:Nx-1
            i_next = (i+1) % Nx
            # nearest neighbors along horizontal direction
            if (pbc || i_next > 0) adjacency[j*Nx + i + 1, j*Nx + i_next + 1] = 1; end
        end
    end
    adjacency = adjacency + transpose(adjacency)
    # only 0 or 1 entries
    return (adjacency .≠ 0)
end

"""
Construct vertical adjacency map for a cartesian Nx × Ny lattice.
"""
function lattice_adjacency_map_vert(Nx::Integer, Ny::Integer; pbc=true)
    L = Nx * Ny
    adjacency = zeros(Int, (L, L))
    for j in 0:Ny-1
        j_next = (j+1) % Ny
        for i in 0:Nx-1
            # nearest neighbors along vertical direction
            if (pbc || j_next > 0) adjacency[j*Nx + i + 1, j_next*Nx + i + 1] = 1; end
        end
    end
    adjacency = adjacency + transpose(adjacency)
    # only 0 or 1 entries
    return (adjacency .≠ 0)
end