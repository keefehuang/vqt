using Logging
using Qaintessent
using Qaintellect
using LinearAlgebra
using SparseArrays
using Flux
using IterTools: ncycle
# visualization
using Plots
using LaTeXStrings
include("./adjacency_map.jl")
include("./thermal_states.jl")
include("./construct_hamiltonian.jl")


"""
Identity map as sparse matrix with real-valued entries.
"""
sparse_identity(n) = sparse(1.0*I, n, n)

global debug = false
global visualize = false

"""
Construct a parametrized quantum layer consisting of single and two qubit rotation gates.
"""
function construct_parameterized_layer(L::Integer, istart::Integer)

    # TODO: take 2D adjacency into account? periodic boundary conditions?

    # local gates
    Uloc = [Moment([circuit_gate(i, RotationGate(0.05*π*randn(3))) for i in 1:L])]

    # interaction gates
    Uint = [
        Moment([circuit_gate(i, i+1, EntanglementXXGate(0.05π*randn())) for i in istart:2:L-1]),
        Moment([circuit_gate(i, i+1, EntanglementYYGate(0.05π*randn())) for i in istart:2:L-1]),
        Moment([circuit_gate(i, i+1, EntanglementZZGate(0.05π*randn())) for i in istart:2:L-1])]

    return vcat(Uloc, Uint)
end


function main()
    
    adj = lattice_adjacency_map(2, 3; pbc=false)
    # number of neighbors
    sum(adj, dims=1)

    # number of lattice sites
    L = size(adj, 1)
    adj_horz = lattice_adjacency_map_horz(2, 3; pbc=false)
    adj_vert = lattice_adjacency_map_vert(2, 3; pbc=false)

    # model parameters
    β = 2.6
    Jx = 1.0
    Jy = 0.6

    # construct Hamiltonian
    J = Jx * reshape(kron(adj_horz, [1, 1, 1]), (3, L, L)) + Jy * reshape(kron(adj_vert, [1, 1, 1]), (3, L, L))
    H = construct_hamiltonian(J)
    if debug
        @debug "Constructed Hamiltonian is equivalent to sample $(H ≈ H')"
        @assert(H ≈ H')
    end

    if visualize
        # show eigenvalues
        scatter(eigvals(Matrix(H)), xlabel=L"i", ylabel=L"\lambda_i", legend=false)
    end

    thermal_logZ(β*H)

    σtherm = thermal_state(β*H)
    # typeof(σtherm)

    # consistency check
    @assert tr(σtherm) - 1 ≈ 0 "Trace of thermal state should be 1"

    if visualize
        heatmap(log.(max.(σtherm, 1e-4)), title=L"\sigma_{\beta}", yflip=true, aspect_ratio=:equal)
    end
    
    # parameterized quantum circuit gate chain
    cgc = vcat(construct_parameterized_layer(L, 1), construct_parameterized_layer(L, 2), construct_parameterized_layer(L, 1));

    # use Hamiltonian as measurement operator; note that H gets converted to a complex matrix here
    Hmeas = MeasurementOperator(H, Tuple(1:L));

    # parameterized quantum circuit
    circ = Circuit{L}(cgc, [Hmeas])

    # use representation of density matrix in terms of Pauli matrices; cos(θ) is Bloch vector coefficient of Pauli-Z for each qubit
    latent_density(θlist) = DensityMatrix(kron([[1, 0, 0, cos(θ)] for θ in θlist]...), length(θlist))

    # example
    latent_density([π/3, 3π/8])

    ifelse(iszero(x), zero(result), result)

    if visualize
        # visualize binary entropy
        plot(0:0.05:1, binary_entropy.(0:0.05:1), xlabel="p", ylabel="S(p)", legend=false)
    end

    # target function: β tr[ρ H] - S(ρ), with ρ = U ρlatent U†
    ftarget(θlist) = β * apply(latent_density(θlist), circ)[1] - sum(binary_entropy.((1 .+ cos.(θlist)) / 2))

    # initial random θ parameters, which are to be optimized
    θopt = π/2 * (1 .+ 0.5*randn(L))

    # example
    ftarget(θopt)

    # consistency check
    @assert neumann_entropy(matrix(apply(latent_density(θopt), cgc))) - sum(binary_entropy.((1 .+ cos.(θopt)) / 2)) ≈ 0
    # example
    @assert trdistance(θopt, σtherm) ≈ 0

    # gather parameters from circuit
    paras = Flux.params(circ)
    # measurement operator (Hamiltonian) not "trainable" here
    delete!(paras, circ.meas[1].operator)
    # add θ parameters
    Flux.params!(paras, θopt)

    # there is not actually any input data for training
    data = ncycle([()], 200)

    # define optimizer
    opt = RMSProp(0.05)

    # define evaluation function
    evalcb() = println("ftarget(θopt): $(ftarget(θopt)), reference: $(thermal_logZ(β*H)); trdistance(θopt, σtherm): $(trdistance(θopt, σtherm))")

    # perform optimization
    Flux.train!(() -> ftarget(θopt), paras, data, opt, cb=Flux.throttle(evalcb, 4.0))

    # seems like the Ansatz is too restricted, or the optimization trapped in a local minimum, to further descrease the distance
    trdistance(θopt, σtherm)

    # optimized variational density matrix
    ρopt = apply(latent_density(θopt), cgc)

    if visualize
        # visualize optimized density matrix
        heatmap(log.(max.(real(matrix(ρopt)), 1e-4)), title=L"\rho_{\mathrm{opt}}", yflip=true, aspect_ratio=:equal)
    end
end


main()



















