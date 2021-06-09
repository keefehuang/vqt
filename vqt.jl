using Logging
using Qaintessent
# using Qaintellect
include("/home/ga53vuw/Documents/PhD/projects/QAI/Qaintellect.jl/src/Qaintellect.jl")
using LinearAlgebra
using SparseArrays
using Flux
using IterTools: ncycle
# visualization
using Plots
using LaTeXStrings
include("./util.jl")
include("./adjacency_map.jl")
include("./thermal_states.jl")
include("./construct_hamiltonian.jl")


"""
Identity map as sparse matrix with real-valued entries.
"""
sparse_identity(n) = sparse(1.0*I, n, n)

global debug = false
global visualize = true

global Nx = 2
global Ny = 3
global isboundary = true

@assert Nx > 0 "Dimension of lattice must be positive and greater than 0"
@assert Ny > 0 "Dimension of lattice must be positive and greater than 0"

function single_qubit_layer(L::Integer)
    # local gates
    # [circuit_gate(i, RotationGate(0.05*π*randn(3))) for i in 1:L]
    cgc = CircuitGate[]
    append!(cgc, [circuit_gate(i, RxGate(0.05*π*randn())) for i in 1:L])
    append!(cgc, [circuit_gate(i, RyGate(0.05*π*randn())) for i in 1:L])
    append!(cgc, [circuit_gate(i, RzGate(0.05*π*randn())) for i in 1:L])
    cgc
end

"""
Construct a parametrized quantum layer consisting of single and two qubit rotation gates.
"""
function trotterized_layer_1d(L::Integer, istart::Integer; boundary::Bool=false)
    # interaction gates
    vcat(
        [circuit_gate(i, i+1, EntanglementXXGate(0.05π*randn())) for i in istart:2:L-1],
        ifelse(boundary, [circuit_gate(L, 1, EntanglementXXGate(0.05π*randn()))], []),
        [circuit_gate(i, i+1, EntanglementYYGate(0.05π*randn())) for i in istart:2:L-1],
        ifelse(boundary, [circuit_gate(L, 1, EntanglementYYGate(0.05π*randn()))], []),
        [circuit_gate(i, i+1, EntanglementZZGate(0.05π*randn())) for i in istart:2:L-1],
        ifelse(boundary, [circuit_gate(L, 1, EntanglementZZGate(0.05π*randn()))], [])
    )
end


function trotterized_layer_2d_h(Nx::Integer, Ny::Integer, istart::Integer; boundary::Bool=false)
    # interaction gates
    h = Iterators.flatten(repeat([i for i in istart:2:Nx-1], 1, Ny) .+ transpose([0:Ny-1...].*Nx))
    h_boundary = fill(Nx, Ny) .+ [0:Ny-1...].*Nx
    L = Nx * Ny
    vcat(
        [circuit_gate(i, i+1, EntanglementXXGate(0.05π*randn())) for i in h],
        ifelse(boundary&&(isodd(Nx)==isodd(istart)), [circuit_gate(i, i-Nx+1, EntanglementXXGate(0.05π*randn())) for i in h_boundary], []),        
        [circuit_gate(i, i+1, EntanglementYYGate(0.05π*randn())) for i in h],
        ifelse(boundary&&(isodd(Nx)==isodd(istart)), [circuit_gate(i, i-Nx+1, EntanglementYYGate(0.05π*randn())) for i in h_boundary], []),
        [circuit_gate(i, i+1, EntanglementZZGate(0.05π*randn())) for i in h],
        ifelse(boundary&&(isodd(Nx)==isodd(istart)), [circuit_gate(i, i-Nx+1, EntanglementZZGate(0.05π*randn())) for i in h_boundary], []),
    )
end

function trotterized_layer_2d_v(Nx::Integer, Ny::Integer, istart::Integer; boundary::Bool=false)
    v = Iterators.flatten(repeat([i for i in 1:Nx], 1, Ny÷2-ifelse(iseven(Ny)&&iseven(istart), 1, 0)) .+ transpose([istart-1:2:Ny-2...].*Nx))
    v_boundary = [1:Nx...] .+ Nx*(Ny-1)
    L = Nx * Ny
    vcat(
        [circuit_gate(i, i+Nx, EntanglementXXGate(0.05π*randn())) for i in v],
        ifelse(boundary&&(isodd(Ny)==isodd(istart)), [circuit_gate(i, j, EntanglementXXGate(0.05π*randn())) for (i,j) in zip(v_boundary, 1:Nx)], []),
        [circuit_gate(i, i+Nx, EntanglementYYGate(0.05π*randn())) for i in v],
        ifelse(boundary&&(isodd(Ny)==isodd(istart)), [circuit_gate(i, j, EntanglementYYGate(0.05π*randn())) for (i,j) in zip(v_boundary, 1:Nx)], []),
        [circuit_gate(i, i+Nx, EntanglementZZGate(0.05π*randn())) for i in v],
        ifelse(boundary&&(isodd(Ny)==isodd(istart)), [circuit_gate(i, j, EntanglementZZGate(0.05π*randn())) for (i,j) in zip(v_boundary, 1:Nx)], [])
    )
end

function main()
    
    adj = lattice_adjacency_map(Nx, Ny; pbc=false)
    
    
    # number of neighbors
    sum(adj, dims=1)

    # number of lattice sites
    L = size(adj, 1)

    adj_horz = lattice_adjacency_map_horz(Nx, Ny; pbc=false)
    adj_vert = lattice_adjacency_map_vert(Nx, Ny; pbc=false)

    # display(adj_vert)
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
    @assert tr(σtherm)  ≈ 1 "Trace of thermal state should be 1"

    if visualize
        heatmap(log.(max.(σtherm, 1e-4)), title=L"\sigma_{\beta}", yflip=true, aspect_ratio=:equal)
    end
    
    # parameterized quantum circuit gate chain
    cgc = CircuitGate[]
    layers = 8
    for _ in layers
        append!(cgc, single_qubit_layer(L))
        append!(cgc, trotterized_layer_1d(L, 1, boundary=isboundary))
        append!(cgc, trotterized_layer_1d(L, 2, boundary=isboundary))
        
        append!(cgc, single_qubit_layer(L))
        append!(cgc, trotterized_layer_2d_h(Nx, Ny, 1; boundary=isboundary))
        append!(cgc, trotterized_layer_2d_h(Nx, Ny, 2; boundary=isboundary))
        append!(cgc, single_qubit_layer(L))
        append!(cgc, trotterized_layer_2d_v(Nx, Ny, 1; boundary=isboundary))
        append!(cgc, trotterized_layer_2d_v(Nx, Ny, 2; boundary=isboundary))

    end
    append!(cgc, single_qubit_layer(L))

    # use Hamiltonian as measurement operator; note that H gets converted to a complex matrix here
    Hmeas = MeasurementOperator(H, Tuple(1:L));
    

    # parameterized quantum circuit
    circ = Circuit{L}(cgc, [Hmeas])

    # example
    latent_density([π/3, 3π/8])


    if visualize
        # visualize binary entropy
        plot(0:0.05:1, binary_entropy.(0:0.05:1), xlabel="p", ylabel="S(p)", legend=false)
    end

    # target function: β tr[ρ H] - S(ρ), with ρ = U ρlatent U†
    ftarget(θlist) = β * apply(latent_density(θlist), circ)[1] - sum(binary_entropy.((1 .+ cos.(θopt)) ./ 2))

    # initial random θ parameters, which are to be optimized
    θopt = π/2 .* (1 .+ 0.5*randn(L))
    # θopt = π/2 .* (6 .+ 3 .* randn(L))
    # # consistency check
    @assert neumann_entropy(matrix(apply(latent_density(θopt), cgc))) ≈ sum(binary_entropy.((1 .+ cos.(θopt)) ./ 2))
    @assert neumann_entropy(matrix(latent_density(θopt))) ≈ sum(binary_entropy.((1 .+ cos.(θopt)) ./ 2))
    f_neumann(θopt) = neumann_entropy(matrix(apply(latent_density(θopt), cgc)))
    f_binary(θopt) = sum(binary_entropy.((1 .+ cos.(θopt)) ./ 2))
    @assert all(isapprox.(ngradient(f_neumann, θopt), ngradient(f_binary, θopt), rtol=1e-5))

    #example
    @show ftarget(θopt)

    # gather parameters from circuit
    paras1 = Flux.params(circ)
    # paras2 = Flux.params(θopt)
    # measurement operator (Hamiltonian) not "trainable" here
    delete!(paras1, circ.meas[1].operator)
    # add θ parameters
    Flux.params!(paras1, θopt)

    # there is not actually any input data for training
    data = ncycle([()], 6000)

    # define optimizer
    opt = RMSProp()
    # opt = AdaMax()

    # define evaluation function
    evalcb() = println("ftarget(θopt): $(ftarget(θopt)), reference: $(thermal_logZ(β*H)); trdistance(θopt, σtherm, cgc): $(trdistance(θopt, σtherm, cgc))")

    # perform optimization
    
    Flux.train!(() -> ftarget(θopt), paras1, data, opt, cb=Flux.throttle(evalcb, 2))
    # println("alternating training")
    # Flux.train!(() -> ftarget(θopt), paras1, data, opt, cb=Flux.throttle(evalcb, 2))
    # println("alternating training")
    # Flux.train!(() -> ftarget(θopt), paras2, data, opt, cb=Flux.throttle(evalcb, 2))
    # println("alternating training")
    # Flux.train!(() -> ftarget(θopt), paras1, data, opt, cb=Flux.throttle(evalcb, 2))

    # seems like the Ansatz is too restricted, or the optimization trapped in a local minimum, to further descrease the distance
    @show trdistance(θopt, σtherm, cgc)

    # optimized variational density matrix
    ρopt = apply(latent_density(θopt), cgc)
    # println()
    # display(real.(matrix(ρopt)))
    # println()
    # display(σtherm)

    if visualize
        # visualize optimized density matrix
        heatmap(log.(max.(real(matrix(ρopt)), 1e-4)), title=L"\rho_{\mathrm{opt}}", yflip=true, aspect_ratio=:equal)
    end
end


main()



















