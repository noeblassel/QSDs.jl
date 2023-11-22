include("ParRep.jl"); using .ParRep, Base.Threads,Random, LinearAlgebra


# n_transitions = parse(Int64,ARGS[1])
# freq_checkpoint = parse(Int64,ARGS[2])
# n_replicas = parse(Int64,ARGS[3])
# pot_type = parse(ARGS[4])


### Gelman-Rubin convergence test

### Polyhedral states
const minima = [-1.0480549928242202 0.0 1.0480549928242202; -0.042093666306677734 1.5370820044494633 -0.042093666306677734]
const saddles = [-0.6172723078764598 0.6172723078764598 0.0; 1.1027345175080963 1.1027345175080963 -0.19999999999972246]
const neg_eigvecs = [0.6080988038706289 -0.6080988038706289 -1.0; 0.793861351075306 0.793861351075306 0.0]
const λs_minima = [8.472211868406559,3.7983868797287945,8.472211868406559]
const λs_saddles = [5.356906312070659,5.356906312070659,11.399661817554989]#neg eigvals

# matrix associating transitions with saddles (J_{ij} = index of saddle point from i to j)
const J = [0 1 3; 1 0 2; 3 2 0]
# matrix giving the sign convention (σ_{ij}*neg_eigvecs[:,J_{ij}] = eigenvector pointing from state i to state j at z_{J_{ij}}
const σ = [0 1 -1; -1 0 -1; 1 1 0]

Base.@kwdef mutable struct GelmanRubinDiagnostic{F}
    observables::F
    num_replicas::Int64
    means = Matrix{Float64}(undef,length(observables),num_replicas)
    sq_means = Matrix{Float64}(undef,length(observables),num_replicas)
    burn_in = 1
    tol = 5e-2
    ratio = 0.0
end

@inbounds function ParRep.check_dephasing!(checker::GelmanRubinDiagnostic,replicas,current_macrostate,step_n)
    
    if step_n == 1 # initialize running mean and running square_mean buffer
        fill!(checker.means,zero(Float64))
        fill!(checker.sq_means,zero(Float64))
    end

    m,n = size(checker.means)
    
    @threads for i=1:n
        for j=1:m
            f = checker.observables[j]
            val = f(replicas[i],current_macrostate)
            sq_val = val^2

            checker.means[j,i] += (val-checker.means[j,i]) / step_n # online way to compute the running mean
            checker.sq_means[j,i] += (sq_val-checker.sq_means[j,i]) / step_n

        end
    end

    (step_n < checker.burn_in) && return false

    Obar = sum(checker.means;dims = 2) / n

    for j=1:m
        checker.ratio = sum(checker.sq_means[j,i] -2checker.means[j,i]*Obar[j] + Obar[j]^2 for i=1:n)/sum(checker.sq_means[j,i]-checker.means[j,i]^2 for i=1:n)-1.0
        (checker.ratio > checker.tol) && return false
    end

    return true
end


### Harmonic Diagnostic
Base.@kwdef mutable struct HarmonicDiagnostic{T}
    λ₂::Vector{T} # a vector of λ₂s associated with each state
    m::Int # number of 1/λ2s elapsed before reaching QSD
    dt::T # timestep of simulation
end


# time > n/λ₂
@inbounds function ParRep.check_dephasing!(checker::HarmonicDiagnostic,replicas,current_macrostate,step_n)
    return step_n*checker.dt*checker.λ₂[current_macrostate]> checker.m
end


### Entropic switch

"""
    entropic_switch(x, y)

Potential energy function.
"""
function entropic_switch(x, y)
    tmp1 = x^2
    tmp2 = (y - 1 / 3)^2
    return 3 * exp(-tmp1) * (exp(-tmp2) - exp(-(y - 5 / 3)^2)) - 5 * exp(-y^2) * (exp(-(x - 1)^2) + exp(-(x + 1)^2)) + 0.2 * tmp1^2 + 0.2 * tmp2^2
end

function entropic_switch(q)
    return entropic_switch(q...)
end

"""
    ∇entropic_switch(x, y)

Gradient of the potential energy function.
"""


grad_entropic_switch(q) = begin X = zeros(2); drift_entropic_switch!(X,-1); return X end


struct PolyhedralStateChecker{T}
    α::Matrix{T}
end

PolyhedralStateChecker(β,m) = begin
    α = [0 m/√(β*λs_saddles[J[1,2]]) m/√(β*λs_saddles[J[1,3]]);
    m/√(β*λs_saddles[J[2,1]]) 0 m/√(β*λs_saddles[J[2,3]]);
    m/√(β*λs_saddles[J[3,1]]) m/√(β*λs_saddles[J[3,2]]) 0]

    return PolyhedralStateChecker(α)
end
#δ ≫ 1/√(βκ)

function ParRep.get_macrostate!(checker::PolyhedralStateChecker,walker,current_state,step_n)
    if current_state === nothing
        for i=1:3
            is_in_i = true

            for j=i+1:3
                is_in_i = is_in_i && (σ[i,j]*dot(walker-saddles[:,J[i,j]],neg_eigvecs[:,J[i,j]]) < 0)
            end

            (is_in_i) && return i
        end

    end

    i = current_state

    for j=1:3
        (j != i) && (σ[i,j]*dot(walker-saddles[:,J[i,j]],neg_eigvecs[:,J[i,j]]) >= checker.α[i,j]) && return j
    end

    return i
end

Base.@kwdef struct EMSimulator{B,Σ,R}
    dt::Float64
    β::Float64
    σ::Float64 = sqrt(2dt/β)
    drift!::B
    diffusion!::Σ
    n_steps=1
    rng::R = Random.GLOBAL_RNG
end

function ParRep.update_microstate!(simulator::EMSimulator,X)
    for k=1:simulator.n_steps
        simulator.drift!(X,simulator.dt)
        simulator.diffusion!(X,simulator.dt,simulator.σ,simulator.rng)
    end
end

function drift_entropic_switch!(X,dt)

    tmp1 = exp(4X[1])
    tmp2 = exp(-X[1]^2 - 2X[1] - X[2]^2 - 1)
    tmp3 = exp(-X[1]^2)
    tmp4 = exp(-(X[2]-1/3)^2)
    tmp5 = exp(-(X[2]-5/3)^2)

    X[1] -= dt * ( 0.8X[1]^3 + 10*(tmp1*(X[1] - 1) + X[1] + 1)*tmp2 - 6tmp3*X[1]*(tmp4 - tmp5) )
    X[2] -=  dt * (  10*(tmp1 + 1)*X[2]*tmp2 + 3tmp3*(2tmp5*(X[2] - 5/3) - 2tmp4*(X[2] - 1/3)) + 0.8*(X[2] - 1/3)^3 )
end

function overdamped_langevin_noise!(X,dt,σ,rng)
    X[1] += σ*randn(rng)
    X[2] += σ*randn(rng)
end


struct ExitEventKiller end
ParRep.check_death(::ExitEventKiller,state_a,state_b,_) = (state_a != state_b)


Base.@kwdef mutable struct TransitionLogger#{X}
    log_dir
    filenames = (states = "states.int64", transition_times = "transition_times.int64", dephased = "dephased.bool",exit_configurations = "exit_configurations.vec2dfloat64",parallel_ticks="parallel_ticks.int64",dephasing_ticks="dephasing_ticks.int64",initialization_ticks="initialization_ticks.int64")
    file_streams = [open(joinpath(log_dir,f),write=true) for f in filenames]
end

function ParRep.log_state!(logger::TransitionLogger,step; kwargs...)
    if step == :transition
        write(logger.file_streams[1],kwargs[:current_macrostate])
        write(logger.file_streams[2],kwargs[:exit_time]+kwargs[:dephasing_time])
        write(logger.file_streams[3],kwargs[:exit_time] != 0)
        write(logger.file_streams[4],kwargs[:algorithm].reference_walker)
        write(logger.file_streams[5],kwargs[:algorithm].n_parallel_ticks)
        write(logger.file_streams[6],kwargs[:algorithm].n_dephasing_ticks)
        write(logger.file_streams[7],kwargs[:algorithm].n_initialisation_ticks)
    end
end

# logger = AnimationLogger2D()

function main()
    println("Usage: N_replicas β dt overlap m_spectral_gaps n_transitions checkpoint_freq")

    n_replicas = parse(Int64,ARGS[1])
    β = parse(Float64,ARGS[2])
    dt = parse(Float64,ARGS[3])
    α_overlap = parse(Float64,ARGS[4])
    m_sg = parse(Float64,ARGS[5])
    n_transitions = parse(Int64,ARGS[6])
    freq_checkpoint = parse(Int64,ARGS[7])

    log_dir = "logs_overlap_$(β)_$(dt)_$(α_overlap)_$(m_sg)"
    if !isdir(log_dir)
        mkdir(log_dir)
    end

    logger = TransitionLogger(log_dir=log_dir)
    ol_sim = EMSimulator(dt = dt,β = β,drift! = drift_entropic_switch!,diffusion! = overdamped_langevin_noise!,n_steps=1)
    state_check = PolyhedralStateChecker(β,α_overlap)
    harm_check = HarmonicDiagnostic(λ₂=[8.472211868406559,3.7983868797287945,8.472211868406559],m=m_sg,dt=dt)
    # gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->sum(abs,x-minima[:,i])],num_replicas=n_replicas,tol=0.05)


    alg = GenParRepAlgorithm(N=n_replicas,
                            simulator = ol_sim,
                            dephasing_checker = harm_check,
                            macrostate_checker = state_check,
                            replica_killer = ExitEventKiller(),
                            logger = logger,
                            reference_walker = copy(minima[:,1]))



   for k=1:(n_transitions ÷ freq_checkpoint)
        println(k)
        ParRep.simulate!(alg,freq_checkpoint;verbose=true)

        for fstream in logger.file_streams
            flush(fstream)
        end

        alg.n_transitions = 0
    end
end

main()