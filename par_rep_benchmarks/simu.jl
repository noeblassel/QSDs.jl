include("ParRep.jl"); using .ParRep, Base.Threads,Random


# n_transitions = parse(Int64,ARGS[1])
# freq_checkpoint = parse(Int64,ARGS[2])
# n_replicas = parse(Int64,ARGS[3])
# pot_type = parse(ARGS[4])

### Gelman-Rubin convergence test

arg_types = [Float64,Float64,Float64,Int64,Int64,Int64]
β, dt,gr_tol, n_transitions,freq_checkpoint, n_replicas = parse.(arg_types,ARGS)

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

function entropic_switch(X)
    return entropic_switch(X[1,:],X[2,:])
end

"""
    ∇entropic_switch(x, y)

Gradient of the potential energy function.
"""


# grad_entropic_switch(q) = begin X = zeros(2); drift_entropic_switch!(X,-1); return X end


###


### Polyhedral states
const minima = [-1.0480549928242202 0.0 1.0480549928242202; -0.042093666306677734 1.5370820044494633 -0.042093666306677734]
const saddles = [-0.6172723078764598 0.6172723078764598 0.0; 1.1027345175080963 1.1027345175080963 -0.19999999999972246]
const pos_eigvecs = [0.6080988038706289 -0.6080988038706289 -1.0; 0.793861351075306 0.793861351075306 0.0]


struct PolyhedralStateChecker end

function ParRep.get_macrostate!(checker::PolyhedralStateChecker,walker,current_state)
    l1,l2,l3 =[(walker-saddles[:,i])'pos_eigvecs[:,i] for i=1:3]
    (l1 <= 0) && (l3 >= 0) && return 1
    (l1 >= 0) && (l2 >= 0) && return 2
    (l3 <= 0) && (l2 <= 0) && return 3
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

function ParRep.update_microstate!(X,simulator::EMSimulator)
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
    X[2] -=  dt * ( 10*(tmp1 + 1)*X[2]*tmp2 + 3tmp3*(2tmp5*(X[2] - 5/3) - 2tmp4*(X[2] - 1/3)) + 0.8*(X[2] - 1/3)^3 )
end

function overdamped_langevin_noise!(X,dt,σ,rng)
    X[1] += σ*randn(rng)
    X[2] += σ*randn(rng)
end


struct ExitEventKiller end
ParRep.check_death(::ExitEventKiller,state_a,state_b,_) = (state_a != state_b)


Base.@kwdef mutable struct TransitionLogger#{X}
    log_dir
    filenames = (state_from = "state_from.int64",state_to = "state_to.int64", dephasing_time = "transition_time.int64",metastable_exit_time="metastable_exit_time.int64", dephased = "dephased.bool",exit_configuration = "exit_configuration.vec2dfloat64")
    file_streams = [open(joinpath(log_dir,f),write=true) for f in filenames]
end

function ParRep.log_state!(logger::TransitionLogger,step; kwargs...)
    if step == :transition
        write(logger.file_streams[1],kwargs[:current_macrostate])
        write(logger.file_streams[2],kwargs[:new_macrostate])
        write(logger.file_streams[3],kwargs[:dephasing_time])
        write(logger.file_streams[4],kwargs[:exit_time])
        write(logger.file_streams[5],kwargs[:exit_time] != 0)
        write(logger.file_streams[6],kwargs[:algorithm].reference_walker)
    end
end


function main(β,dt,gr_tol,n_transitions,freq_checkpoint,n_replicas)
    log_dir = "logs_$(gr_tol)"

    if !isdir(log_dir)
        mkdir(log_dir)
    end

    logger = TransitionLogger(log_dir=log_dir)
    ol_sim = EMSimulator(dt = dt,β = β,drift! = drift_entropic_switch!,diffusion! = overdamped_langevin_noise!,n_steps=1)
    state_check = PolyhedralStateChecker()
    gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->sum(abs,x-minima[:,i])],num_replicas=n_replicas,tol=gr_tol)


    alg = GenParRepAlgorithm(N=n_replicas,
                            simulator = ol_sim,
                            dephasing_checker = gelman_rubin,
                            macrostate_checker = state_check,
                            replica_killer = ExitEventKiller(),
                            logger = logger,
                            reference_walker = [minima[1,1],minima[1,2]])


    if !isdir(log_dir)
        mkdir(log_dir)
    end

   for k=1:(n_transitions ÷ freq_checkpoint)
        println(k)
        ParRep.simulate!(alg,freq_checkpoint;verbosity=1)

        for fstream in logger.file_streams
            flush(fstream)
        end

        alg.n_transitions = 0
    end
end

main(β,dt,gr_tol,n_transitions,freq_checkpoint,n_replicas)