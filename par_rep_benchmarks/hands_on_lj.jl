include("ParRep.jl"); using .ParRep, Base.Threads,Random


# n_transitions = parse(Int64,ARGS[1])
# freq_checkpoint = parse(Int64,ARGS[2])
# n_replicas = parse(Int64,ARGS[3])
# pot_type = parse(ARGS[4])

### Gelman-Rubin convergence test

arg_types = [Float64,Float64,Float64,Int64,Int64,Int64,Int64,Int64]
β, dt,gr_tol,state_check_freq, n_transitions,freq_checkpoint, n_replicas,N_cluster = parse.(arg_types,ARGS)


## dephasing diagnostic

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

## lj potential


Base.@kwdef struct LJClusterInteraction2D{N}
    σ=1.0
    σ6=σ^6
    σ12=σ6^2
    ε=1.0
    α=1.0 # sharpness of harmonic confining potential
     # for multithread force computation
end

function lj_energy(X,inter::LJClusterInteraction2D{N}) where {N}
    V = 0.0
    for i=1:N-1
        for j=2:N
            inv_r6=inv(sum(abs2,X[:,i]-X[:,j]))^3
            V += (inter.σ12*inv_r6^2-inter.σ6*inv_r6)
        end
    end
    return 4inter.ε*V+inter.α*sum(abs2,X)/2 # add confining potential
end

function lj_grad(X,inter::LJClusterInteraction2D{N}) where {N}
    F_threaded = fill(zeros(2,N),nthreads())
    @threads for i=1:N-1
        r = zeros(2)
        f = zeros(2)
        for j=i+1:N
            r = X[:,i]-X[:,j]
            inv_r2 = inv(sum(abs2,r))
            inv_r4 = inv_r2^2
            inv_r8 = inv_r4^2
            
            f = (6inter.σ6 - 12inter.σ12*inv_r2*inv_r4)*inv_r8*r

            F_threaded[threadid()][:,i] .+= f
            F_threaded[threadid()][:,j] .-= f
        end
    end

    return 4inter.ε*sum(F_threaded) + inter.α*X #add confining potential
end

## simulator

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

###

### Replica Killer

struct ExitEventKiller end
ParRep.check_death(::ExitEventKiller,state_a,state_b,_) = (state_a != state_b)


### Logger

Base.@kwdef mutable struct TransitionLogger
    log_dir
    filenames = (state_from = "state_from.int64",state_to = "state_to.int64", dephasing_time = "transition_time.int64",metastable_exit_time="metastable_exit_time.int64", dephased = "dephased.bool",exit_configuration = "exit_configuration.vec$(2N_cluster)Dfloat64",parallel_ticks="parallel_ticks.int64",dephasing_ticks="dephasing_ticks.int64")
    file_streams = [open(joinpath(log_dir,f),write=true) for f in filenames]
    transition_count = 0
    flush_freq = 100
end

function ParRep.log_state!(logger::TransitionLogger,step; kwargs...)
    if step == :transition
        write(logger.file_streams[1],kwargs[:current_macrostate])
        write(logger.file_streams[2],kwargs[:new_macrostate])
        write(logger.file_streams[3],kwargs[:dephasing_time])
        write(logger.file_streams[4],kwargs[:exit_time])
        write(logger.file_streams[5],kwargs[:exit_time] != 0)
        write(logger.file_streams[6],kwargs[:algorithm].reference_walker)
        write(logger.file_streams[7],kwargs[:algorithm].n_parallel_ticks)
        write(logger.file_streams[8],kwargs[:algorithm].n_dephasing_ticks)
        logger.transition_count +=1
    end
    
    (logger.transition_freq % flush_freq ==0) && flush.(logger.file_streams)

end

### state check

Base.@kwdef mutable struct SteepestDescentStateSym{X,E,F} # determine state by steepest descent (case of symetric energy landscape) 
    η = 5e-4
    grad_tol = 1e-2
    e_tol = 1e-1
    minima = X[]
    minimums = Float64[]
    V::E
    ∇V::F
end

function ParRep.get_macrostate!(checker::SteepestDescentStateSym,walker,current_macrostate)
    x = copy(walker)
    grad_norm = Inf

    while (grad_norm > checker.grad_tol) #for k=1:checker.steps
        grad = checker.∇V(x)
        x .-= checker.η * grad # gradient descent step
        grad_norm = maximum(abs.(grad))
    end
    
    if grad_norm < checker.grad_tol
        
        pot = checker.V(x)

        if isempty(checker.minimums)

            push!(checker.minima,x)
            push!(checker.minimums,pot)

            return 1
        else
            ΔEs = @. abs(pot - checker.minimums)
            imin = argmin(ΔEs)

            if ΔEs[imin] < checker.e_tol
                return imin
            else
                push!(checker.minima,x)
                push!(checker.minimums,pot)

                return length(checker.minima)
            end 
        end
    else # steepest descent has not converged in alloted number of iterations
        return nothing
    end
end

## main function

function main(β,dt,gr_tol,state_check_freq,n_transitions,freq_checkpoint,n_replicas)
    log_dir = "logs_$(gr_tol)"

    if !isdir(log_dir)
        mkdir(log_dir)
    end

    N_cluster = 7
    inter = LJClusterInteraction2D{N_cluster}()

    pot(X) = lj_grad(X,inter)
    gradpot(X) = lj_grad(X,inter)

    function drift_lj!(X,dt)
        X .-= gradpot(X)*dt
    end

    function overdamped_langevin_noise!(X,dt,σ,rng)
        X .+= σ*randn(rng,size(X)...)
    end

    X = zeros(2,N_cluster)

    k = ceil(Int,sqrt(N_cluster))

    for i=1:N_cluster
        X[1,i] = i ÷ k
        X[2,i] = i % k
    end

    X .-= [sum(X[1,:]);sum(X[2,:])]/N_cluster

    logger = TransitionLogger(log_dir=log_dir,flush_freq=freq_checkpoint)
    ol_sim = EMSimulator(dt = dt,β = β,drift! = drift_lj!,diffusion! = overdamped_langevin_noise!,n_steps=state_check_freq)
    state_check = SteepestDescentStateSym{Matrix{Float64},typeof(pot),typeof(gradpot)}(V=pot,∇V=gradpot,η=5e-3,grad_tol=2e-2,e_tol=5e-2)
    gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->sum(abs2,x-state_check.minima[i])],num_replicas=n_replicas,tol=gr_tol)


    alg = GenParRepAlgorithm(N=n_replicas,
                            simulator = ol_sim,
                            dephasing_checker = gelman_rubin,
                            macrostate_checker = state_check,
                            replica_killer = ExitEventKiller(),
                            logger = logger,
                            reference_walker = X)


    if !isdir(log_dir)
        mkdir(log_dir)
    end

    ParRep.simulate!(alg,n_transitions;verbose=true)

end

main(β,dt,gr_tol,state_check_freq,n_transitions,freq_checkpoint,n_replicas)