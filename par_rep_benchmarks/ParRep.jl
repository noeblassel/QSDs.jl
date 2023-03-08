module ParRep

export Replica,
    ParRepAlgorithm

using Random

    mutable struct Replica{X,H}
        state::X
        gr_history::H # for equilibrium diagnostic
    end

    mutable struct ParRepAlgorithm{X,T,H,K,E,R,L,U,S,B}

        # hyperparameters
        N_rep::Uint32

        state_check_freq::Uint32 # frequency of state determination
        eq_check_freq::Uint32 # frequency of equilibriation diagnostic
        killing_check_freq::Uint32 # frequency of killing criterion check

        # algorithm parameters
        dt::T # simulation timestep
        rng::R
        equilibrium_diagnostic::E
        killing_criterion::K
        update_microstate!::U
        get_macrostate::S # a method to compute the macrostate. (Should return Uint32 or nothing)
        branch_replica!::B # a method to branch replicas
        logger::P # a logger for diagnoses and recording of exit events / state to state dynamics

        # Internal variables

        reference_walker::Replica{X,H}
        replicas::Vector{Replica{X,H}}

        cpu_clock_ticks::Uint64
        wall_clock_ticks::Uint64
        physical_time::T

        macrostate::Union{Nothing,Uint32}
    end


    """
    A Generalized Parallel Dynamics algorithm. Keyword arguments:
        - N_replicas : the number of replicas of the system
        - state_check_freq = 1 : the frequency at which the state of each replica is resolved
        - eq_check_freq = 1 : the frequency at which convergence to local equilibrium is assessed
        - killing_check_freq = 1 : the frequency at which the occurence of an exit event is checked
        - dt : the timestep of the simulation
        - rng = Random.GLOBAL_RNG : the pseudo-random number generator
        - equilibrium_diagnostic : A function with a method equilibrium_diagnostic(replicas::Vector{Replica{X,H}},dephasing_clock::T) returning true or false
        - killing_criterion : A function with a method killing_criterion(replica::Replica{X,H},current_state::X,dt::T,rng) returning true or false
        - update_microstate! : A function with methods update_microstate!(replica::Replica{X,H},dt::T,rng,gr_log_step::Bool)
        - get_macrostate : A function with a method get_macrostate(replica::Replica{X,H},current_state)::Union{Nothing,Uint32}
        - branch_replica! : A function with methods branch_replica!(source::Replica{X,H},dest::Replica{X,H},copy_hist::Bool)
        - logger : A logger (todo)
        - reference_walker : The initial state of the system represented by a Replica::{X,T,H}.
    """
    function ParRepAlgorithm(;N_replicas,
        state_check_freq=1,
        eq_check_freq=1,
        killing_check_freq=1,
        dt,
        rng=Random.GLOBAL_RNG,
        equilibrium_diagnostic,
        killing_criterion,
        update_microstate!,
        get_macrostate,
        branch_replica!,
        logger,
        reference_walker)

        return ParRepAlgorithm(N_replicas,state_check_freq,eq_check_freq,killing_check_freq,dt,rng,equilibrium_diagnostic,killing_criterion,update_microstate!,get_macrostate,branch_replica!,logger,reference_walker,typeof(reference_walker)[],0,0,zero(dt),nothing)
    end

    function simulate!(alg::ParRepAlgorithm, simulation_time::T) where {T}
        initial_state = alg.get_macrostate(alg.reference_walker,initial_state)
        killed_ixs = Uint32[]
        while alg.physical_clock <= simulation_time

            ## === INITIALISATION PHASE ===
            initialization_step = 0

            while initial_state === nothing
                alg.update_microstate!(alg.reference_walker,alg.dt,alg.rng,false)

                alg.cpu_clock_ticks += 1
                alg.wall_clock_ticks += 1

                alg.physical_time += alg.dt

                if initialization_step % alg.state_check_freq == 0
                    initial_state = alg.get_macrostate(alg.reference_walker,initial_state)
                end

                initialization_step += 1
            end

            ## === DEPHASING/DECORRELATION PHASE ===
            for i=1:alg.N_rep
                replica = branch_replica!(alg.reference_walker,false)
                push!(replica,alg.replicas)
            end
            
            has_dephased = false

            dephasing_step = 0

            while !has_dephased
                alg.update_microstate!(alg.reference_walker,alg.dt,alg.rng,false)

                alg.cpu_clock_ticks += 1
                alg.wall_clock_ticks += 1
                alg.physical_time += sim.dt

                empty!(killed_ixs)

                # check if reference_walker has escaped

                if dephasing_step % alg.killing_check_freq == 0
                    is_killed = alg.killing_criterion(alg.reference_walker,initial_state,alg.dt,alg.rng)
                    (is_killed) && break # reference walker escaped before succesful decorrelation
                end

                for i=1:alg.N_rep # to parallelize in production
                    alg.update_microstate!(alg.replicas[i],alg.dt,alg.rng, dephasing_step % alg.eq_check_freq == 0) # store gr_observables if this is an equilibriation check step
                    alg.cpu_clock_ticks += 1

                    if (dephasing_step % alg.kiling_check_freq ==0) && (alg.killing_criterion(alg.replicas[i],initial_state,alg.dt,alg.rng))
                        push!(killed_ixs,i)
                     end
                end

                alg.wall_clock_ticks += 1

                survivors = setdiff(1:alg.N_rep,killed_ixs)

                if length(survivors)==0
                    println("Error: total extinction event.")
                    return 1
                end

                # branch killed replicas
                for i=killed_ixs
                    branch_replica!(alg.replicas[rand(alg.rng,survivors)],alg.replicas[i],true)
                end

                if (dephasing_step % alg.eq_check_freq == 0)
                    has_dephased = alg.equilibrium_diagnostic(alg.replicas,dephasing_step * alg.dt)
                end

            end

            alg.physical_time += alg.dt * dephasing_step

            if has_dephased
                ## === PARALLEL PHASE ===
                killed = false
                parallel_step = 0

                while !killed
                    for i=1:alg.N_rep
                        alg.update_microstate!(alg.replicas[i],alg.dt,alg.rng,false)
                        alg.cpu_clock_ticks += 1
                        
                        if (parallel_step % alg.killing_check_freq == 0) && (alg.killing_criterion(alg.replicas[i],initial_state,alg.dt,alg.rng))
                            killed = true
                            alg.reference_walker = branch_replica!(alg.replicas[i],alg.reference_walker,false) # set reference walker to escaped replica
                            escape_time = (alg.N_rep * parallel_step +i)*sim.dt
                            break
                        end
                    end

                    parallel_step += 1
                    alg.wall_clock_ticks += 1
                end

                alg.physical_clock += escape_time
            end
        end
    end
end