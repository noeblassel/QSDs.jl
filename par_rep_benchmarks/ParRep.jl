module ParRep

    export GenParRepAlgorithm

    using Random, Statistics

    mutable struct GenParRepAlgorithm{S,D,P,M,B,L,X,R,K}
        N::Int #number of replicas

        # algorithm parameters
        simulator::S # a method to evolve the microscopic dynamics

        dephasing_checker::P # an object to check if replicas have dephased
        state_checker::M # an object to check the macrostate
        replica_killer::K # an object to kill the replicas

        logger::L # to log characteristics of the trajectory

        # Internal variables

        reference_walker::X
        replicas::Vector{X}

        n_initialisation_ticks::Int # number of simulation steps in initialisation step
        n_dephasing_ticks::Int # number of simulation steps in dephasing step
        n_parallel_ticks::Int # number of (parallel) simulation steps in parallel step

        rng::R
    end

    function GenParRepAlgorithm(;N,simulator,dephasing_checker,state_checker,replica_brancher,logger,reference_walker,rng=GLOBAL_RNG)
        return GenParRepAlgorithm(N,simulator,dephasing_checker,state_checker,replica_brancher,logger,reference_walker,typeof(reference_walker)[],0,0,0,rng)
    end

    function simulate!(alg::GenParRepAlgorithm, simulation_time)
        current_state = get_state(alg.state_checker,alg.reference_walker,nothing)
        killed_ixs = Int[]

        sim_ticks = 0

        while sim_ticks < simulation_time

            ## === INITIALISATION PHASE ===
            initialisation_step = 0

            while current_state === nothing
                update_state!(alg.reference_walker,alg.simulator;rng=alg.rng)
                current_state = get_state(alg.state_checker,current_state,initialization_step)
                initialization_step +=1
            end

            alg.n_initialisation_ticks += initialization_step

            ## === DECORRELATION/DEPHASING === 
            for i=1:alg.N_rep
                push!(alg.replicas,copy(alg.reference_walker))
            end
            
            has_dephased = false
            dephasing_step = 0

            while !has_dephased
                update_state!(alg.reference_walker,alg.simulator;rng=alg.rng)

                alg.cpu_clock_ticks += 1
                alg.wall_clock_ticks += 1
                alg.physical_time += sim.dt

                empty!(killed_ixs)

                # check if reference_walker has escaped
                is_killed = check_death(alg.replica_killer,dephasing_step,alg.reference_walker,current_state,alg.rng)
                (is_killed) && break

                for i=1:alg.N # to parallelize in production
                    update_state!(alg.replicas[i],alg.simulator;rng=alg.rng)
                    check_death(alg.replica_killer,dephasing_step,alg.replicas[i],current_state,alg.rng) && push!(i,killed_ixs)
                end

                dephasing_ticks += 1

                survivors = setdiff(1:alg.N_rep,killed_ixs)

                if length(survivors)==0
                    ErrorException("Extinction event! All replicas have been killed")
                end
                
                
                for i=killed_ixs
                    alg.replicas[i] = alg.replicas[rand(alg.rng,survivors)]
                end
                
                has_dephased = check_dephasing(alg.dephasing_checker,alg.replicas,dephasing_step)

            end

            alg.physical_time += alg.dt * dephasing_step

            if has_dephased
                ## === PARALLEL PHASE ===
                killed = false
                parallel_step = 0

                while !killed
                    for i=1:alg.N_rep
                        update_state!(alg.replicas[i],alg.simulator;rng=alg.rng)
                        alg.cpu_clock_ticks += 1
                        
                        if check_death(alg.replica_killer,alg.replicas[i],current_state,alg.rng)
                            killed = true
                            alg.reference_walker = branch_replica!(alg.replica_brancher,alg.replicas[i],alg.reference_walker) # set reference walker to escaped replica
                            escape_time = (alg.N_rep * parallel_step +i)
                            empty!(alg.replicas)
                            break
                        end
                    end

                    parallel_step += 1
                    alg.wall_clock_ticks += 1
                end

                sim_ticks += escape_time
            end
        end
    end

end