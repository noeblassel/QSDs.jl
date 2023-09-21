module ParRep

    export GenParRepAlgorithm

    using Random, Base.Threads, Logging

    Base.@kwdef mutable struct GenParRepAlgorithm{S,P,M,K,R,X,L}
        N::Int #number of replicas

        # algorithm parameters
        simulator::S # a method to evolve the microscopic dynamics

        dephasing_checker::P # an object to check if replicas have dephased
        macrostate_checker::M # an object to check the macrostate
        replica_killer::K # an object to kill the replicas

        logger::L # log characteristics of the trajectory / debug 

        # Internal variables

        rng::R = Random.default_rng()
        reference_walker::X
        replicas::Vector{X} = typeof(reference_walker)[]

        n_initialisation_ticks::Int=0 # number of calls to simulator in the initialisation step
        n_dephasing_ticks::Int=0 # number of (parallel) calls to the simulator in the dephasing/decorrelation step
        n_parallel_ticks::Int=0 # number of (parallel) simulation steps in parallel step
        n_transitions::Int=0 # number of observed transitions
        simulation_time::Int=0 # simulation time -- in number of equivalent steps of the DNS Markov chain
        wallclock_time::Int=0
    end

    @inbounds function simulate!(alg::GenParRepAlgorithm, n_transitions;verbose=false)
        current_macrostate = get_macrostate!(alg.macrostate_checker,alg.reference_walker,nothing,0)
        survived = fill(true,alg.N)

        while length(alg.replicas) < alg.N# initialize list of replicas
            push!(alg.replicas,copy(alg.reference_walker))
        end

        while alg.n_transitions < n_transitions
            ## === INITIALISATION PHASE ===
            initialization_step = 0
            while current_macrostate === nothing
                update_microstate!(alg.simulator,alg.reference_walker)
                alg.n_simulation_ticks += 1
                current_macrostate = get_macrostate!(alg.macrostate_checker,alg.reference_walker,current_macrostate,initialization_step)
                log_state!(alg.logger,:initialisation; algorithm = alg)
                initialization_step +=1
            end
           (verbose) && @info "Initialized in state $(current_macrostate)."

            alg.n_initialisation_ticks += initialization_step
            alg.simulation_time += initialization_step
            alg.wallclock_time += initialization_step

            ## === DECORRELATION/DEPHASING === 
            @threads for i=1:alg.N # add @threads
                alg.replicas[i] = deepcopy(alg.reference_walker)
            end
            
            has_dephased = false
            dephasing_step = 0

            while !has_dephased
                update_microstate!(alg.simulator,alg.reference_walker)
                ref_macrostate = get_macrostate!(alg.macrostate_checker,alg.reference_walker,current_macrostate,dephasing_step)
                # check if reference_walker has escaped
                if check_death(alg.replica_killer,ref_macrostate,current_macrostate,alg.rng)
                    log_state!(alg.logger,:transition; algorithm = alg,current_macrostate=current_macrostate,new_macrostate=ref_macrostate,exit_time=0,dephasing_time=dephasing_step)
                     current_macrostate = ref_macrostate
                     (verbose) && @info "Unsuccesful dephasing: reference walker has crossed to state $(current_macrostate) after $(dephasing_step) steps."
                    alg.n_transitions += 1
                    break
                end


                @threads for i=1:alg.N # add @threads
                    update_microstate!(alg.simulator,alg.replicas[i])
                    rep_macrostate = get_macrostate!(alg.macrostate_checker,alg.replicas[i],current_macrostate,dephasing_step)

                    if check_death(alg.replica_killer,rep_macrostate,current_macrostate,alg.rng)
                        survived[i] = false
                        (verbose) && @info "Branching replica $i which crossed to state $(rep_macrostate)"
                    end

                end

                log_state!(alg.logger,:dephasing; algorithm = alg)
                
                if !any(survived)
                    throw(DomainError("Extinction event!"))
                end
                
                survivors = (1:alg.N)[survived]

                for i=1:alg.N #ADD @threads
                    !survived[i] && (alg.replicas[i] .= alg.replicas[rand(alg.rng,survivors)])
                end
                
                fill!(survived,true)
                has_dephased = check_dephasing!(alg.dephasing_checker,alg.replicas,current_macrostate,dephasing_step)

                dephasing_step += 1
            end

            alg.n_dephasing_ticks += dephasing_step
            alg.simulation_time += dephasing_step
            alg.wallclock_time += dephasing_step

            if has_dephased
                (verbose) && @info "Successful dephasing in state $current_macrostate after $(dephasing_step) steps."
                ## === PARALLEL PHASE === 
                i_min = nothing
                new_macrostate = fill(current_macrostate,alg.N)
                parallel_step = 0
                
                fill!(survived,true)

                while all(survived)
                    @threads for i=1:alg.N ## ADD @threads
                        update_microstate!(alg.simulator,alg.replicas[i])
                        rep_macrostate = get_macrostate!(alg.macrostate_checker,alg.replicas[i],current_macrostate,parallel_step)
                        if check_death(alg.replica_killer,rep_macrostate,current_macrostate,alg.rng)                            
                            new_macrostate[i] = rep_macrostate
                            survived[i] = false
                            (verbose) && @info "Replica $i crossed to state $(rep_macrostate) after $(parallel_step+1) parallel steps"
                            break
                        end
                    end

                    log_state!(alg.logger,:parallel; algorithm = alg)
                    parallel_step += 1
                end

                i_min = argmin(survived)
                
                alg.reference_walker .= alg.replicas[i_min]
                exit_time = (alg.N*parallel_step + i_min)

                log_state!(alg.logger,:transition; algorithm = alg,current_macrostate=current_macrostate,new_macrostate=new_macrostate[i_min],exit_time=exit_time,dephasing_time=dephasing_step)
                
                alg.n_parallel_ticks += parallel_step
                alg.wallclock_time += parallel_step
                alg.n_transitions += 1
                alg.simulation_time += exit_time
                current_macrostate = new_macrostate[i_min]
            end
        end
    end


    # dummy method definitions

    function check_dephasing!(checker,replicas,current_macrostate,step_n) end
    function get_macrostate!(checker,walker,current_macrostate,step_n) end
    function update_microstate!(simulator,walker) end
    function check_death(checker,macrostate_a,macrostate_b,rng) end
    function log_state!(logger; kwargs...) end
end