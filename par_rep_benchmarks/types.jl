module ParRep

export 

begin

    mutable struct Replica{X,T,H,K}
        state::X
        clock::T
        history::H
        killing_criterion::K
    end

    mutable struct ParRepAlgorithm{X,T,H,K,M,E,R,L,U,S,B,F}
        N_rep :: Int64
        reference_walker::Replica{X,T,H,K}
        replicas::Vector{Replica{X,T,H,K}}
        macrostate::M

        rng::R

        dt::T
        cpu_clock::T
        wall_clock::T
        physical_clock::T

        equilibrium_diagnostic::E
        logger::P

        update_microstate!::U
        get_macrostate!::S
        branch_replica!::B
        set_reference_walker!::F

    end

    function simulate!(alg::ParRepAlgorithm, simulation_time::T) where {T}
        initial_state = alg.get_macrostate!(alg.reference_walker,initial_state)

        while alg.physical_clock <= simulation_time
            # initialization step
            while initial_state === nothing
                alg.update_microstate!(alg.reference_walker,alg.dt,alg.rng)

                alg.cpu_clock += alg.dt
                alg.physical_clock += alg.dt
                alg.wall_clock += alg.dt

                initial_state = alg.get_macrostate!(alg.reference_walker,initial_state)
            end

            # dephasing / decorrelation step
            for i=1:alg.N_rep
                replica = branch_replica!(alg.reference_walker)
                push!(replica,alg.replicas)
            end
            
            dephasing_clock = zero(T)
            has_dephased = false

            while !has_dephased
                alg.update_microstate!(alg.reference_walker,alg.dt,alg.rng)
                alg.cpu_clock += alg.dt
                alg.wall_clock += alg.dt
                dephasing_clock += alg.dt

                is_killed = alg.reference_walker.killing_criterion(alg.reference_walker,initial_state,alg.dt,alg.rng)

                (is_killed) && break

                killed_ixs = Int64[]

                for i=1:alg.N_rep
                    alg.update_microstate!(alg.replicas[i],alg.dt,alg.rng)
                    alg.cpu_clock += alg.dt

                    if alg.replicas[i].killing_criterion(alg.replicas[i],initial_state,alg.dt,alg.rng)
                        push!(killed_ixs,i)
                    end
                end

                survivors = setdiff(1:alg.N_rep,killed_ixs)

                if length(survivors)==0
                    println("Error: all replicas have been killed during dephasing.")
                    return 1
                end

                # branch killed replicas
                for i=killed_ixs
                    alg.replicas[i]=branch_replica!(alg.replicas[rand(alg.rng,survivors)])
                end

                has_dephased = alg.equilibrium_diagnostic(alg.replicas,dephasing_clock) # to figure out
            end

            escape_time = zero(T)
            if has_dephased
                # parallel step
                killed = false
                while !killed
                    escape_time += alg.N_rep*alg.dt
                    for i=1:alg.N_rep
                        alg.update_microstate!(alg.replicas[i],alg.dt,alg.rng)
                        alg.cpu_clock += alg.dt

                        if alg.replicas[i].killing_criterion(alg.replicas[i],initial_state,alg.dt,alg.rng)
                            killed = true
                            alg.reference_walker = alg.set_reference_walker!(alg.replicas[i])
                            break
                        end
                    end
                end

                alg.physical_clock += escape_time
            end
        end
    end
end