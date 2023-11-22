include("ParRep.jl"); using .ParRep,Random,LinearAlgebra

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
function grad_entropic_switch(x, y)

    tmp1 = exp(4*x)
    tmp2 = exp(-x^2 - 2*x - y^2 - 1)
    tmp3 = exp(-x^2)
    tmp4 = exp(-(y-1/3)^2)
    tmp5 = exp(-(y-5/3)^2)

    dx = 0.8*x^3 + 10*(tmp1*(x - 1) + x + 1)*tmp2 - 6*tmp3*x*(tmp4 - tmp5)

    dy = 10*(tmp1 + 1)*y*tmp2 + 3*tmp3*(2*tmp5*(y - 5/3) - 2*tmp4*(y - 1/3)) + 0.8*(y - 1/3)^3

    return [dx, dy]
end

function grad_entropic_switch(q)
    return grad_entropic_switch(q...)
end


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
###

Base.@kwdef struct EMSimulator{B,Σ,R}
    dt::Float64
    β::Float64
    drift!::B
    diffusion!::Σ
    n_steps=1
    σ = sqrt(2dt/β)
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

if !isdir("logs_dns")
    mkdir("logs_dns")
end

n_transitions = 1000000
freq_checkpoint = 100
dt = 5e-3
β = 4.0

function main()

    println("Usage: β dt overlap n_transitions checkpoint_freq")

    β = parse(Float64,ARGS[1])
    dt = parse(Float64,ARGS[2])
    α_overlap = parse(Float64,ARGS[3])
    n_transitions = parse(Int64,ARGS[4])
    freq_checkpoint = parse(Int64,ARGS[5])

    state_check = PolyhedralStateChecker(β,α_overlap)
    ol_sim = EMSimulator(dt = dt,β = β,drift! = drift_entropic_switch!,diffusion! = overdamped_langevin_noise!,n_steps=1)
    replica_killer = ExitEventKiller()
    
    reference_walker = copy(minima[:,1])
    old_state = ParRep.get_macrostate!(state_check,reference_walker,nothing,1)

    states = Int64[]
    transition_times = Int64[]
    exit_configurations = Vector{Float64}[]

    ## unused
    parallel_ticks = Int64[]
    dephasing_ticks = Int64[]
    initialization_ticks = Int64[]
    dephased = Bool[]
    ##

    log_dir = "logs_dns_$(β)_$(dt)_$(α_overlap)"

    if !isdir(log_dir)
        mkdir(log_dir)
    end

    filenames = (states = "states.int64", transition_times = "transition_times.int64", dephased = "dephased.bool",exit_configurations = "exit_configurations.vec2dfloat64",parallel_ticks="parallel_ticks.int64",dephasing_ticks="dephasing_ticks.int64",initialization_ticks="initialization_ticks.int64")

    f_handles = [open(joinpath(log_dir,f),"w") for f in filenames]

    series = [states,transition_times,dephased,exit_configurations,parallel_ticks,dephasing_ticks,initialization_ticks]
    transition_timer = 0
    # sts = Int64[]

    for k=1:(n_transitions ÷ freq_checkpoint)
        for i=1:freq_checkpoint
            transitioned = false
            while !transitioned
                ParRep.update_microstate!(reference_walker,ol_sim)
                transition_timer += ol_sim.n_steps
                new_state = ParRep.get_macrostate!(state_check,reference_walker,old_state,1)
                # push!(sts,new_state)
                if ParRep.check_death(replica_killer,old_state,new_state,nothing)
                    push!(states,old_state)
                    push!(transition_times,transition_timer)
                    push!(exit_configurations,copy(reference_walker))

                    old_state = new_state
                    transition_timer = 0

                    println((k-1)*freq_checkpoint+length(states)," /$n_transitions")

                    transitioned = true
                end
            end
        end


        for (f,s) in zip(f_handles,series)

            for x in s
                write(f,x)
            end

            empty!(s)

           flush(f)
        end

    end

    # write(open("sts_dns_$(β)_$(α_overlap).jl","w"),sts)
end
main()