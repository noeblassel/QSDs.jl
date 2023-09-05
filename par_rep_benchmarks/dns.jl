include("ParRep.jl"); using .ParRep,Random

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
const pos_eigvecs = [0.6080988038706289 -0.6080988038706289 -1.0; 0.793861351075306 0.793861351075306 0.0]


struct PolyhedralStateChecker end

function ParRep.get_macrostate!(checker::PolyhedralStateChecker,walker,current_state)
    l1,l2,l3 =[(walker-saddles[:,i])'pos_eigvecs[:,i] for i=1:3]
    (l1 <= 0) && (l3 >= 0) && return 1
    (l1 >= 0) && (l2 >= 0) && return 2
    (l3 <= 0) && (l2 <= 0) && return 3
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

function main(n_transitions,freq_checkpoint)
    state_check = PolyhedralStateChecker()
    ol_sim = EMSimulator(dt = dt,β = β,drift! = drift_entropic_switch!,diffusion! = overdamped_langevin_noise!,n_steps=1)
    replica_killer = ExitEventKiller()
    
    reference_walker = minima[:,1]
    old_state = ParRep.get_macrostate!(state_check,reference_walker,nothing)
    transition_timer = 0

    state_from = Int64[]
    state_to = Int64[]
    transition_time = Float64[]
    exit_configuration = Vector{Float64}[]
    println(old_state)

    f_handles = [open(joinpath("logs_dns"))]

    @time for k=1:(n_transitions ÷ freq_checkpoint)
        for i=1:freq_checkpoint
            transitioned = false
            while !transitioned
                ParRep.update_microstate!(reference_walker,ol_sim)
                transition_timer += ol_sim.n_steps
                new_state = ParRep.get_macrostate!(state_check,reference_walker,old_state)
                # println(new_state)

                if ParRep.check_death(replica_killer,old_state,new_state,nothing)
                    push!(state_from,old_state)
                    push!(state_to,new_state)
                    push!(transition_time,transition_timer)
                    push!(exit_configuration,copy(reference_walker))

                    old_state = new_state
                    transition_timer = 0

                    println(length(state_from)," /$n_transitions")

                    transitioned = true
                end
            end
        end

        write(open(joinpath("logs_dns","state_from.int64"),"w"),state_from)
        write(open(joinpath("logs_dns","state_to.int64"),"w"),state_to)
        write(open(joinpath("logs_dns","transition_time.int64"),"w"),transition_time)
        write(open(joinpath("logs_dns","is_metastable.bool"),"w"),falses(length(state_from)))
        write(open(joinpath("logs_dns","exit_configuration.vec2f64"),"w"),stack(exit_configuration))

    end

end

main(n_transitions,freq_checkpoint)