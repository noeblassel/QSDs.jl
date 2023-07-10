include("ParRep.jl")

using .ParRep
using Random

### potential and force definition ###
    V(x,y)= cos(2π*x)-cos(2π*(x-y))
    V(X) = V(X...)

    function ∇V(x,y)
        sin_xmy = sin(2π*(x-y))
        return -2π*[sin(2π*x)-sin_xmy,sin_xmy]
    end
    
    ∇V(X) = ∇V(X...)
### Algorithm specification

rseed = 2023
β=4.0
T_rep = Replica{Vector{Float64},Nothing}
dt=1e-3

n_transition = ceil(Int64,exp(β*2)/dt)
println("n transition ≈ $(n_transition)")

x = [-0.5, 0.5]

state_traj = Int64[]

function update_microstate!(rep, dt, σ = sqrt(2dt/β), rng = Xoshiro(rseed))
    x = rep.state
    x .+= σ*randn(rng,2) - ∇V(x)*dt
    x .= @. mod(x+1,2)-1 # reproj
    push!(state_traj,get_state(x))
end

rng = Xoshiro(rseed)

walker = T_rep(x,nothing)
qrange=range(-1,1,50)

using Plots

σ = sqrt(2dt/β)

sim(n::Int) = (for i=1:n update_microstate!(walker,dt,σ,rng) end)
k= 10

r_coreset = 0.3
minima = [[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]]

function get_state(x)
    η = 1/8π
    for i=1:50
        x -= η*∇V(x)
    end
    return argmin([sum(abs2,x-m) for m=minima])
end 


anim= @animate for i=1:100 * k
    (i%10 == 0) && println(i)
    x = walker.state
    heatmap(qrange,qrange,V)    
    scatter!([first(x)],[last(x)],color=:green)
    sim(1000)
end

mp4(anim,"test_2D.mp4")
