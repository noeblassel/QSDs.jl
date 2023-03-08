include("ParRep.jl")
include("../SplinePotentials.jl")

using .ParRep, .SplinePotentials
using Random

### potential definition ###
    critical_points = [0.001, 0.25,0.5,0.75,0.999]
    heights = [0.0, -1.5 , 0.0, -2.0, 0.0]

    V,∇V,∇²V = SplinePotentials.spline_potential_derivatives(critical_points,heights,1)
    V_periodic(x)=V(mod(x,1))

    β = 7.0
    Ωs = [(0.0,0.5),(0.5,1.0)] 

### Algorithm specification

rseed = 1234

T_rep = Replica{Float64,Tuple{Vector{Float64},Vector{Float64}}}

function update_microstate!(walker::T_rep, dt, rng = Xoshiro(rseed),gr_log_)
    walker.state -= dt*∇V(walker.state)
    walker.state += sqrt(2dt/β)*randn(rng)

    walker.clock += dt
    push!(walker.history,walker.state)
end

function get_macrostate(walker::T_rep,state) # case of a partition
    x = mod(walker.state,1.0)
    for i=Cint[1,2]
        a,b = Ωs[i]
        if a<= x <= b
            return i
        end
    end
end

gr_tol = 1e-5

function equilibrium_diagnostic(replicas::Vector{T_rep},)


