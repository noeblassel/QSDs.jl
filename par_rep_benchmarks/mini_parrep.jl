##################################################################
#           Mini GenParRep
# Author: Noé Blassel
# Date: September 2023
# Description: A minimalist (but still multi-threaded) implementation of GenParRep, applied to the exit problem for the three-well entropic switch potential.
# The aim is to assess the impact of the metastable state definition on maximal parallel efficiency of the algorithm. (in terms of wall-clock time gain)
##################################################################


using Base.Threads
using Statistics, Random

T = [Int,Float64,Int,Float64,Float64]
n_exits,β,N,gr_α,state_α = parse.(T,ARGS[1:end-1])
of = last(ARGS)

##

function entropic_switch(x, y)
    tmp1 = x^2
    tmp2 = (y - 1 / 3)^2
    return 3 * exp(-tmp1) * (exp(-tmp2) - exp(-(y - 5 / 3)^2)) - 5 * exp(-y^2) * (exp(-(x - 1)^2) + exp(-(x + 1)^2)) + 0.2 * tmp1^2 + 0.2 * tmp2^2
end

function grad_entropic_switch!(x,y,grad)

    tmp1 = exp(4*x)
    tmp2 = exp(-x^2 - 2*x - y^2 - 1)
    tmp3 = exp(-x^2)
    tmp4 = exp(-(y-1/3)^2)
    tmp5 = exp(-(y-5/3)^2)

    grad[1] = 0.8*x^3 + 10*(tmp1*(x - 1) + x + 1)*tmp2 - 6*tmp3*x*(tmp4 - tmp5)
    grad[2] = 10*(tmp1 + 1)*y*tmp2 + 3*tmp3*(2*tmp5*(y - 5/3) - 2*tmp4*(y - 1/3)) + 0.8*(y - 1/3)^3

    return nothing
end

entropic_switch(q) = entropic_switch(q[1],q[2])
grad_entropic_switch!(q,grad) =grad_entropic_switch!(q[1],q[2],grad) 

const minima = [-1.0480549928242202 0.0 1.0480549928242202; -0.042093666306677734 1.5370820044494633 -0.042093666306677734] # 2x3 matrix of local minima coordinates (x_1 | x_2 | x_3)
const saddles = [-0.6172723078764598 0.6172723078764598 0.0; 1.1027345175080963 1.1027345175080963 -0.19999999999972246] # 2x3 matrix of saddle point coordinates (z_{12} | z_{23} | z_{13} )
const neg_eigvecs = [0.6080988038706289 -0.6080988038706289 -1.0; 0.793861351075306 0.793861351075306 0.0]

"""
overlapping states:
- s::Int, current state
- q::Vector{Float64} configuration
- α::Float64 fattening parameter
- l:: Vector{Float64}: a 3x1 vector to store scalar products
"""
function get_state!(s,q,α,l)
    for i=1:3 # compute scalar products
        l[i] = (q[1]-saddles[1,i])*neg_eigvecs[1,i] + (q[2]-saddles[2,i])*neg_eigvecs[2,i]
    end

    if s==1
        (l[1] <= α) && (l[3] >= -α) && return 1
    elseif s==2
        (l[1] >= -α) && (l[2] >= -α) && return 2
    elseif s==3
        (l[3] <= α) && (l[2] <= α) && return 3
    end

    (l[1] <= 0) && (l[3] >= 0) && return 1
    (l[1] >= 0) && (l[2] >= 0) && return 2
    (l[3] <= 0) && (l[2] <= 0) && return 3
end

function sample_exit_dns(∇V!,dt,β,q⁰,state_α)
    σ = sqrt(2dt/β)
    G = zero(q⁰)
    grad = zero(q⁰)
    q = copy(q⁰)
    dims=size(q)
    step = 0
    l = zeros(3)

    while get_state!(1,q,state_α,l) == 1
        step +=1
        ∇V!(q,grad)
        randn!(G)
        q .+= -dt .* grad .+ σ .* G
    end

    return (step*dt,q,step)

end

function sample_exit_genparrep(∇V!,dt,β,q⁰,N,gr_α,ϕs,state_α)
    σ = sqrt(2dt/β)
    q = copy(q⁰)
    dims=size(q)

    decorr_step = 0
    parallel_step = 0

    X = [copy(q) for i=1:N]
    G = [zero(q) for i=1:N]
    grad = [zero(q) for i=1:N]

    K = length(ϕs)

    S = zeros(K,N)
    Q = zeros(K,N)

    ls = [zeros(3) for i=1:N]

    killed = falses(N)

    decorr = false
    
    # decorrelation/dephasing step
    while !decorr
        decorr_step += 1
        
        @threads for n=1:N # update replicas 
            ∇V!(X[n],grad[n])
            randn!(G[n])
            X[n] .+= -dt .* grad[n] .+ σ .* G[n]
            killed[n] = get_state!(1,X[n],state_α,ls[n]) != 1 # compute states
        end

        if killed[1] # failed decorrelation: reference walker has exited
            return (decorr_step*dt,X[1],decorr_step,parallel_step)
        end

        X[killed] .= copy.(rand(X[.! killed],sum(killed))) # Flemming-Viot branching
        
        # update Gelman-Rubin quantities
        @threads for n=1:N  
            for k=1:K
                ϕ = ϕs[k]
                v = ϕ(X[n])
                S[k,n] += v
                Q[k,n] += v^2
            end  
        end

        M = -Inf

        for k=1:K
            M = max(M,@views (sum(Q[k,:])-sum(S[k,:])^2/(N*decorr_step))/(sum(Q[k,:])-sum(abs2,S[k,:])/decorr_step))
            (M >= 1 + gr_α) && break
        end

        if M < 1 + gr_α
            decorr = true
        end

    end

    # parallel step

    while true
        parallel_step += 1

        @threads for n=1:N # update replicas 
            ∇V!(X[n],grad[n])
            randn!(G[n])
            X[n] .+= -dt .* grad[n] .+ σ .* G[n]
            killed[n] = get_state!(1,X[n],state_α,ls[n]) != 1 # compute state
        end
        
        if any(killed) # succesful exit
            ifirst = argmax(killed)
            return ((decorr_step+N*parallel_step+ifirst)*dt,X[ifirst],decorr_step,parallel_step)
        end

    end
end

const times_gpr = Float64[]
const times_dns = Float64[]

const exits_gpr = Vector{Float64}[]
const exits_dns = Vector{Float64}[]

const decorr_steps = Int[]
const parallel_steps = Int[]
const dns_steps = Int[]


const q⁰ = [-1.04805,-0.0420936]
const ϕs = [q->q[1],q->q[2]]

const dt = 5e-3

for i=1:n_exits
    (i%20 == 0) && println("Sampling $i-th exit")
    # print("gpr: ")
    t,x,ds,ps = sample_exit_genparrep(grad_entropic_switch!,dt,β,q⁰,N,gr_α,ϕs,state_α)
    push!(times_gpr,t)
    push!(exits_gpr,x)
    push!(decorr_steps,ds)
    push!(parallel_steps,ps)
    # print("dns :")
    t,x,n = sample_exit_dns(grad_entropic_switch!,dt,β,q⁰,state_α)
    push!(times_dns,t)
    push!(exits_dns,x)
    push!(dns_steps,n)
end

f = open(of,"w")

println(f,"times_gpr=",times_gpr)
println(f,"exits_gpr=",exits_gpr)
println(f,"decorr_steps=",decorr_steps)
println(f,"parallel_steps=",parallel_steps)
println(f,"times_dns=",times_dns)
println(f,"exits_dns=",exits_dns)
println(f,"dns_steps=",dns_steps)