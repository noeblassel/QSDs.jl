using Base.Threads
using Statistics, Random

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

entropic_switch(q) = entropic_switch(q...)
grad_entropic_switch!(q,grad) =grad_entropic_switch!(q...,grad) 

const minima = [-1.0480549928242202 0.0 1.0480549928242202; -0.042093666306677734 1.5370820044494633 -0.042093666306677734] # 2x3 matrix of local minima coordinates (x_1 | x_2 | x_3)
const saddles = [-0.6172723078764598 0.6172723078764598 0.0; 1.1027345175080963 1.1027345175080963 -0.19999999999972246] # 2x3 matrix of saddle point coordinates (z_{12} | z_{23} | z_{13} )
const neg_eigvecs = [0.6080988038706289 -0.6080988038706289 -1.0; 0.793861351075306 0.793861351075306 0.0]

function get_state(s,q,α)
    l1,l2,l3 =[(q-saddles[:,i])'neg_eigvecs[:,i] for i=1:3]# Fill me in

    if s==1
        (l1 <= α) && (l3 >= -α) && return 1
    elseif s==2
        (l1 >= -α) && (l2 >= -α) && return 2
    elseif s==3
        (l3 <= α) && (l2 <= α) && return 3
    end

    (l1 <= 0) && (l3 >= 0) && return 1
    (l1 >= 0) && (l2 >= 0) && return 2
    (l3 <= 0) && (l2 <= 0) && return 3
end


"""
Arguments:
∇V: gradient of potential energy function implementing a valid method for ∇V(q⁰)
dt: timestep
β: inverse temperature
q⁰: initial point of the trajectory, a d-dimensional vector

Returns:
A (t,q)::Tuple{Float64,Vector{Float64},Int} containing the exit time, the exit configuration and the number of steps of the Markov chain
"""
function sample_exit_dns(∇V!,dt,β,q⁰,state_α)
    σ = sqrt(2dt/β)
    G = zero(q⁰)
    grad = zero(q⁰)
    q = copy(q⁰)
    dims=size(q)
    step = 0

    while get_state(1,q,state_α) == 1
        step +=1
        ∇V!(q,grad)
        randn!(G)
        q .+= -dt .* grad .+ σ .* G
    end

    return (step*dt,q,step)

end

"""
Arguments:
∇V: gradient of potential energy function implementing a valid method for ∇V(q⁰)
dt: timestep
β: inverse temperature
T: simulation time
q⁰: initial point of the trajectory, a d-dimensional vector
N: number of replicas for the Flemming-Viot
gr_α: tolerance parameter for the Gelman-Rubin diagnostic
ϕs: a vector of K functions, such that ϕs[k] implements a method for ϕs[k](q⁰).

Returns:
A tuple (t,q,decorr_steps,parallel_steps)::Tuple{Float64,Vector{Float64},Int,Int} containing the exit time, exit configuration, the number of decorrelation/dephasing steps and the number of parallel exit steps.
"""
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

    decorr = false
    
    # decorrelation/dephasing step
    while !decorr
        decorr_step += 1
        
        @threads for n=1:N # update replicasn
            ∇V!(X[n],grad[n])
            randn!(G[n])
            X[n] .+= -dt .* grad[n] .+ σ .* G[n]
        end

        killed = @. get_state(1,X,state_α) != 1 # compute states

        if killed[1] # failed decorrelation: reference walker has exited
            return (decorr_step*dt,X[1],decorr_step,parallel_step)
        end

        X[killed] .= copy.(rand(X[.! killed],sum(killed))) # Flemmin-Viot branching
        
        # update Gelman-Rubin quantities
        @threads for n=1:N
            for k=1:K
                ϕ = ϕs[k]
                v = ϕ(X[n])
                S[k,n] += v
                Q[k,n] += v^2
            end  
        end

        if maximum((sum(Q[k,n] for n=1:N)-sum(S[k,n] for n=1:N)^2/(N*decorr_step))/(sum(Q[k,n]-S[k,n]^2/decorr_step for n=1:N)) for k=1:K) < 1 + gr_α # Gelman-Rubin diagnostic
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
        end

        killed = @. get_state(1,X,state_α) != 1 # compute states
        
        if any(killed) # succesful exit
            ifirst = argmax(killed)
            return ((decorr_step+N*parallel_step+ifirst)*dt,X[ifirst],decorr_step,parallel_step)
        end

    end
end

n_exits = 1000

times_gpr = Float64[]
times_dns = Float64[]

exits_gpr = Vector{Float64}[]
exits_dns = Vector{Float64}[]

decorr_steps = Int[]
parallel_steps = Int[]
dns_steps = Int[]

ϕs = [q->q[1],q->q[2]]

dt = 5e-3

N = 32
gr_α = 0.1 # Gelman-Rubin tolerance threshold

state_α = 0.0 # state overlap

β = 3.0

for i=1:n_exits
    (i%20 == 0) && println("Sampling $i-th exit")
    # print("gpr: ")
    t,x,ds,ps = sample_exit_genparrep(grad_entropic_switch!,dt,β,q⁰,N,α,ϕs,state_α)
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

println("times_gpr=",times_gpr)
println("exits_gpr=",exits_gpr)
println("decorr_steps=",decorr_steps)
println("parallel_steps=",parallel_steps)
println("times_dns=",times_dns)
println("exits_dns=",exits_dns)
println("dns_steps=",dns_steps)