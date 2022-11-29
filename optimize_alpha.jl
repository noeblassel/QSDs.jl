using Cubature,LinearAlgebra,Plots

include("QSDs.jl")
include("optimisation.jl")

#= N,β,Niter,cutoff = ARGS

N,Niter=parse.(Int64,[N,Niter])
β,cutoff=parse.(Float64,[β,cutoff]) =#

N=30
β=8.0
Niter=1000000
cutoff=500.0

V(q) = 2cos(2π*q) - cos(4π*q)
dV(q) = -4π*sin(2π*q)+4π*sin(4π*q)
d2V(q)= -8π^2*cos(2π*q)+16π^2*cos(4π*q)
W_pot(q) = -(0.5d2V(q)-β*dV(q)^2/4)

D=collect(Float64,range(0,1,N+1))
kill_rate=ones(N)

core_set_left=[0.0,0.05]
core_set_middle=[0.45,0.55]
core_set_right=[0.95,1.0]

Δx=D[2]-D[1]
fixed_indices=Int64[]

for (i,x)=enumerate(D[1:end-1])
    c = x +Δx/2
    if (first(core_set_left) <= c <= last(core_set_left)) || (first(core_set_right)<= c <= last(core_set_right))
        push!(fixed_indices,i)
        kill_rate[i] = cutoff
    elseif (first(core_set_middle)<=c <=last(core_set_middle))
        push!(fixed_indices,i)
        kill_rate[i]=0
    end
end

<<<<<<< HEAD
α=SoftKillingRate(kill_rate,D,fixed_indices)
=======
    function optimize_alpha(V,β,α0;weights, grad_tol = 1e-6, max_iter=100000, cutoff=10000.0, lr=0.1, log_every=1000)
        α_star = copy(α0)
        J_hist=Float64[]
        grad_hist=Float64[]
        N=length(α_star)
        for i=1:max_iter
            λs,_,us = SQSD_1D_FEM(V,β,α_star;weights=weights)
            λ1,λ2=λs
            u1,u2=collect.(Float64,eachcol(us))

            ∂λ1= Dλ_Dα(u1,weights=weights)
            ∂λ2 = Dλ_Dα(u2,weights=weights)

            gradient = (λ1 * ∂λ2 - λ2 * ∂λ1) / λ1^2

            (maximum(abs.(gradient)) < grad_tol) && return α_star,J_hist,grad_hist

            α_star -= lr * gradient
            α_star = clamp.(α_star,0.0,cutoff)

            if i%log_every==0
                push!(J_hist,(λ2-λ1)/λ1)
                push!(grad_hist,norm(gradient))
                println("$i / $max_iter")
                flush(stdout)
                f=open("results_optim/results_$(β)_$(N).out","w")
                println(f,α_star)
                println(f,J_hist)
                println(f,grad_hist)
            end

            #println((λ2-λ1)/λ1)
        end
>>>>>>> 2e6276a403415e8692b506dfba4b4a0422fd7751

function pw_constant(q)
    ix=1
    while q>α.domain[ix]
        ix+=1
    end
    return α.values[ix-1]
end

γ=0.001
f(λ1,λ2) = 1/(λ2 - λ1) + γ/λ1
∂1f(λ1,λ2) = 1/(λ2-λ1)^2 - γ/λ1^2
∂2f(λ1,λ2)=-1/(λ2-λ1)^2

g(λ1,λ2) = λ2/λ1
∂1g(λ1,λ2)=-λ2/λ1^2
∂2g(λ1,λ2)=1/λ1

<<<<<<< HEAD
grad_hist,obj_hist,λ1_hist,λ2_hist = optim_gradient_ascent!(W_pot,β,g,∂1g,∂2g,α; max_iterations=Niter, lr=10.0, η=0.9, log_every=1000,grad_tol=1e-7,obj_tol=1e-11,clip=true)
=======
f=open("results_optim/results_$(β)_$(N).out","w")
                println(f,α_star)
                println(f,J_hist)
                println(f,grad_hist)
>>>>>>> 2e6276a403415e8692b506dfba4b4a0422fd7751
