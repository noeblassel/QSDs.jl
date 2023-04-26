using LinearAlgebra,Plots, Arpack

base_path="/home/nblassel/Documents/QSDs.jl/"


include(joinpath(base_path,"QSDs.jl"))
include(joinpath(base_path,"SplinePotentials.jl"))

function optimize_domain!(a0, b0, V, β, N, Ω_hist,obj_hist,grad_hist, N_iter, η0)

    a,b=a0,b0
    η = η0

    best_J = Inf

    for i=1:N_iter
        M,B,δM,∂λ=QSDs.build_FEM_matrices_1D_Neumann(V_periodic,β,range(a,b,N+1))
        λs,us = eigs(M[2:N-1,2:N-1],B[2:N-1,2:N-1],nev=2,which=:LR,sigma=0.0)
        λ1,λ2=real.(λs)
        us = real.(us)
        u1 = us[:,1]
        u2 = us[:,2]

        h=(b-a)/N
        
        ∂aJ = inv(β*λ2^2)*(λ2*(first(u1)/h)^2 - λ1*(first(u2)/h)^2) #from shape derivative computations (minus sign accounts for direction of outward normal at a)
        ∂bJ = -inv(β*λ2^2)*(λ2*(last(u1)/h)^2-λ1*(last(u2)/h)^2)
        
        obj = best_obj = (λ2-λ1)/λ1
        J = λ1 / λ2

        if J < best_J
           # println("=== Iteration $i ===")
            best_J = J
            best_obj = (λ2 - λ1)/λ1
            a -= ∂aJ*η
            b -= ∂bJ*η

            #println("\tDomain : ]",a,", ",b,"[")
           # println("\tObjective: ",obj)

            push!(Ω_hist,[a,b])
            push!(obj_hist,best_obj)
            push!(grad_hist,max(abs(∂aJ),abs(∂bJ)))

            #println("\tGradient: ",max(abs(∂aJ),abs(∂bJ)))
        
            η *= 1.05
        
        else
            η /= 2
            (η < 1e-25) && return
        end
        
    end
end

data_dir = "/home/nblassel/Documents/QSDs.jl/results_optim/"
## double well potential
    critical_points = [0.001, 0.25,0.5,0.75,0.999]
    heights = [0.0, -1.0 , 0.0, -1.0, 0.0]

    V,∇V,∇²V = SplinePotentials.spline_potential_derivatives(critical_points,heights,1)
    V_periodic(x)=V(mod(x,1))
    ## 

    as = Float64[]
    bs = Float64[]


    N=1200
    core_set_B = [0.73,0.77]
    core_set_A = [0.23,0.27]

    for (i,β)=enumerate(range(4,6,100))
        println("$i/100, β=$β")
        a,b = 0.5,1.0
        N_iter = 10000000
        η0=0.1
        Ω_hist=Vector{Float64}[]
        obj_hist=Float64[]
        grad_hist=Float64[]
        optimize_domain!(a,b,V_periodic,β,N,Ω_hist,obj_hist,grad_hist,N_iter,η0)
        Ω_max = Ω_hist[argmax(obj_hist)]
        println("Max objective: ", maximum(obj_hist))
        println("Optimal domain: ",a," ",b)
        push!(as,a)
        push!(bs,b)
    
    end