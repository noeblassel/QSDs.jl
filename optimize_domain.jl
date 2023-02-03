using LinearAlgebra,Plots

include("QSDs.jl")
include("SplinePotentials.jl")

function optimize_domain!(a0, b0, V, β, N, Ω_hist,obj_hist, N_iter, η0, n_line_search, grad_tol)

    a,b=a0,b0

    for i=1:N_iter
        println("=== Iteration $i ===")
        M,B,δM,∂λ=QSDs.build_FEM_matrices_1D(V_periodic,β,range(a,b,N+1))
        λs,us = eigen(M[1:N-1,1:N-1],B[1:N-1,1:N-1])
        λ1,λ2=λs[1:2]
        u1,u2=us[:,1],us[:,2]
        
        ∂aJ = -inv(β*λ1^2)*(λ2*first(u1)^2 - λ1*first(u2)^2) #from shape derivative computations (minus sign accounts for direction of outward normal at a)
        ∂bJ = inv(β*λ1^2)*(λ2*last(u1)^2-λ1*last(u2)^2)
        
        obj = best_obj = (λ2-λ1)/λ1
        println("\tDomain : ]",a,", ",b,"[")
        println("\tObjective: ",obj)

        norm_grad = sqrt(∂aJ^2 + ∂bJ^2)
        println("\tGradient: ",norm_grad)
        (norm_grad < grad_tol) && return

        println("\t== LineSearch ==")
        η=η0

        for i=1:n_line_search # line search to adjust step size
            println("\t\t step $i, η=$η")
            a_tent = a + ∂aJ*η
            b_tent = b + ∂bJ*η
            M,B,δM,∂λ=QSDs.build_FEM_matrices_1D(V_periodic,β,range(a_tent,b_tent,N+1))
            λs,us = eigen(M[1:N-1,1:N-1],B[1:N-1,1:N-1])
            λ1,λ2 = λs[1:2]
            obj_tent = (λ2-λ1)/λ1
            println("\t\tObjective :", obj_tent)

            if obj_tent < best_obj
                η /= 2
                break
            end

            best_obj = obj_tent
            η *= 2
        end

        a += ∂aJ*η
        b += ∂bJ*η
        
        push!(Ω_hist,[a,b])
        push!(obj_hist,best_obj)
    end
end

data_dir = "/home/nblassel/Documents/QSDs.jl/results_optim/"
## double well potential
    critical_points = [0.001, 0.25,0.5,0.75,0.999]
    heights = [0.0, -1.5 , 0.0, -2.0, 0.0]

    V,∇V,∇²V = SplinePotentials.spline_potential_derivatives(critical_points,heights,1)
    V_periodic(x)=V(mod(x,1))
    ## 

    β=4.0

    N=300
    core_set_B = [0.73,0.77]
    core_set_A = [0.23,0.27]

    a,b = core_set_A
    N_iter = 1500
    η0=0.1
    Ω_hist=Vector{Float64}[]
    obj_hist=Float64[]
    optimize_domain!(a,b,V_periodic,β,N,Ω_hist,obj_hist,N_iter,η0,20,1e-11)
    Ω_max = Ω_hist[argmax(obj_hist)]

    f=open(joinpath)