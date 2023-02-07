using LinearAlgebra,Plots

include("QSDs.jl")
include("SplinePotentials.jl")


function opt_alpha!(M,B,δM,∂λ,Ω_indices,N_iter,η0,n_line_search,log_α,min_log_α,max_log_α, grad_mask,log_α_hist_array=nothing,obj_hist_array=nothing,store_every=1)
    log_α0=copy(    log_α)
    for i=0:N_iter

        ((i%store_every == 0)&&(log_α_hist_array !== nothing)) && (push!(log_α_hist_array,copy(log_α0)))
        println("=== Iteration $i ===")
        u1,u2,λ1,λ2,∇λ1,∇λ2 = QSDs.soft_killing_grads(M,B,δM,∂λ,log_α0,Ω_indices)

        obj = best_obj = (λ2-λ1)/λ1
        ∇obj = (λ1*∇λ2-λ2*∇λ1)/(λ1^2)
        ∇obj[grad_mask] .= 0 # no update on home core_set
        println("\t normalized gradient : ", norm(∇obj/N))
        println("\t objective : ", best_obj)
        
        η=η0
        for i=1:n_line_search # line search to adjust step size
            println("\t\t step $i, η=$η")
            log_α_tent = log_α0 + η*∇obj
            clamp!(log_α_tent,min_log_α,max_log_α)
            u1,u2,λ1,λ2,∇λ1,∇λ2 = QSDs.soft_killing_grads(M,B,δM,∂λ,log_α_tent,Ω_indices)
            obj_tent = (λ2-λ1)/λ1
            println("\t\tObjective :", obj_tent)
            
            if obj_tent < best_obj
                η /= 2
                break
            end
            
            best_obj = obj_tent
            η *= 2
        end
        
        ((i%store_every == 0)&&(obj_hist_array !== nothing)) && (push!(obj_hist_array,best_obj))
        log_α0 += η*∇obj
        
        clamp!(log_α0,min_log_α,max_log_α)
    end
    return log_α_hist_array,obj_hist_array
end


data_dir = "/home/nblassel/Documents/QSDs.jl/results_optim/"
## double well potential
    critical_points = [0.001, 0.25,0.5,0.75,0.999]
    heights = [0.0, -1.0 , 0.0, -1.0, 0.0]

    V,∇V,∇²V = SplinePotentials.spline_potential_derivatives(critical_points,heights,1)
    ## 

    β=5.0

    N=500
    domain=range(0,1,N)

    core_set_B = [0.73,0.77]
    core_set_A = [0.23,0.27]
  
    core_set_ix_A_l = ceil(Int64,N*first(core_set_A))
    core_set_ix_A_r = floor(Int64,N*last(core_set_A))

    core_set_ix_B_l = ceil(Int64,N*first(core_set_B))
    core_set_ix_B_r = floor(Int64,N*last(core_set_B))

    core_set_ix_A = core_set_ix_A_l:core_set_ix_A_r
    core_set_ix_B= core_set_ix_B_l:core_set_ix_B_r

    M,B,δM,∂λ=QSDs.build_FEM_matrices_1D_Neumann(V,β,domain)

    Ω_indices = setdiff(1:N,core_set_ix_B)

    max_log_α=10
    min_log_α=-10
    
    N_iter = 250
    η0=0.1

    # initialize α and history
    log_α=zeros(N)
    log_α[core_set_ix_A] .= -Inf
    log_α_hist = typeof(log_α)[]
    obj_hist=Float64[]


    opt_alpha!(M,B,δM,∂λ,Ω_indices, N_iter,η0,20,log_α,min_log_α, max_log_α, core_set_ix_A,log_α_hist,obj_hist)
    log_α_max = log_α_hist[argmax(obj_hist)]
    α_max = exp.(log_α_max)
    α_max[core_set_ix_B] .= NaN
    f=open(joinpath(data_dir,"double_well_$(heights[2])_$(heights[4])_β$(β).out"),"w")
    println(f,"critical_points=",critical_points)
    println(f,"heights=",heights)
    println(f,"N=",N)
    println(f,"coreset_A=",core_set_A)
    println(f,"coreset_B=",core_set_B)

    println(f,"alpha_star_A=",α_max)


    Ω_indices = setdiff(1:N,core_set_ix_A)
    log_α=zeros(N)
    log_α[core_set_ix_A] .= -Inf
    log_α_hist = typeof(log_α)[]
    obj_hist=Float64[]


    opt_alpha!(M,B,δM,∂λ,Ω_indices, N_iter,η0,20,log_α,min_log_α, max_log_α, core_set_ix_A,log_α_hist,obj_hist)
    log_α_max = log_α_hist[argmax(obj_hist)]
    α_max = exp.(log_α_max)
    α_max[core_set_ix_A] .= NaN
    println(f,"alpha_star_B=",α_max)

    close(f)

