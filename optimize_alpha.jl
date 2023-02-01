using LinearAlgebra,Plots

include("QSDs.jl")
include("SplinePotentials.jl")


function opt_alpha!(N_iter,η,log_α,max_log_α, grad_mask,log_α_hist_array=nothing,obj_hist_array=nothing,store_every=10)
    log_α0=copy(log_α)
    for i=0:N_iter
        ((i%store_every == 0)&&(log_α_hist_array !== nothing)) && (push!(log_α_hist_array,copy(log_α0)))
        println("=== Iteration $i ===")
        u1,u2,λ1,λ2,∇λ1,∇λ2 = QSDs.calc_soft_killing_grads(M,B,δM,∂λ,log_α0,Ω_indices)
        println("\t objective : ", λ2/λ1)
        ((i%store_every == 0)&&(obj_hist_array !== nothing)) && (push!(obj_hist_array,λ2/λ1))
        ∇obj = (λ1*∇λ2-λ2*∇λ1)/(λ1^2)
        ∇obj[grad_mask] .= 0 # no update on home core_set
        println("\t normalized gradient : ", norm(∇obj/N))
        log_α0 += η*∇obj
        clamp!(log_α0,-Inf,max_log_α)
    end
    return log_α_hist_array,obj_hist_array
end

## double well potential
    critical_points = [0.001, 0.25,0.5,0.75,0.999]
    heights = [0.0, -1.5 , 0.0, -2.0, 0.0]

    V,∇V,∇²V = SplinePotentials.spline_potential_derivatives(critical_points,heights,1)
    ## 

    β=2.0

    N=400
    domain=range(0,1,N+1)

    core_set_A = [0.72,0.78]
    core_set_B = [0.22,0.28]
  
    core_set_ix_A_l = ceil(Int64,N*first(core_set_A))
    core_set_ix_A_r = floor(Int64,N*last(core_set_A))

    core_set_ix_B_l = ceil(Int64,N*first(core_set_B))
    core_set_ix_B_r = floor(Int64,N*last(core_set_B))

    core_set_ix_A = core_set_ix_A_l:core_set_ix_A_r
    core_set_ix_B= core_set_ix_B_l:core_set_ix_B_r

    M,B,δM,∂λ=QSDs.build_FEM_matrices_1D(V,β,domain)

    Ω_indices = setdiff(1:N,core_set_ix_B)

    max_log_α=Inf
    # initialize α and history
    log_α=zeros(N)
    log_α[core_set_ix_A] .= -Inf
    log_α_hist = typeof(log_α)[]
    obj_hist=Float64[]

    N_iter = 4000
    η=5.0
    store_every=100

    opt_alpha!(N_iter,η,log_α,max_log_α, core_set_ix_A,log_α_hist,obj_hist)

    # to do adaptive step gradient descent
