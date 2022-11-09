using Cubature,LinearAlgebra

include("QSDs.jl")

N,β,Niter,cutoff = ARGS

N,Niter=parse.(Int64,[N,Niter])
β,cutoff=parse.(Float64,[β,cutoff])

V(q) = 2cos(2π*q) - cos(4π*q)
D = collect(Float64,range(0,1,N+1))
mu_no(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_no,0,1)
mu(q) = mu_no(q)/Z

W=calc_weights_periodic(mu,D)

#construct u -> u⊺(∂δM/∂α)u
    

    """
    Compute the gradient of an eigenvalue of the FEM generator with respect to α.
    The argument u is the corresponding eigenvector.
    """
    function Dλ_Dα(u; weights)
        _,_,diag_weights,off_diag_weights = weights
        grad=zero(u)
        N=length(u)

        for i=1:N
            j = (i==1) ? N : i-1
            grad[i] = u[i]^2 * diag_weights[2i-1] + u[j]^2 * diag_weights[2j] + 2off_diag_weights[j]*u[i]*u[j]
        end

        return grad
    end

    function optimize_alpha(V,β,α0;weights, grad_tol = 1e-6, max_iter=100000, cutoff=10000.0, lr=0.1, log_every=1000)
        α_star = copy(α0)
        J_hist=Float64[]
        grad_hist=Float64[]
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
            end

            #println((λ2-λ1)/λ1)
        end

        return α_star,J_hist,grad_hist
    end


α0=ones(N)
α_star,J_hist,grad_hist=optimize_alpha(V,β,α0; weights=W , max_iter=Niter, cutoff=cutoff)

if !isdir("results_optim")
    mkdir("results_optim")
end

f=open("results_optim/results_$(β)_$(N).out","w")
println(f,α_star)
println(f,J_hist)
println(f,grad_hist)