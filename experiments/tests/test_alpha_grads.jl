using Plots,LinearAlgebra

include("SplinePotentials.jl")
include("QSDs.jl")

using .SplinePotentials, .QSDs

V(x)=sin(2π*x)
β=5.0

N=400

domain=range(0,1,N+1)
M,B,δM,∂λ=QSDs.build_FEM_matrices_1D(V,β,domain)

Ω_indices = setdiff(1:N,250:350)

α= 500*abs.(randn(N))

u1,u2,λ1,λ2,∇λ1,∇λ2=QSDs.calc_soft_killing_grads(M,B,δM,∂λ,α,Ω_indices)

eps = 0.001

num_λ1=Float64[]
num_λ2=Float64[]
for i=1:N
    println(i)
    α_eps=copy(α)
    α_eps[i] += eps
    _,_,λ1_eps,λ2_eps,_,_ = QSDs.calc_soft_killing_grads(M,B,δM,∂λ,α_eps,Ω_indices)

    push!(num_λ1,(λ1_eps - λ1)/eps)
    push!(num_λ2,(λ2_eps - λ2)/eps)
end

plot(num_λ2,label="∇λ2(num)")
plot!(∇λ2,label="∇λ2")

