using Plots,LinearAlgebra,Cubature

include("QSDs.jl")

β=0.8


V(q) = 2cos(2π*q) - cos(4π*q)
dV(q) = -4π*sin(2π*q)+4π*sin(4π*q)
d2V(q)= -8π^2*cos(2π*q)+16π^2*cos(4π*q)

dVnum(q;h=0.1)=(V(q+h)-V(q))/h
d2Vnum(q;h=0.1)=(V(q+h)-2V(q)+V(q-h))/h^2

W(q) = 0.5 * (0.5β*dV(q)^2 - d2V(q))
N=100
println(N)
Ωl=1/6
Ωr=5/6
D = collect(Float64,range(0,1,N+1))
Ω=range(Ωl,Ωr,N)

mu_no(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_no,0,1)
mu(q) = mu_no(q)/Z

weights_schrodinger=calc_weights_schrodinger_periodic(W,D)
weights_naive=calc_weights_periodic(mu,D)