include("QSDs.jl")
include("SplinePotentials.jl")

using Plots

h1,h2,h3=(2.0,1.0,7.0)
critical_points=[0.01,1/6,1/3,1/2,2/3,5/6,0.99]
heights = [0.0,h1,h2,h3,h2,h1,0.0]

β=1.0
V,dV,d2V= SplinePotentials.spline_potential_derivatives(critical_points,heights,1.0)
mu_tilde(q)=exp(-β*V(q))
Z = 0.12715233833442796
mu(q) = mu_tilde(q) / Z
W(q) = (β*dV(q)^2/2-d2V(q))/2

Ω=range(critical_points[2:4:6]...,N)
N = 400
λs,us = QSDs.QSD_1D_FEM(mu,β,Ω)
λs_schro,us_schro =QSDs.QSD_1D_FEM_schrodinger(W,β,Ω)
w=us[:,1]

push!(w,0.0)
pushfirst!(w,0.0)

mus_qsd = mu.(Ω)
qsd = w .* mus_qsd
qsd /= sum(qsd) * (critical_points[6]-critical_points[2])/N

w_schro=us_schro[:,1]

push!(w_schro,0.0)
pushfirst!(w_schro,0.0)

renorm_qsd_schro = exp.(-β*V.(Ω)/2)
qsd_schro = w_schro .* renorm_qsd_schro
qsd_schro /= sum(qsd_schro) * (critical_points[6]-critical_points[2])/N


plot(mu,0,1,label="μ",legend=:topright)
plot!(Ω,qsd,label="classic")
plot!(Ω,qsd_schro,label="schro")
plot!(twinx(),V,0,1,label="",linestyle=:dot,color=:grey)