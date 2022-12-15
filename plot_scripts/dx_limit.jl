using Plots, Cubature

include("../QSDs.jl")

V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)
β=2.0
mu_tilde(q)=exp(-β*V(q))
Z,_=hquadrature(mu_tilde,0,1)
mu(q)=mu_tilde(q)/Z

saddle_points=[0.350482,0.649518]
Ωl,Ωr=saddle_points

qsd_plot=plot(xlabel="q",ylabel="likelihood")
sqsd_plot=plot(xlabel="q",ylabel="likelihood")

α_max=100

for N=[10,20,50,100,200,500]
    println(N)
    Ω=range(Ωl,Ωr,N+1)
    λs,us=QSDs.QSD_1D_FEM(mu,β,Ω;)
    w=us[:,1]
    push!(w,0.0)
    pushfirst!(w,0.0)
    w .*= mu.(Ω)
    w /= sum(w)*(Ωr-Ωl)/N
    plot!(qsd_plot,Ω,w,label="N=$N")

    full_domain=range(0,1,N+1)
    α=Float64[(Ωl<=x<=Ωr) ? 0.0 : α_max for x in full_domain]
    λs,us = QSDs.SQSD_1D_FEM(mu,β,α)
    w=us[:,1]
    w .*=mu.(full_domain)
    w /= sum(w)*1/N
    plot!(sqsd_plot,full_domain,w,label="N=$N")

end

savefig(qsd_plot,"figures/limit_dx_qsd.pdf")
savefig(sqsd_plot,"figures/limit_dx_sqsd.pdf")