
using Plots, LinearAlgebra, Cubature

include("../QSDs.jl")


V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)

β = 2.0

W(q) = (β*dV(q)^2/2-d2V(q))/2

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z

saddle_points = [0.350482,0.649518]

fig_classic= plot(xlims=(0,1),xlabel="q")
plot!(fig_classic, V,0,1,color=:black, linestyle=:dot,label="")
vline!(fig_classic,saddle_points,label="",linestyle=:dot,color=:black)

fig_schrodinger= plot(xlims=(0,1),xlabel="q")
plot!(fig_schrodinger, V,0,1,color=:black, linestyle=:dot,label="")
vline!(fig_schrodinger,saddle_points,label="",linestyle=:dot,color=:black)

N = 1000

domain=collect(Float64,range(0,1,N+1))
midpoints = domain[1:end-1] .+ (domain[2]-domain[1])/2

dq= domain[2]-domain[1]

weights_schrodinger = calc_weights_schrodinger_periodic(W,domain)
weights_classic = calc_weights_periodic(mu,domain)
αs=zeros(N)

mu_reweighting_classic= mu.(domain[2:end])
mu_reweighting_schrodinger= exp.(-β * V.(domain[2:end]) /2)

function fill_αs!(val::Float64)
    for (i,p)=enumerate(midpoints)
        !(first(saddle_points) < p < last(saddle_points)) && (αs[i] = val)
    end
end

for α=[0.1,1.0,10.0,100.0,1000.0]
    println(α)
    fill_αs!(α)

    λs_classic,us_classic=SQSD_1D_FEM(V,β,αs;weights=weights_classic)
    λs_schrodinger,us_schrodinger=SQSD_1D_FEM_schrodinger(W,β,αs;weights=weights_schrodinger)
    qsd_classic = us_classic[:,1] .* mu_reweighting_classic
    qsd_schrodinger = us_schrodinger[:,1] .* mu_reweighting_schrodinger

    #normalize
    qsd_schrodinger /= sum(qsd_schrodinger)*dq
    qsd_classic /= sum(qsd_classic)*dq

    plot!(fig_classic,domain[2:end],qsd_classic,label="α=$α",linewidth=1)
    plot!(fig_schrodinger,domain[2:end],qsd_schrodinger,label="α=$α",linewidth=1)
end

savefig(fig_classic,"./figures/interpolation_classic.pdf")
savefig(fig_schrodinger,"./figures/interpolation_schrodinger.pdf")
