#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature

mode=ARGS[1]
N=parse(Int64,ARGS[2])
β=parse(Float64,ARGS[3])

if mode=="plot"
    plot_mode=true
else
    plot_mode=false
end

if plot_mode
    using Plots
end
include("../QSDs.jl")

V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)


W(q) = (β*dV(q)^2/2-d2V(q))/2

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z

saddle_points = [0.350482,0.649518]
Ωl,Ωr=saddle_points

h0=-0.05
hmax=first(saddle_points)

if plot_mode
    fig_λ1= plot(xlabel="h",ylabel="λ1",legend=:topright)
    fig_gap=plot(xlabel="h",ylabel="λ2-λ1",legend=:topright)
    fig_ratio=plot(xlabel="h",ylabel="(λ2-λ1)/λ1")
end

λ1s_classic=Float64[]
gaps_classic=Float64[]

λ1s_schrodinger=Float64[]
gaps_schrodinger=Float64[]

hrange=h0:0.01:hmax
V_rel(h)=V(last(saddle_points)+h)
V_rels=V_rel.(hrange)

for h=hrange
    println(h)

    λs_classic,_=QSD_1D_FEM(mu,β,Ωl-h,Ωr+h,N)
    λs_schrodinger,_=QSD_1D_FEM_schrodinger(W,β,Ωl-h,Ωr+h,N)

    λ1,λ2=λs_classic
    push!(λ1s_classic,λ1)
    push!(gaps_classic,λ2-λ1)

    λ1,λ2=λs_schrodinger
    push!(λ1s_schrodinger,λ1)
    push!(gaps_schrodinger,λ2-λ1)
end

if plot_mode
    plot!(fig_λ1,hrange,λ1s_classic,label="classic",linewidth=1,color=:red,linestyle=:dash)
    plot!(twinx(fig_λ1),hrange,V_rels,linewidth=1,color=:black,linestyle=:dot,label="")
    plot!(fig_gap,hrange,gaps_classic,label="classic",linewidth=1,color=:red,linestyle=:dash)
    plot!(twinx(fig_gap),hrange,V_rels,linewidth=1,color=:black,linestyle=:dot,label="")
    plot!(fig_ratio,hrange,gaps_classic ./ λ1s_classic,label="classic",linewidth=1,color=:red,linestyle=:dash)
    plot!(twinx(fig_ratio),hrange,V_rels,linewidth=1,color=:black,linestyle=:dot,label="")

    plot!(fig_λ1,hrange,λ1s_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
    plot!(fig_gap,hrange,gaps_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
    plot!(fig_ratio,hrange,gaps_schrodinger ./ λ1s_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)

    savefig(plot(fig_λ1,fig_gap),"./figures/domain_lambdas_$β.pdf")
    savefig(fig_ratio,"./figures/ratios_$β.pdf")
else
    f=open("results_domain_lambdas.jl","w")
    println(f,"hs=[",join(hrange,","),"]")
    println(f,"λ1s_classic=[",join(λ1s_classic,","),"]")
    println(f,"gaps_classic=[",join(gaps_classic,","),"]")
    println(f,"λ1s_schrodinger=[",join(λ1s_schrodinger,","),"]")
    println(f,"gaps_schrodinger=[",join(gaps_schrodinger,","),"]")
    close(f)
end
