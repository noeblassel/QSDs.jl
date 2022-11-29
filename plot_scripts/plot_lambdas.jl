#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature

mode=ARGS[1]
N=parse(Int64,ARGS[2])

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

β = 2.0

W(q) = (β*dV(q)^2/2-d2V(q))/2

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z

saddle_points = [0.350482,0.649518]

if plot_mode
    fig_λ1= plot(xlabel="α",ylabel="λ1",xaxis=:log,yaxis=:log)
    fig_gap=plot(xlabel="α",ylabel="λ2-λ1",xaxis=:log,yaxis=:log)

    fig_schrodinger_λ1= plot(xlabel="α",ylabel="λ1",xaxis=:log,yaxis=:log)
    fig_schrodinger_gap=plot(xlabel="α",ylabel="λ2-λ1",xaxis=:log,yaxis=:log)
end

domain=collect(Float64,range(0,1,N+1))
midpoints = domain[1:end-1] .+ (domain[2]-domain[1])/2

dq= domain[2]-domain[1]

weights_schrodinger = calc_weights_schrodinger_periodic(W,domain)
weights_classic = calc_weights_periodic(mu,domain)
αs=zeros(N)

function fill_αs!(val::Float64)
    for (i,p)=enumerate(midpoints)
        !(first(saddle_points) < p < last(saddle_points)) && (αs[i] = val)
    end
end

lg_alpha_range= -4:0.01:4

λ1s_classic=Float64[]
gaps_classic=Float64[]

λ1s_schrodinger=Float64[]
gaps_schrodinger=Float64[]


for α= 10 .^ lg_alpha_range
    println(α)
    fill_αs!(α)

    λs_classic,_=SQSD_1D_FEM(V,β,αs;weights=weights_classic)
    λs_schrodinger,_=SQSD_1D_FEM_schrodinger(W,β,αs;weights=weights_schrodinger)

    λ1,λ2=λs_classic
    push!(λ1s_classic,λ1)
    push!(gaps_classic,λ2-λ1)

    λ1,λ2=λs_schrodinger
    push!(λ1s_schrodinger,λ1)
    push!(gaps_schrodinger,λ2-λ1)
end

if plot_mode
    plot!(fig_λ1,10 .^ lg_alpha_range,λ1s_classic,label="classic",linewidth=1,color=:red,linestyle=:dot)
    plot!(fig_gap,10 .^ lg_alpha_range,gaps_classic,label="classic",linewidth=1,color=:red,linestyle=:dot)
    # savefig(plot(fig_classic_λ1,fig_classic_gap),"./figures/lambdas_classic.pdf")#=  =#

    plot!(fig_λ1,10 .^ lg_alpha_range,λ1s_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
    plot!(fig_gap,10 .^ lg_alpha_range,gaps_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
    savefig(plot(fig_λ1,fig_gap),"./figures/lambdas.pdf")
else
    f=open("results_lambdas.jl","w")
    println(f,"αs=[",join(10 .^ lg_alpha_range,","),"]")
    println(f,"λ1s_classic=[",join(λ1s_classic,","),"]")
    println(f,"gaps_classic=[",join(gaps_classic,","),"]")
    println(f,"λ1s_schrodinger=[",join(λ1s_schrodinger,","),"]")
    println(f,"gaps_schrodinger=[",join(gaps_schrodinger,","),"]")
    close(f)
end