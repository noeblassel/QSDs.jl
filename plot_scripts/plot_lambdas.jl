#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature

mode=ARGS[1]
N=parse(Int64,ARGS[2])
β = parse(Float64,ARGS[3])

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

#W(q) = (β*dV(q)^2/2-d2V(q))/2

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z

saddle_points = [0.350482,0.649518]

if plot_mode
    fig_λ1= plot(xlabel="α",ylabel="λ1",xaxis=:log,yaxis=:log,legend=:topleft)
    fig_gap=plot(xlabel="α",ylabel="λ2-λ1",xaxis=:log,yaxis=:log,legend=:topleft)
    fig_ratios=plot(xlabel="α",ylabel="(λ2-λ1)/λ1",xaxis=:log,yaxis=:log,legend=:topleft)
end

domain=collect(Float64,range(0,1,N+1))
midpoints = domain[1:end-1] .+ (domain[2]-domain[1])/2

dq= domain[2]-domain[1]

#weights_schrodinger = calc_weights_schrodinger_periodic(W,domain)
weights_classic = QSDs.calc_weights_periodic(mu,domain)
αs=zeros(N)

function fill_αs!(val::Float64)
    for (i,p)=enumerate(midpoints)
        !(first(saddle_points) < p < last(saddle_points)) && (αs[i] = val)
    end
end

lg_alpha_range= -4:0.02:4

λ1s_classic=Float64[]
gaps_classic=Float64[]

ratios_classic=Float64[]
n_modes=Int64[]
#λ1s_schrodinger=Float64[]
#gaps_schrodinger=Float64[]


for α= 10 .^ lg_alpha_range
    println(α)
    fill_αs!(α)

    λs_classic,us_classic=QSDs.SQSD_1D_FEM(V,β,αs;weights=weights_classic)
    #λs_schrodinger,_=SQSD_1D_FEM_schrodinger(W,β,αs;weights=weights_schrodinger)
    w=us_classic[:,1]
    w .*= mu.(domain[2:end])
    w /= sum(w)/N
    #= savefig(plot(domain[2:end],w),"figures/test_qsd/$α.pdf")
    println(length(w)) =#
    nbumps = sum(1 for i=2:N-1 if (w[i-1]<w[i]) && (w[i+1]<w[i]))

    λ1,λ2=λs_classic
    push!(λ1s_classic,λ1)
    push!(gaps_classic,λ2-λ1)
    push!(ratios_classic, (λ2-λ1)/λ1)
    push!(n_modes,nbumps)
#=     λ1,λ2=λs_schrodinger
    push!(λ1s_schrodinger,λ1)
    push!(gaps_schrodinger,λ2-λ1) =#
end
#print(n_modes)
bifurcation = first(10^lg_alpha_range[i] for i=1:length(lg_alpha_range) if n_modes[i]==1)

# record optimal QSD

ix_max=argmax(ratios_classic)
α_max=10 ^ lg_alpha_range[ix_max]
fill_αs!(α_max)
λs,us=QSDs.SQSD_1D_FEM(V,β,αs;weights=weights_classic)
w_opt=us[:,1]
w_opt .*= mu.(domain[2:end])
w_opt /= sum(w_opt)/N


if plot_mode
    qsd_plot=plot(domain[2:end],w_opt,xlabel="q",ylabel="likelihood",label="qsd")
    plot!(qsd_plot,domain,mu.(domain),label="mu")
    #plot!(twinx(),qsd_plot,domain,V.(domain),label="",linestyle=:dot,linewidth=1,color=:black)
    savefig(qsd_plot,"./figures/soft_qsd_$β.pdf")

    plot!(fig_λ1,10 .^ lg_alpha_range,λ1s_classic,linewidth=1,color=:red)
    plot!(fig_gap,10 .^ lg_alpha_range,gaps_classic,linewidth=1,color=:red)
    vline!(fig_λ1,[bifurcation],label="α₀",linewidth=1,color=:black,linestyle=:dot)
    vline!(fig_gap,[bifurcation],label="α₀",linewidth=1,color=:black,linestyle=:dot)

    plot!(fig_ratios,10 .^ lg_alpha_range,ratios_classic,label="",linewidth=1,color=:red)
    vline!(fig_ratios,[bifurcation],label="α₀",linewidth=1,color=:black,linestyle=:dot)

    savefig(plot(fig_λ1,fig_gap),"./figures/soft_lambdas_$β.pdf")
    savefig(fig_ratios,"./figures/soft_ratios_$β.pdf")
#=     plot!(fig_λ1,10 .^ lg_alpha_range,inv.(λ1s_schrodinger),label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
    plot!(fig_gap,10 .^ lg_alpha_range,gaps_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
     =#
end
f=open("result_files/soft_lambdas_$β.out","w")
println(f,"""
V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)

#W(q) = (β*dV(q)^2/2-d2V(q))/2

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z

saddle_points = [0.350482,0.649518]
""")
println(f,"β=$β")
println(f,"αs=[",join(10 .^ lg_alpha_range,","),"]")
println(f,"λ1s_classic=[",join(λ1s_classic,","),"]")
println(f,"gaps_classic=[",join(gaps_classic,","),"]")
println(f,"domain=[",join(domain,","),"]")
println(f,"opt_qsd=[",join(w_opt,","),"]")#= 
println(f,"λ1s_schrodinger=[",join(λ1s_schrodinger,","),"]")
println(f,"gaps_schrodinger=[",join(gaps_schrodinger,","),"]") =#
close(f)