#!/libre/blasseln/julia-1.8.2/bin/julia

using Plots
include("../SplinePotentials.jl")
data_dir=ARGS[1]
fig_dir=ARGS[2]

include(joinpath(data_dir,"potential.out"))
V,dV,d2V=SplinePotentials.spline_potential_derivatives(critical_pts,potential_heights,1.0)


data_file_regex=r"eigen_β(\d+\.\d+)_N(\d+)\.out"
eig_filenames = [f for f in readdir(data_dir) if occursin(data_file_regex,f)]

for f in eig_filenames
    fig_λ1= plot(xlabel="h",ylabel="λ1",legend=:topright)
    fig_gap=plot(xlabel="h",ylabel="λ2-λ1",legend=:topright)
    fig_ratio=plot(xlabel="h",ylabel="(λ2-λ1)/λ1")
end
#= #= V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q) =#
critical_pts=[0.0,0.25,0.5,0.75,1.0-1/N]
hmax=2.0 # height of potential barrier at boundary
hbarrier=0.0 # height of potential barrier separating the two wells
V,dV,d2V = SplinePotentials.spline_potential_derivatives(critical_pts,[hmax,-h1,hbarrier,-h2,hmax],1.0)

W(q) = (β*dV(q)^2/2-d2V(q))/2

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z
domain=range(0,1,N+1)

saddle_points = [0.350482,0.649518]
Ωl,Ωr=saddle_points

if plot_mode
    fig_λ1= plot(xlabel="h",ylabel="λ1",legend=:topright)
    fig_gap=plot(xlabel="h",ylabel="λ2-λ1",legend=:topright)
    fig_ratio=plot(xlabel="h",ylabel="(λ2-λ1)/λ1")
end

λ1s_classic=Float64[]
gaps_classic=Float64[]

λ1s_schrodinger=Float64[]
gaps_schrodinger=Float64[]

weights_classic = QSDs.calc_weights_periodic(mu,domain)
diag_weights_diff_classic,off_diag_weights_diff_classic,diag_weights_classic,off_diag_weights_classic = weights_classic

weights_schrodinger = QSDs.calc_weights_schrodinger_periodic(W,domain)
diag_weights_diff_schrodinger,off_diag_weights_diff_schrodinger,diag_weights_schrodinger,off_diag_weights_schrodinger = weights_schrodinger

trunc_weights_classic(i)=(weights_classic[1][1:i],weights_classic[2][1:i],weights_classic[3][1:2i],weights_classic[4][1:i])
trunc_weights_schrodinger(i)=(weights_schrodinger[1][1:i],weights_schrodinger[2][1:i],weights_schrodinger[3][1:i],weights_schrodinger[4][1:i],weights_schrodinger[5][1:2i],weights_schrodinger[6][1:i])


for i=istart:length(domain)-istart
    println(i,": ",domain[i])

    λs_classic,_=QSDs.QSD_1D_FEM(mu,β,domain[1:i];weights=trunc_weights_classic(i))
    λs_schrodinger,_=QSDs.QSD_1D_FEM_schrodinger(W,β,domain[1:i],weights=trunc_weights_schrodinger(i))

    λ1,λ2=λs_classic
    push!(λ1s_classic,λ1)
    push!(gaps_classic,λ2-λ1)

    λ1,λ2=λs_schrodinger
    push!(λ1s_schrodinger,λ1)
    push!(gaps_schrodinger,λ2-λ1)
end

if plot_mode
    plot!(fig_λ1,domain[istart:end-istart],λ1s_classic,label="classic",linewidth=1,color=:red,linestyle=:dash)
    plot!(twinx(fig_λ1),V,0,1,linewidth=1,color=:black,linestyle=:dot,label="")
    plot!(fig_gap,domain[istart:end-istart],gaps_classic,label="classic",linewidth=1,color=:red,linestyle=:dash)
    plot!(twinx(fig_gap),V,0,1,linewidth=1,color=:black,linestyle=:dot,label="")
    plot!(fig_ratio,domain[istart:end-istart],gaps_classic ./ λ1s_classic,label="classic",linewidth=1,color=:red,linestyle=:dash)
    plot!(twinx(fig_ratio),V,0,1,linewidth=1,color=:black,linestyle=:dot,label="")

    plot!(fig_λ1,domain[istart:end-istart],λ1s_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
    plot!(fig_gap,domain[istart:end-istart],gaps_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)
    plot!(fig_ratio,domain[istart:end-istart],gaps_schrodinger ./ λ1s_schrodinger,label="schrodinger",linewidth=1,color=:blue,linestyle=:dot)

    savefig(plot(fig_λ1,fig_gap),"./figures/splines/domain_lambdas_$(β)_$(h1)_$(h2).pdf")
    savefig(fig_ratio,"./figures/splines/ratios_$(β)_$(h1)_$(h2).pdf")
else
    f=open("results_domain_lambdas.jl","w")
    println(f,"hs=[",join(hrange,","),"]")
    println(f,"λ1s_classic=[",join(λ1s_classic,","),"]")
    println(f,"gaps_classic=[",join(gaps_classic,","),"]")
    println(f,"λ1s_schrodinger=[",join(λ1s_schrodinger,","),"]")
    println(f,"gaps_schrodinger=[",join(gaps_schrodinger,","),"]")
    close(f)
end =#