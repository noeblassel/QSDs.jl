#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature#, Plots

println("usage: N dβ well_depth_A well_depth_B output_file [istart]")
N=parse(Int64,ARGS[1])
dβ=parse(Float64,ARGS[2])
h1=parse(Float64,ARGS[3])
h2=parse(Float64,ARGS[4])
output_file=ARGS[5]

istart = (length(ARGS)>5) ? parse(Int64,ARGS[6]) : 100
include("../QSDs.jl")
include("../SplinePotentials.jl")

critical_pts=[0.0,0.25,0.5,0.75,1.0-1/N]
hmax=2.0 # height of potential barrier at boundary
hbarrier=0.0 # height of potential barrier separating the two wells
V,dV,d2V = SplinePotentials.spline_potential_derivatives(critical_pts,[hmax,-h1,hbarrier,-h2,hmax],1.0)

#= V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)

saddle_points = [0.350482,0.649518]
Ωl,Ωr=saddle_points =#

domain=range(0,1,N+1)

βrange=dβ:dβ:5

hstars_classic=Float64[]
qsd_stars=Vector{Float64}[]
domains=Vector{Float64}[]
#hstars_schrodinger=Float64[]

for β=βrange
    println(β)

    mu_tilde(q) = exp(-β * V(q))
    Z,_ = hquadrature(mu_tilde,0,1)
    mu(q) = mu_tilde(q) / Z
    
    W(q) = (β*dV(q)^2/2-d2V(q))/2
    
    weights_classic = QSDs.calc_weights_periodic(mu,domain)
    diag_weights_diff_classic,off_diag_weights_diff_classic,diag_weights_classic,off_diag_weights_classic = weights_classic

  #=   weights_schrodinger = calc_weights_schrodinger_periodic(W,domain)
    diag_weights_diff_schrodinger,off_diag_weights_diff_schrodinger,diag_weights_schrodinger,off_diag_weights_schrodinger = weights_schrodinger
 =#
    trunc_weights_classic(i)=(weights_classic[1][1:i],weights_classic[2][1:i],weights_classic[3][1:2i],weights_classic[4][1:i])
    #trunc_weights_schrodinger(i)=(weights_schrodinger[1][i:end-i+1],weights_schrodinger[2][i:end-i+1],weights_schrodinger[3][i:end-i+1],weights_schrodinger[4][i:end-i+1],weights_schrodinger[5][2i-1:end-2i+2],weights_schrodinger[6][i:end-i+1])
    
    λ1s_classic=Float64[]
    gaps_classic=Float64[]
    ws_classic=Vector{Float64}[]
    #λ1s_schrodinger=Float64[]

    #gaps_schrodinger=Float64[]

    for i=istart:length(domain)-istart
        #println("\t",i)
        λs_classic,us_classic=QSDs.QSD_1D_FEM(mu,β,domain[1:i];weights=trunc_weights_classic(i))
       # λs_schrodinger,_=QSD_1D_FEM_schrodinger(W,β,domain[i:end-i+1];weights=trunc_weights_schrodinger(i))

        λ1,λ2=λs_classic
        push!(λ1s_classic,λ1)
        push!(gaps_classic,λ2-λ1)
        push!(ws_classic,us_classic[:,1])
#=         λ1,λ2=λs_schrodinger
        push!(λ1s_schrodinger,λ1)
        push!(gaps_schrodinger,λ2-λ1) =#
    end

    ratios_classic = gaps_classic ./ λ1s_classic
    #println(ratios_classic)
   # ratios_schrodinger = gaps_schrodinger ./ λ1s_schrodinger
    imax=argmax(ratios_classic)
    hstar_classic=domain[imax]
   # hstar_schrodinger=domain[argmax(ratios_schrodinger)]
    w_star=ws_classic[imax]
    push!(w_star,0.0)
    pushfirst!(w_star,0.0)
    w_star .*= mu.(domain[1:imax+istart-1])
    w_star /= sum(w_star)*inv(N)
    push!(hstars_classic,hstar_classic)
    push!(qsd_stars,w_star)
    #push!(hstars_schrodinger,hstar_schrodinger)
    flush(stdout)
end



f=open(output_file,"w")
println(f,"qs=[",join(domain,","),"]")
println(f,"Vs=[",join(V.(domain),","),"]")
println(f,"critical_pts=[",join(critical_pts,","),"]")
println(f,"βs=[",join(βrange,","),"]")
println(f,"hstars_classic=[",join(hstars_classic,","),"]")

println(f,"qsd_stars=[\n",join(qsd_stars,",\n"),"]")
println(f,"domains=[\n",join(domains,",\n"),"]")
#println(f,"hstars_schrodinger=[",join(hstars_schrodinger,","),"]")
close(f)
