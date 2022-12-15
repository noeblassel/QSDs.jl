#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature#, Plots

println("usage: N βmin βmax dβ well_depth_A well_depth_B data_dir [istart=100]")
N=parse(Int64,ARGS[1])
βmin=parse(Float64,ARGS[2])
βmax=parse(Float64,ARGS[3])
dβ=parse(Float64,ARGS[4])
h1=parse(Float64,ARGS[5])
h2=parse(Float64,ARGS[6])
data_dir=ARGS[7]

istart = (length(ARGS)>7) ? parse(Int64,ARGS[8]) : 100
include("../QSDs.jl")
include("../SplinePotentials.jl")

critical_pts=[0.0,0.25,0.5,0.75,1.0-1/N]
hmax=2.0 # height of potential barrier at boundary
hbarrier=0.0 # height of potential barrier separating the two wells
V,dV,d2V = SplinePotentials.spline_potential_derivatives(critical_pts,[hmax,-h1,hbarrier,-h2,hmax],1.0)

domain=range(0,1,N+1)

f=open(joinpath(data_dir,"potential.out"),"w")
println(f,"N=",N)
println(f,"domain=[",join(domain,","),"]")
println(f,"Vs=[",join(V.(domain),","),"]")
println(f,"critical_pts=",critical_pts)
println(f,"potential_heights=",[hmax,-h1,hbarrier,-h2,hmax])
close(f)

βrange=βmin:dβ:βmax
for β=βrange
    mu_tilde(q) = exp(-β * V(q))
    Z,_ = hquadrature(mu_tilde,0,1)
    mu(q) = mu_tilde(q) / Z
    
    weights_classic = QSDs.calc_weights_periodic(mu,domain)
    diag_weights_diff_classic,off_diag_weights_diff_classic,diag_weights_classic,off_diag_weights_classic = weights_classic

    trunc_weights_classic(i)=(weights_classic[1][1:i],weights_classic[2][1:i],weights_classic[3][1:2i],weights_classic[4][1:i])

    λ1s_classic=Float64[]
    gaps_classic=Float64[]

    for i=istart:length(domain)-istart
        λs_classic,us_classic=QSDs.QSD_1D_FEM(mu,β,domain[1:i];weights=trunc_weights_classic(i))
        
        λ1,λ2=λs_classic
        push!(λ1s_classic,λ1)
        push!(gaps_classic,λ2-λ1)

        qsd=us_classic[:,1]
        push!(qsd,0)
        pushfirst!(qsd,0)
        qsd /=sum(qsd)*inv(N)

        f=open(joinpath(data_dir,"beta$(β)_N$(N)_ix$(i).out"),"w")
        println(f,"λs=",λs_classic)
        println(f,"us=",us_classic)
        println(f,"qsd=",qsd)
        close(f)
    end

    ratios_classic = gaps_classic ./ λ1s_classic
    f=open(joinpath(data_dir,"eigen_β$(β)_N$(N).out"),"w")
    println(f,"λ1s=",λ1s_classic)
    println(f,"gaps=",gaps_classic)
    println(f,"ratios=",ratios_classic)
    close(f)
end