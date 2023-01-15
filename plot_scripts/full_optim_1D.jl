#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature#, Plots

println("usage: N β l_well_depth well_depth r_well_depth l_coreset r_coreset min_dim output_file progress_log_file")

N=parse(Int64,ARGS[1])
β=parse(Float64,ARGS[2])

l_well_depth=parse(Float64,ARGS[3])
well_depth = parse(Float64,ARGS[4])
r_well_depth=parse(Float64,ARGS[5])

l_coreset=parse(Float64,ARGS[6])
r_coreset=parse(Float64,ARGS[7])

min_dim = parse(Int64,ARGS[8])

output_file=ARGS[9]
progress_log_file = ARGS[10]

include("../QSDs.jl")
include("../SplinePotentials.jl")
-
critical_pts=[0.0,l_coreset,(0.5+l_coreset)/2,0.5,(0.5+r_coreset)/2,r_coreset,1-1/N]
potential_heights = [0.0,-l_well_depth,0.0,-well_depth,0.0,-r_well_depth,0.0]
V,dV,d2V = SplinePotentials.spline_potential_derivatives(critical_pts,potential_heights,1.0)


domain=range(l_coreset,r_coreset,N+1)

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z


weights_classic = QSDs.calc_weights_periodic(mu,domain)
diag_weights_diff_classic,off_diag_weights_diff_classic,diag_weights_classic,off_diag_weights_classic = weights_classic

slice_weights_classic(i,j)=(weights_classic[1][i:j],weights_classic[2][i:j],weights_classic[3][2i-1:2j],weights_classic[4][i:j]) # a slice of weights

λ1s = zeros(N,N)
λ2s = zeros(N,N)

l_normal_derivative_u1 = zeros(N,N)
r_normal_derivative_u1 = zeros(N,N)

l_normal_derivative_u2 = zeros(N,N)
r_normal_derivative_u2 = zeros(N,N)

h=1/N

for i=1:N-min_dim
    for j=i+min_dim:N

        λs,us=QSDs.QSD_1D_FEM(mu,β,domain[i:j];weights=slice_weights_classic(i,j))
        

        λ1s[i,j] = first(λs)
        λ2s[i,j] = last(λs)

        u1=us[:,1]
        u2=us[:,2]

        (u1[1] <= 0 ) && (u1 = -u1)
        (u2[1] <= 0) && (u2 = -u2)

        l_normal_derivative_u1[i,j]=(-u1[1])
        r_normal_derivative_u1[i,j]=(u1[end])

        l_normal_derivative_u2[i,j]=(-u2[1])
        r_normal_derivative_u2[i,j]=(u2[end])
    end
    f= open(progress_log_file,"w")
    println(f,"$(round(100*i/(N-min_dim),digits=2))% done")
    close(f)
end

f=open(output_file,"w")
println(f,"l_normal_derivative_u1=",l_normal_derivative_u1)
println(f,"r_normal_derivative_u1=",r_normal_derivative_u1)
println(f,"l_normal_derivative_u2=",l_normal_derivative_u2)
println(f,"r_normal_derivative_u2=",r_normal_derivative_u2)

println(f,"λ1s=",λ1s)
println(f,"λ2s=",λ2s)

close(f)