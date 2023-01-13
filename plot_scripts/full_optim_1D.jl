#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature#, Plots

println("usage: N β well_depth_A well_depth_B output_file Δix")
N=parse(Int64,ARGS[1])
β=parse(Float64,ARGS[2])
h1=parse(Float64,ARGS[3])
h2=parse(Float64,ARGS[4])
output_file=ARGS[5]
Δix = parse(Int64,ARGS[6])

include("../QSDs.jl")
include("../SplinePotentials.jl")

critical_pts=[0.0,0.25,0.5,0.75,1.0-1/N]
hmax=2.0 # height of potential barrier at boundary
hbarrier=0.0 # height of potential barrier separating the two wells
V,dV,d2V = SplinePotentials.spline_potential_derivatives(critical_pts,[hmax,-h1,hbarrier,-h2,hmax],1.0)


domain=range(0,1,N+1)

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z


weights_classic = QSDs.calc_weights_periodic(mu,domain)
diag_weights_diff_classic,off_diag_weights_diff_classic,diag_weights_classic,off_diag_weights_classic = weights_classic

slice_weights_classic(i,j)=(weights_classic[1][i:j],weights_classic[2][i:j],weights_classic[3][2i-1:2j],weights_classic[4][i:j]) # a slice of weights
ix_range = 1:Δix:N
L = length(ix_range)

λ1s = zeros(L,L)
λ2s = zeros(L,L)

l_normal_derivative_u1 = zeros(L,L)
r_normal_derivative_u1 = zeros(L,L)

l_normal_derivative_u2 = zeros(L,L)
r_normal_derivative_u2 = zeros(L,L)

h=1/N

for i=1:L-1
    println(i)
    for j=i+1:L
        println("\t",j)
        ix_a = ix_range[i]
        ix_b = ix_range[j]

        λs,us=QSDs.QSD_1D_FEM(mu,β,domain[ix_a:ix_b];weights=slice_weights_classic(ix_a,ix_b))
        

        λ1s[i,j] = first(λs)
        λ2s[i,j] = last(λs)

        u1=us[:,1]
        u2=us[:,2]

        (u1[5] <= 0 ) && (u1 = -u1)
        (u2[5] <= 0) && (u2 = -u2)

        l_normal_derivative_u1[i,j]=(u1[2]-u1[1])/h
        r_normal_derivative_u1[i,j]=(u1[end-1]-u1[end])/h

        l_normal_derivative_u2[i,j]=(u2[2]-u2[1])/h
        r_normal_derivative_u2[i,j]=(u2[end-1]-u2[end])/h
    end
end

f=open(output_file,"w")
println(f,"l_normal_derivative_u1=",l_normal_derivative_u1)
println(f,"r_normal_derivative_u1=",r_normal_derivative_u1)
println(f,"l_normal_derivative_u2=",l_normal_derivative_u2)
println(f,"r_normal_derivative_u2=",r_normal_derivative_u2)

println(f,"λ1s=",λ1s)
println(f,"λ2s=",λ2s)

close(f)