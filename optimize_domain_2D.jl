#!/libre/blasseln/julia-1.8.2/bin/julia

include("QSDs.jl")
include("GeometryUtils.jl")

using .QSDs, .GeometryUtils, Triangulate, Optim, Arpack

β = parse(Float64,ARGS[1])

output_file = ARGS[2]

Nx = parse(Int64,ARGS[3])
Ny = parse(Int64,ARGS[4])

N_coreset_boundary_points = parse(Int64,ARGS[5])

max_area = parse(Float64,ARGS[6])
max_iter= parse(Int64,ARGS[7])
max_α = parse(Float64,ARGS[8])

println("Usage: β output_file Nx Ny N_coreset_boundary_points max_area max_iter max_α")

function opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points, grad_mask,Ntri,x0,max_iter,max_α,cons_tol=1e-12)
    update_ixs = setdiff(1:Ntri,grad_mask)

    α=zeros(Ntri)
    N=first(size(M))

    ∇λ1 = zeros(Ntri-length(grad_mask))
    ∇λ2 = zeros(Ntri-length(grad_mask))


    function fg!(f,g,x)
        α[update_ixs] .= min.(max_α,max.(0,x))
        
        Lper,Bper = apply_bc(M+δM(α),B,periodic_images,dirichlet_boundary_points)

        λs,us = eigs(Lper,Bper,nev=2,sigma=0,which=:LR)
        λ1 , λ2 = real.(λs)

        u1 = real.(us[:,1])
        u2 = real.(us[:,2])

        u1 = reverse_bc(u1,N,periodic_images,dirichlet_boundary_points)
        u2 = reverse_bc(u2,N,periodic_images,dirichlet_boundary_points)

        ∇λ1 .= [∂λ(u1,i) for i=update_ixs]
        ∇λ2 .= [∂λ(u2,i) for i=update_ixs]

        if g !== nothing
            g .= -(λ1*∇λ2 - λ2*∇λ1)/λ1^2
            g[α[update_ixs] .<= cons_tol] .= 0
            g[α[update_ixs] .>= max_α - cons_tol] .= 0
        end

        if f!== nothing
            return -(λ2 - λ1)/λ1
        end

    end

    options = Optim.Options(show_every=1,iterations = max_iter,show_trace=true)
    return optimize(Optim.only_fg!(fg!),x0,LBFGS(),options)
end

V(x,y) = cos(2π*x)-cos(2π*(y-x))

r=0.2

core_sets = [t->[0.5 + r*cos(2π*t),-0.5 + r*(sin(2π*t)+cos(2π*t))]]
core_set_tests = [(x,y)->((x-0.5)^2+(x-y-1)^2 < r^2)]

n_core_set_boundary = fill(N_coreset_boundary_points,length(core_sets))

Lx = Ly=2.0
cx,cy = -1.0,-1.0


min_angle = 20
triout, periodic_images , core_set_ixs = conforming_triangulation(cx,cy,Lx,Ly,Nx,Ny,core_sets,core_set_tests,n_core_set_boundary,max_area,min_angle,quiet=false)
Ntri = numberoftriangles(triout)
N = numberofpoints(triout)
dirichlet_boundary_points = Cint[]#vcat(core_set_ixs[1],core_set_ixs[3],core_set_ixs[4])
home_coreset_points = core_set_ixs[1]

X=triout.pointlist[1,:]
Y=triout.pointlist[2,:]
T=triout.trianglelist

f = open(output_file,"w")
println(f,"β=",β)
println(f,"X=",X)
println(f,"Y=",Y)
println(f,"T=",T)
println(f,"N=",N)
println(f,"Ntri=",Ntri)
println(f,"periodic_images=",periodic_images)
println(f,"dirichlet_boundary_points=",dirichlet_boundary_points)
println(f,"home_coreset_points=",home_coreset_points)
close(f)

grad_mask=Cint[]

for n=1:Ntri
    (i,j,k)=triout.trianglelist[:,n]
    
    for c=1:1
        if (i in core_set_ixs[c]) && (j in core_set_ixs[c]) && (k in core_set_ixs[c])
            push!(grad_mask,n)
        end
    end
end


M,B,δM,∂λ=build_FEM_matrices_2D(V,β,triout)

x = ones(Ntri-length(grad_mask))

results = opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points, grad_mask,Ntri,x,max_iter,max_α)

println(results)

x_star = results.minimizer

α_star = zeros(Ntri)
α_star[setdiff(1:Ntri,grad_mask)] .= α_star
clamp!(α_star,0,max_α)

Lred,Bred = apply_bc(M+δM(α_star),B,periodic_images,dirichlet_boundary_points)
λs,us = eigs(Lred,Bred,nev=2,sigma=0,which=:LR)
λ1,λ2 = real.(λs)
u1=real.(us[:,1])
u2=real.(us[:,2])
u1=reverse_bc(u1,N,periodic_images,dirichlet_boundary_points)
u2=reverse_bc(u2,N,periodic_images,dirichlet_boundary_points)
sqsd=qsd_2d(u1,V,β,triout)

#α_star[dirichlet_boundary_points] .= Inf

f=open(output_file,"a")
println(f,"α_star=",α_star)

println(f,"λ1=",λ1)
println(f,"λ2=",λ2)
println(f,"u1=",u1)
println(f,"u2=",u2)
println(f,"sqsd=",sqsd)

close(f)