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
max_iter=parse(Int64,ARGS[7])

println("Usage: β output_file Nx Ny N_coreset_boundary_points max_area max_iter")

function opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points, grad_mask,Ntri,x0,max_iter)
    update_ixs = setdiff(1:Ntri,grad_mask)

    α=zeros(Ntri)
    N=first(size(M))

    ∇λ1 = zeros(Ntri-length(grad_mask))
    ∇λ2 = zeros(Ntri-length(grad_mask))

    function fg!(f,g,x)

        α[update_ixs] .= (x .^ 2)/2 # optimize square alpha

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
            g .= -2x .* (λ1*∇λ2 - λ2*∇λ1)/λ1^2
        end

        if f!== nothing
            return -(λ2 - λ1)/λ1
        end
    end
    options = Optim.Options(show_every=1,iterations = max_iter,show_trace=true)
    return optimize(Optim.only_fg!(fg!),x0,LBFGS(),options)
end

V(x,y)= cos(2π*x)-cos(2π*(y-x))

core_sets = [-0.5 0.5 -0.5 0.5 ; -0.5 -0.5 0.5 0.5; 0.05 0.05 0.05 0.05]
n_core_set_boundary = repeat([N_coreset_boundary_points],size(core_sets)[2])

Lx = Ly=2.0
cx,cy = -1.0,-1.0


min_angle = 20
triout, periodic_images , core_set_ixs = conforming_triangulation(cx,cy,Lx,Ly,Nx,Ny,core_sets,n_core_set_boundary,max_area,min_angle,quiet=false)
Ntri = numberoftriangles(triout)
N = numberofpoints(triout)
dirichlet_boundary_points = vcat(core_set_ixs[1],core_set_ixs[2],core_set_ixs[3])
home_coreset_points = core_set_ixs[4]

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
    
    if (i in home_coreset_points) && (j in home_coreset_points) && (k in home_coreset_points)
        push!(grad_mask,n)
    end
end

M,B,δM,∂λ=build_FEM_matrices_2D(V,β,triout)
x0= ones(Ntri-length(grad_mask))
results = opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points, grad_mask,Ntri,x0,max_iter)

println(results)

α_star = zeros(Ntri)
α_star[setdiff(1:Ntri,grad_mask)] .= (results.minimizer) .^ 2
α_star[grad_mask] .= 0

f=open(output_file,"a")
println(f,"α_star=",α_star)

Lred,Bred = apply_bc(M+δM(α_star),B,periodic_images,dirichlet_boundary_points)
λs,us = eigs(Lred,Bred,nev=2,sigma=0,which=:LR)
λ1,λ2 = real.(λs)
u1=real.(us[:,1])
u2=real.(us[:,2])
u1=reverse_bc(u1,N,periodic_images,dirichlet_boundary_points)
u2=reverse_bc(u2,N,periodic_images,dirichlet_boundary_points)
sqsd=qsd_2d(u1,V,β,triout)

println(f,"λ1=",λ1)
println(f,"λ2=",λ2)
println(f,"u1=",u1)
println(f,"u2=",u2)
println(f,"sqsd=",sqsd)

close(f)