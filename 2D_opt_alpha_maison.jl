#!/libre/blasseln/julia-1.8.2/bin/julia

include("QSDs.jl")
include("GeometryUtils.jl")

using .QSDs, .GeometryUtils, Triangulate, LinearAlgebra, Arpack

β = parse(Float64,ARGS[1])

checkpoint_file = ARGS[2]
output_file = ARGS[3]
max_iter = parse(Int64,ARGS[4])

Nx = parse(Int64,ARGS[5])
Ny = parse(Int64,ARGS[6])

N_coreset_boundary_points=parse(Int64,ARGS[7])

max_area = parse(Float64,ARGS[8])
r = parse(Float64,ARGS[9])

min_log_α = parse(Float64,ARGS[10])
max_log_α = parse(Float64,ARGS[11])

η0 = parse(Float64,ARGS[12])
n_line_search = parse(Int64,ARGS[13])
grad_tol = parse(Float64,ARGS[14])



println("Usage: β checkpoint_file output_file Niter Nx Ny N_coreset_boundary_points max_area r log_α_min log_α_max η0 n_line_search grad_tol")

function opt_log_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points,N_iter,η0,n_line_search,log_α,min_log_α,max_log_α,grad_mask,checkpoint_file,grad_tol)
    Nα = length(log_α)
    update_ixs = setdiff(1:Nα,grad_mask)
    N = first(size(M))
    goat_α = copy(log_α)
    goat_obj = -Inf
    
    f=open(checkpoint_file,"w")
    println(f,"log_alpha_star=",goat_α)
    println(f,"obj_star=",goat_obj)
    close(f)

    for i=0:N_iter
        println("=== Iteration $i ===")
        α = exp.(log_α)
        ΔM = δM(α)
        Lred,Bred = apply_bc(M+ΔM,B,periodic_images,dirichlet_boundary_points)

        us_red,λs_red = eigs(Lred,Bred,nev=2,sigma=0,which=:LR)
        λ1,λ2 = real.(λs_red)

        u1 = reverse_bc(real.(us_red[:,1]),N,periodic_images,dirichlet_boundary_points)
        u2 = reverse_bc(real.(us_red[:,2]),N,periodic_images,dirichlet_boundary_points)

        ∇λ1 = [∂λ(u1,i) for i=update_ixs]
        ∇λ2 = [∂λ(u2,i) for i=update_ixs]

        ∇obj = (λ1*∇λ2 - λ2*∇λ1)/λ1^2
        ∇obj .*= α[update_ixs]
        
        (maximum(abs.(∇obj)) < grad_tol) && (return goat_α,goat_obj) # gradient exit condition
        
        η=η0

        best_obj = (λ2-λ1)/λ1
        
        for i=1:n_line_search # line search to adjust step size
            println("\t\t step $i, η=$η")
            log_α_tent = log_α + η*∇obj
            clamp!(log_α_tent,min_log_α,max_log_α)
            log_α_tent[grad_mask] .= -Inf
            u1,u2,λ1,λ2,∇λ1,∇λ2 = QSDs.soft_killing_grads_2D(M,B,δM,∂λ,log_α_tent,periodic_images,dirichlet_boundary_points)
            obj_tent = (λ2-λ1)/λ1
            println("\t\tObjective :", obj_tent)
            
            if obj_tent < best_obj
                η /= 2
                break
            elseif obj_tent > goat_obj
                goat_obj = obj_tent
                goat_α = copy(log_α_tent)
            end
            
            best_obj = obj_tent
            η *= 2
        end
        
        log_α .+= η*∇obj

        saturated_ixs = @.( (log_α < min_log_α) || (log_α > max_log_α))
        update_ixs = setdiff(update_ixs,saturated_ixs) # don't update parameters which have saturated their constraints
        clamp!(log_α,min_log_α,max_log_α)

        log_α[grad_mask] .= -Inf

        f=open(checkpoint_file,"w")
        println(f,"log_alpha_star=",goat_α)
        println(f,"obj_star=",goat_obj)
        close(f)

    end

    return goat_α, goat_obj
end

V(x,y)= cos(2π*x)-cos(2π*(y-x))

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
log_α = zeros(Ntri)
log_α[grad_mask] .= -Inf

goat_alpha,goat_obj = opt_log_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points,max_iter,η0,20,log_α,min_log_α,max_log_α,grad_mask,checkpoint_file,1e-8)
α_star = exp.(goat_alpha)



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