include("QSDs.jl")
include("GeometryUtils.jl")
include("PlotUtils.jl")

using .QSDs, .GeometryUtils, .PlotUtils, TriplotRecipes, Triangulate, Plots, LinearAlgebra


function opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points,N_iter,η0,n_line_search,log_α,min_log_α,max_log_α, grad_mask,alpha_log_file,obj_log_file)
    Nα = length(log_α)

    f=open(alpha_log_file,"w")
    println(f,"log_alpha_star=",log_α)
    close(f)

    f=open(obj_log_file,"w")
    println(f,"obj_star=",-Inf)
    close(f)

    goat_α = copy(log_α)
    goat_obj = -Inf

    for i=0:N_iter
        println("=== Iteration $i ===")
        u1,u2,λ1,λ2,∇λ1,∇λ2 = QSDs.soft_killing_grads_2D(M,B,δM,∂λ,log_α,periodic_images,dirichlet_boundary_points)

        obj = best_obj = (λ2-λ1)/λ1
        ∇obj = (λ1*∇λ2-λ2*∇λ1)/(λ1^2)
        ∇obj[grad_mask] .= 0 # no update on home core_set
        println("\t normalized gradient : ", norm(∇obj/Nα))
        println("\t objective : ", best_obj)
        
        η=η0
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
        clamp!(log_α,min_log_α,max_log_α)
        log_α[grad_mask] .= -Inf

        f=open(alpha_log_file,"w")
        println(f,"log_alpha_star=",goat_α)
        close(f)
    
        f=open(obj_log_file,"w")
        println(f,"obj_star=",-Inf)
        close(f)

    end
end

V(x,y)= cos(2π*x)-cos(2π*(y-x))

β=2.0
core_sets = [-0.5 0.5 -0.5 0.5 ; -0.5 -0.5 0.5 0.5; 0.05 0.05 0.05 0.05]
n_core_set_boundary = [10,10,10,10]

Lx = Ly=2.0
cx,cy = -1.0,-1.0

Nx = 40
Ny = 40

max_area = 0.002
min_angle = 20
triout, periodic_images , core_set_ixs = conforming_triangulation(cx,cy,Lx,Ly,Nx,Ny,core_sets,n_core_set_boundary,max_area,min_angle)
Ntri = numberoftriangles(triout)
N = numberofpoints(triout)
dirichlet_boundary_points = vcat(core_set_ixs[1],core_set_ixs[2],core_set_ixs[3])
home_coreset_points = core_set_ixs[4]
grad_mask=Cint[]

for n=1:Ntri
    (i,j,k)=triout.trianglelist[:,n]
    
    if (i in home_coreset_points) && (j in home_coreset_points) && (k in home_coreset_points)
        push!(grad_mask,n)
    end
end

#println(grad_mask)

M,B,δM,∂λ=build_FEM_matrices_2D(V,β,triout)
Niter=50
η0=0.05
log_α= zeros(Ntri)
log_α[grad_mask] .= -Inf

u1,u2,λ1,λ2,∇λ1,∇λ2=QSDs.soft_killing_grads_2D(M,B,δM,∂λ,log_α,periodic_images,dirichlet_boundary_points)

log_α_min = -10
log_α_max = 10

opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points,Niter,η0,20,log_α,log_α_min,log_α_max,grad_mask,"/home/nblassel/Documents/QSDs.jl/alpha_2D.out","/home/nblassel/Documents/QSDs.jl/obj_2D.out",10)

X=triout.pointlist[1,:]
Y=triout.pointlist[2,:]
T=triout.trianglelist


log_α_clamped = copy(log_α)

log_α_clamped[ log_α .< log_α_min] .= log_α_min
log_α_clamped[ log_α .> log_α_max] .= log_α_max

α = exp.(log_α_clamped)

z_α=PlotUtils.to_vertex_function(α,triout)
z_α[dirichlet_boundary_points] .= exp(log_α_max)

tripcolor(X,Y,log.(z_α),T,aspectratio=1,size=(800,800))