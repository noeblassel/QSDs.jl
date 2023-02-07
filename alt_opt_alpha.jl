#!/libre/blasseln/julia-1.8.2/bin/julia

include("QSDs.jl")
include("GeometryUtils.jl")

using .QSDs, .GeometryUtils, Triangulate, Optim

β = parse(Float64,ARGS[1])

output_file = ARGS[2]

Nx = parse(Int64,ARGS[3])
Ny = parse(Int64,ARGS[4])

N_coreset_boundary_points = parse(Int64,ARGS[5])

max_area = parse(Float64,ARGS[6])

println("Usage: β output_file Nx Ny N_coreset_boundary_points max_area")

function opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points, grad_mask,Ntri,x0)
    update_ixs = setdiff(1:Ntri,grad_mask)
    function fg!(f,g,x)

        log_α = zeros(Ntri)
        log_α[update_ixs] .= x

        log_α[grad_mask] .= -Inf

        u1,u2,λ1,λ2,∇λ1,∇λ2 = QSDs.soft_killing_grads_2D(M,B,δM,∂λ,log_α,periodic_images,dirichlet_boundary_points)

        if g !== nothing
            g .= (- (λ1*∇λ2-λ2*∇λ1)/(λ1^2))[update_ixs]
        end

        if f!== nothing
            return -(λ2 - λ1)/λ1
        end
    end
    return optimize(Optim.only_fg!(fg!),x0,LBFGS())
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
x0= zeros(Ntri-length(grad_mask))
results = opt_alpha!(M,B,δM,∂λ,periodic_images,dirichlet_boundary_points, grad_mask,Ntri,x0)

println(results)

log_α_star = zeros(Ntri)
log_α_star[setdiff(1:Ntri,grad_mask)] .= results.minimizer
log_α_star[grad_mask] .= -Inf

f=open(output_file,"a")
println(f,"log_α_star=",log_α_star)
close(f)