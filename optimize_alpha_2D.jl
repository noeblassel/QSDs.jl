using  PyPlot,Triangulate,LinearAlgebra

include("QSDs.jl")
include("GeometryUtils.jl")

using .QSDs, .GeometryUtils

Lx = Ly=2.0
cx,cy = -1.0,-1.0

Nx = 40
Ny = 40

max_area = 0.003
min_angle = 20

core_sets = [-0.5 0.5 ; -0.25 0.25; 0.1 0.1]
n_core_set_boundary = [6,6]

triout, periodic_images , core_set_ixs = conforming_triangulation(cx,cy,Lx,Ly,Nx,Ny,core_sets,n_core_set_boundary,max_area,min_angle)
dirichlet_boundary_points = core_set_ixs[1]
V(x,y)= (sin(π*2x+π/2)+sin(π*(y-x/2)-π/2))
β=0.6
mu(x,y) = exp(-β*V(x,y))

N = numberofpoints(triout)
X = triout.pointlist[1,:]
Y= triout.pointlist[2,:]
mus = mu.(X,Y)
t=triout.trianglelist

M,B,δM,∂λ = build_FEM_matrices_2D(V,β,triout)

Mred,Bred=apply_bc(M,B,periodic_images,dirichlet_boundary_points)
λs,us_red=eigen(Mred,Bred)

u1= reverse_bc(us_red[:,1],N,periodic_images,dirichlet_boundary_points)
qsd = qsd_2d(u1,V,β,triout)
lg_qsd=log.(qsd)
clamp!(lg_qsd,-6,Inf)

tricontourf(X,Y,transpose(t).-1,qsd,cmap=:hsv)