using   Plots,Triangulate,LinearAlgebra,TriplotRecipes, Arpack

include("QSDs.jl")
include("GeometryUtils.jl")

using .QSDs, .GeometryUtils

Nin = 1000
t = range(0,2π,Nin+1)
t = t[1:Nin]
#x = 16sin.(t).^3
#y = 13cos.(t) - 5cos.(2t) - 2cos.(3t) - cos.(4t)

x = 16cos.(-t)
y = 16sin.(-t)

γ(t) = [16*cos(2π*t),16*sin(2π*t)]

triout, dirichlet_boundary_points,boundary_triangles = dirichlet_triangulation(γ,Nin,0.1)

V(x,y)=0

β=4.0
N = numberofpoints(triout)
dirichlet_boundary = 1:Nin
Ω = setdiff(1:N,dirichlet_boundary)
N = numberofpoints(triout)
M,B = QSDs.build_FEM_matrices_2D(V,β,triout)
Mₒ=M[Ω,Ω]
Bₒ=B[Ω,Ω]

λs,us=eigs(Mₒ,Bₒ,nev=20,sigma=0,which=:LR)
λs=real.(λs)
us=real.(us)

Z = zeros(N)
Z[Ω] .= us[:,2]

Vs= V.(x,y)
X = triout.pointlist[1,:]
Y= triout.pointlist[2,:]
t=triout.trianglelist

η = QSDs.sq_normal_derivatives(Z,triout,boundary_triangles)
plot(η[sortperm(boundary_triangles[1,:])])
