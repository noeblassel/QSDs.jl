using   Plots,Triangulate,LinearAlgebra,TriplotRecipes

include("QSDs.jl")

using .QSDs

Nin = 100
t = range(0,2π,Nin)
t = t[1:Nin-1]
x = 4*cos.(t) +1.2*sin.(2t)
y = 4*sin.(t) -0.8*cos.(3t)

triin = TriangulateIO()

points=transpose(hcat(x,y))
triin.pointlist = points
i = collect(Int32,1:Nin-1)
push!(i,Int32(1))
edges = transpose(hcat(i[1:Nin-1],i[2:Nin]))
triin.segmentlist = edges

max_area = 0.01
min_angle = 25

(triout, vorout) = triangulate("pq$(min_angle)Da$(max_area)",triin)

#V(x,y)= 0.1*(x^4+2y^4) + 0.01((x-1)^4 + (y-2)^4)
#V(x,y)= sin(x) +3cos(2y)
V(x,y)=0
β=2.0

dirichlet_boundary = 1:49
N = numberofpoints(triout)
M,B = QSDs.build_FEM_matrices_2D(V,β,triout,dirichlet_boundary)

λs,us = eigen(M,B)

#λs,us = eigen(M,B)

x = triout.pointlist[1,:]
y = triout.pointlist[2,:]

Ω = setdiff(1:N,dirichlet_boundary)
z = zeros(N)
z[Ω] .= us[:,20]

Vs= V.(x,y)

t=triout.trianglelist
tripcolor(x,y,z,t,aspectratio=1,size=(1000,1000),colorbar=:false,showaxis=:false)


