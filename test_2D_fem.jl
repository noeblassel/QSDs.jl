using   Plots,Triangulate,LinearAlgebra,TriplotRecipes

include("QSDs.jl")

using .QSDs

Nin = 100
t = range(0,2π,Nin)
t = t[1:Nin-1]
x = 16sin.(t).^3
y = 13cos.(t) - 5cos.(2t) - 2cos.(3t) - cos.(4t)

triin = TriangulateIO()

points=transpose(hcat(x,y))
triin.pointlist = points
i = collect(Int32,1:Nin-1)
push!(i,Int32(1))
edges = transpose(hcat(i[1:Nin-1],i[2:Nin]))
triin.segmentlist = edges

max_area = 0.1
min_angle = 20

(triout, vorout) = triangulate("pq$(min_angle)Da$(max_area)",triin)

#V(x,y)= 0.1*(x^4+2y^4) + 0.01((x-1)^4 + (y-2)^4)
#V(x,y)= sin(x) +3cos(2y)
V(x,y)=0
β=4.0
N = numberofpoints(triout)
dirichlet_boundary = 1:49
Ω = setdiff(1:N,dirichlet_boundary)
N = numberofpoints(triout)
M,B = QSDs.build_FEM_matrices_2D(V,β,triout)
Mₒ=M[Ω,Ω]
Bₒ=B[Ω,Ω]
λs,us=eigen(Mₒ,Bₒ)

Z = zeros(N)
Z[Ω] .= us[:,29]

Vs= V.(x,y)
X = triout.pointlist[1,:]
Y= triout.pointlist[2,:]
t=triout.trianglelist
tripcolor(X,Y,Z,t,aspectratio=1,size=(1000,1000),colorbar=:false,showaxis=false,showgrid=false,cmap=:hsv)


