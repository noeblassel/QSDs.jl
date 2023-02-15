using   Plots,Triangulate,LinearAlgebra,TriplotRecipes, Arpack

include("QSDs.jl")

using .QSDs

Nin = 40
t = range(0,2π,Nin+1)
t = t[1:Nin]
#x = 16sin.(t).^3
#y = 13cos.(t) - 5cos.(2t) - 2cos.(3t) - cos.(4t)

x = 16sin.(t)
y = 16cos.(t)

triin = TriangulateIO()

points=transpose(hcat(x,y))
triin.pointlist = points
i = collect(Int32,1:Nin)
push!(i,Int32(1))
edges = transpose(hcat(i[1:Nin],i[2:Nin+1]))
triin.segmentlist = edges

max_area = 0.5
min_angle = 20

(triout, vorout) = triangulate("pq$(min_angle)Da$(max_area)",triin)

#V(x,y)= 0.1*(x^4+2y^4) + 0.01((x-1)^4 + (y-2)^4)
#V(x,y)= sin(x) +3cos(2y)
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
Z[Ω] .= us[:,11]

Vs= V.(x,y)
X = triout.pointlist[1,:]
Y= triout.pointlist[2,:]
t=triout.trianglelist
tripcolor(X,Y,Z,t,aspectratio=1,size=(1000,1000),colorbar=:false,showaxis=false,showgrid=false,cmap=:hsv)
