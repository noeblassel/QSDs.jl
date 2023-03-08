#!/libre/blasseln/julia-1.8.2/bin/julia

include("QSDs.jl")
include("GeometryUtils.jl")

using .QSDs, .GeometryUtils, Triangulate, Arpack, ProgressMeter

β = parse(Float64,ARGS[1])

output_file = ARGS[2]

Nx = 200
Ny = 200

N_coreset_boundary_points = 50

max_area = 0.0001

println("Usage: β output_file Nx Ny N_coreset_boundary_points")

V(x,y)= cos(2π*x)-cos(2π*(y-x))

r_range = 0.1:0.001:0.3
Nr=length(r_range)
λ1s = zeros(Nr,Nr,Nr)
λ2s = zeros(Nr,Nr,Nr)

r3 = 0.3

n_core_set_boundary = fill(N_coreset_boundary_points,4)

@showprogress for (i1,r1)=enumerate(r_range)
    @showprogress for (i2,r2)=enumerate(r_range)
        @showprogress for (i3,r4)=enumerate(r_range)

            minima = [(-0.5,-0.5,r1),(-0.5,0.5,r2),(0.5,-0.5,r3),(0.5,0.5,r4)]
            core_sets = [t->[xm + rm*cos(2π*t),ym + rm*(sin(2π*t)+cos(2π*t))] for (xm,ym,rm)=minima]
            core_set_tests = [(x,y)->((x-xm)^2+((x-xm)-(y-ym))^2 < rm^2) for (xm,ym,rm)=minima]


            Lx = Ly=2.0
            cx,cy = -1.0,-1.0


            min_angle = 20
            triout, periodic_images , core_set_ixs = conforming_triangulation(cx,cy,Lx,Ly,Nx,Ny,core_sets,core_set_tests,n_core_set_boundary,max_area,min_angle,quiet=true)
            Ntri = numberoftriangles(triout)
            N = numberofpoints(triout)
            Ωᶜ = vcat(core_set_ixs[1],core_set_ixs[2],core_set_ixs[4])

            M,B,δM,∂λ = build_FEM_matrices_2D(V,β,triout)

            M,B = apply_bc(M,B,periodic_images,Ωᶜ)

            λs,us = eigs(M,B,nev=2,which=:LR,sigma=0.0)

            λ1,λ2 = real.(λs)

            λ1s[i1,i2,i3] = λ1
            λ2s[i1,i2,i3] = λ2

            println("r1 = $r1 ; r2 = $r2 ; r4 = $r4")
        end
    end
end

f = open(output_file,"w")
println(f,"λ1s=",λ1s)
println(f,"λ2s=",λ2s)
close(f)