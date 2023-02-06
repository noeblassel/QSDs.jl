module GeometryUtils
    using Triangulate

    export conforming_triangulation

    """
    Build a triangulation of a given rectangular domain,
    conforming to additional circular inner boundaries given by the core_set argument.
    core_set should be a (3 x k) matrix where k is the number of core sets. 
    The first line corresponds to the x coordinates of the core set centers, the second the y coordinates, and the third the radii of the core sets.
    cx,cy are the coordinates of the bottom-left corner of the rectangular domain, Lx, Ly are the side lengths, and Nx, Ny are the number of boundary points on the horizontal and vertical outer boundaries respectively.
    n_core_set_boundary gives the number of boundary points for each core set.
    The function returns the triangulation, an array of index pairs which are related by periodicity, as well as a vector of vectors of indices, giving for each core set the index of points lying on or within its boundary.
    """
    function conforming_triangulation(cx,cy,Lx,Ly,Nx,Ny,core_sets,n_core_set_boundary,max_area,min_angle=20,quiet=true)
        Rx = range(cx,cx+Lx,Nx)
        Ry=range(cy,cy+Ly,Ny)

        x = vcat(Rx[1:Nx-1],repeat([cx+Lx],Ny-1),reverse(Rx)[1:Nx-1],repeat([cx],Ny-1))
        y = vcat(repeat([cy],Nx-1),Ry[1:Ny-1],repeat([cy+Ly],Nx-1),reverse(Ry)[1:Ny-1])

        boundary_points=transpose(hcat(x,y))

        Nin = 2Nx+2Ny-4
        i = collect(Cint,1:Nin)
        push!(i,1)
        boundary_edges = transpose(hcat(i[1:Nin],i[2:Nin+1]))

        periodic_images = zeros(Cint,2,Nx+Ny-1)

        for (i,t)=enumerate(zip(Nx+1:Nx+Ny-2,2Nx+2Ny-4:-1:2Nx+Ny-1)) # map right to left
            j,k = t
            periodic_images[1,i] = j
            periodic_images[2,i] = k
        end

        for (i,t)=enumerate(zip(Nx+Ny:2Nx+Ny-3,Nx-1:-1:2)) # map top to bottom
            j,k = t
            periodic_images[1,Ny-2+i] = j
            periodic_images[2,Ny-2+i] = k
        end

        ## map corners
        periodic_images[1,Nx+Ny-3] = Nx
        periodic_images[1,Nx+Ny-2] = Nx+Ny-1
        periodic_images[1,Nx+Ny-1] = 2Nx+Ny-2

        periodic_images[2,Nx+Ny-3:Nx+Ny-1] .= 1

        (_,k) = size(core_sets)

        core_set_ixs=[Cint[] for i=1:k]
        
        for ix=1:k
            x0,y0,r = core_sets[:,ix]

            n = n_core_set_boundary[ix]
            t = range(0,2π,n+1)
            x =  x0 .+ r*cos.(t[1:n])
            y = y0 .+ r*sin.(t[1:n])

            i= collect(Cint,1:n)
            push!(i,1)
            (_,Npts) = size(boundary_points)
            boundary_points = hcat(boundary_points,transpose(hcat(x,y)))
            boundary_edges = hcat(boundary_edges,transpose(Npts .+ hcat(i[1:n],i[2:n+1])))
            append!(core_set_ixs[k],Npts .+ 1:n)
        end

        triin = TriangulateIO()
        
        triin.pointlist = boundary_points
        triin.segmentlist = boundary_edges
        flags = (quiet) ? "pq$(min_angle)Da$(max_area)QY" : "pq$(min_angle)Da$(max_area)Y"
        (triout,vorout) = triangulate(flags,triin)

        for ix=1:numberofpoints(triout)
            x,y=triout.pointlist[:,ix]
            
            for j=1:k
                x0,y0,r = core_sets[:,j]
                if (x-x0)^2 +(y-y0)^2 < r^2
                    push!(core_set_ixs[j],ix)
                end
            end
        end
        
        core_set_ixs = unique!.(core_set_ixs)

        return triout,periodic_images,core_set_ixs
    end

end