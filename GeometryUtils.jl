module GeometryUtils
    using Triangulate

    export conforming_triangulation, dirichlet_triangulation

    """
    Build a triangulation of a given rectangular domain,
    conforming to additional circular inner boundaries given by the core_set argument.
    core_set should be a vector of curves (functions γ(t)=[x(t),y(t)]) defined on [0,1]. It is assumed γ(0)=γ(1)
    cx,cy are the coordinates of the bottom-left corner of the rectangular domain, Lx, Ly are the side lengths, and Nx, Ny are the number of boundary points on the horizontal and vertical outer boundaries respectively.
    n_core_set_boundary gives the number of boundary points for each core set.
    core_set_tests is a vector of boolean-valued functions T(x,y) returning true if (x,y) belongs to the corresponding coreset.
    The function returns the triangulation, an array of index pairs which are related by periodicity, as well as a vector of vectors of indices, giving for each core set the index of points lying on or within its boundary.
    """
    function conforming_triangulation(cx,cy,Lx,Ly,Nx,Ny,core_sets,core_set_tests,n_core_set_boundary,max_area,min_angle=20;quiet=true)
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

        k = length(core_sets)

        core_set_ixs=[Cint[] for i=1:k]
        
        for i=1:k
            γ = core_sets[i]
            n = n_core_set_boundary[i]
            t = range(0,1,n+1)

            P=hcat(γ.(t[1:n])...)

            (_,Npts) = size(boundary_points)

            ixs = collect(Cint,Npts+1:Npts+n)
            push!(ixs ,Npts+1)
            boundary_points = hcat(boundary_points,P)
            boundary_edges = hcat(boundary_edges,transpose(hcat(ixs[1:n],ixs[2:n+1])))
            append!(core_set_ixs[i],ixs)
        end

        triin = TriangulateIO()
        
        triin.pointlist = boundary_points
        triin.segmentlist = boundary_edges
        flags = (quiet) ? "pq$(min_angle)Da$(max_area)QY" : "pq$(min_angle)Da$(max_area)Y"
        (triout,vorout) = triangulate(flags,triin)

        for ix=1:numberofpoints(triout)
            x,y=triout.pointlist[:,ix]
            for i=1:k
                if core_set_tests[i](x,y)
                    push!(core_set_ixs[i],ix)
                end
            end

        end
        core_set_ixs = unique!.(core_set_ixs)
        return triout,periodic_images,core_set_ixs
    end

    """
    Computes a triangulation of the domain define by the enclosing curve γ defined on [0,1], with Npts dirichlet boundary points on γ([0,1]).
    returns the triangulation, the indices of dirichlet boundary points, as well as a
    3 x Npts matrix boundary_triangles of indices of points of the triangulation, where
    boundary_triangles[1,k], boundary_triangles[2,k] are the points on the boundary and boundary_triangles[3,k] are the inner points.
    The ordering corresponds to a clockwise orientation.
    """

    function dirichlet_triangulation(γ,Npts,max_area,min_angle=20;quiet=true)
        t_range = range(0,1,Npts+1)
        
        triin = TriangulateIO()
        points_in = hcat(γ.(t_range[1:Npts])...)
        triin.pointlist = points_in
        ixs = collect(Cint,1:Npts)
        push!(ixs,1)
        edges_in = transpose(hcat(ixs[1:Npts],ixs[2:Npts+1]))
        triin.segmentlist = edges_in

        dirichlet_boundary_points = ixs[1:Npts]
        flags = (quiet) ? "pq$(min_angle)Da$(max_area)QY" : "pq$(min_angle)Da$(max_area)Y"
        (triout,vorout) = triangulate(flags,triin)

        Ntri = numberoftriangles(triout)

        boundary_triangles = Vector{Cint}[]

        for n=1:Ntri
            i,j,k = triout.trianglelist[:,n]

            if (i<=Npts) && (j<=Npts)
                push!(boundary_triangles,[i,j,k])
            elseif (j<=Npts) && (k<=Npts)
                push!(boundary_triangles,[j,k,i])
            elseif (k<=Npts) && (i<=Npts)
                push!(boundary_triangles,[k,i,j])
            end

        end

        return triout, dirichlet_boundary_points, hcat(boundary_triangles...)
    end

end