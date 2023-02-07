module PlotUtils

export to_vertex_function

    using Triangulate, Plots, Colors


    """
    Converts a function defined on cells to a function defined on vertices
    """
    function to_vertex_function(α,domain::TriangulateIO)

        Ntri=numberoftriangles(domain)
        N=numberofpoints(domain)
        z = zeros(N)
        weights = zeros(N)

        for n=1:Ntri
            (i,j,k) = domain.trianglelist[:,n]
            xi,xj,xk = domain.pointlist[1,[i,j,k]]
            yi,yj,yk = domain.pointlist[2,[i,j,k]]

            cx = (xi+xj+xk)/3
            cy = (yi+yj+yk)/3

            di = sqrt((xi-cx)^2+(yi-cy)^2)
            dj = sqrt((xj-cx)^2+(yj-cy)^2)
            dk = sqrt((xk-cx)^2+(yk-cy)^2)

            z[i] += α[n]*di
            z[j] += α[n]*dj
            z[k] += α[n]*dk

            weights[i] += di
            weights[j] += dj
            weights[k] += dk
        end

        return z ./ weights
    end

end