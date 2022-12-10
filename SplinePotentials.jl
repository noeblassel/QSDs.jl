module SplinePotentials
    export spline_potential_derivatives

    h(t) = t^2*(3-2t) # unit spline
    dh(t) = 6t*(1-t) # derivative
    d2h(t) = 6-12t  #second derivative

    function cubic_spline(xa,ya,xb,yb)
        f(q) = ya + (yb - ya) * h( (q - xa) / (xb - xa) )
        return f
    end

    function grad_cubic_spline(xa,ya,xb,yb)
        df(q) = (yb - ya) * dh( (q - xa) / (xb -xa) ) / (xb - xa)
        return df 
    end

    function grad_grad_cubic_spline(xa,ya,xb,yb)
        d2f(q) = (yb - ya) * d2h( (q - xa) / (xb -xa) ) / (xb - xa)^2
        return d2f
    end

    """
        Constructs a periodic 1D potential with given critical points and values,
        with cubic spline interpolations, returning the potential and its two first derivatives.

        L is the length of the periodic domain R/LZ. 
        We assume the critical_points are distinct, given in order, and represented in the same cell.
        i.e. (last(extrema)-first(extrema) < L)
    """
    function spline_potential_derivatives(critical_points::T,heights::T ,L) where {T <: AbstractVector{S} where {S<:Real}}
        n = length(critical_points)
        @assert n == length(heights) "critical_points and heights have different lengths"

        aug_critical_points = [critical_points[n]-L,critical_points...,critical_points[1]+L] # augment critical points with periodic images
        aug_heights = [heights[n], heights...,heights[1]] # augment with corresponding heights

        splines = [cubic_spline(aug_critical_points[i],aug_heights[i],aug_critical_points[i+1],aug_heights[i+1]) for i=1:n+1]
        grads = [grad_cubic_spline(aug_critical_points[i],aug_heights[i],aug_critical_points[i+1],aug_heights[i+1]) for i=1:n+1]
        grad_grads = [grad_grad_cubic_spline(aug_critical_points[i],aug_heights[i],aug_critical_points[i+1],aug_heights[i+1]) for i=1:n+1]
        
        ix(q) = last(i for i=1:n+1 if q >= aug_critical_points[i])

        V(q) = splines[ix(q)](q)
        ∂V(q) = grads[ix(q)](q)
        ∂∂V(q) = grad_grads[ix(q)](q)

        return V,∂V,∂∂V
    end

end

