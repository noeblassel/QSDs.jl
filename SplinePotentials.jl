module SplinePotentials
    export spline_potential_derivatives, quintic_spline_potential_derivatives

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

    function quintic_segment_derivatives(x₁,x₂,y₁,y₂,α) # p′′(x₂)/p′′(x₁) = α
        A = [1 x₁ x₁^2 x₁^3 x₁^4;
             1 x₂ x₂^2 x₂^3 x₂^4;
             0 1 2x₁ 3x₁^2 4x₁^3;
             0 1 2x₂ 3x₂^2 4x₂^3;
             0 0 2*(α-1) 6*(α*x₁-x₂) 12*(α*x₁^2-x₂^2)]

        y = [y₁,y₂,0,0,0]

        a,b,c,d,e = A\y

        p(x) = a + b*x + c*x^2 +d*x^3 + e*x^4
        p′(x) = b +2c*x +3d*x^2 + 4e*x^3
        p′′(x) = 2c + 6d*x + 12e*x^2

        return p,p′,p′′

    end

    function quintic_spline_potential_derivatives(zs,Vs,κs)
        n = length(zs)
        @assert n == length(Vs) == length(κs) "critical_points and heights have different lengths"

        spline_ders = [quintic_segment_derivatives(x₁,x₂,y₁,y₂,κ₂/κ₁) for (x₁,x₂,y₁,y₂,κ₁,κ₂)=zip(zs[1:n-1],zs[2:n],Vs[1:n-1],Vs[2:n],κs[1:n-1],κs[2:n])]
        splines = [spline[1] for spline in spline_ders]
        grads = [spline[2] for spline in spline_ders]
        grad_grads=[spline[3] for spline in spline_ders]

        ix(q) = last(i for i=1:n-1 if q >= zs[i])

        V(q) = splines[ix(q)](q)
        ∂V(q) = grads[ix(q)](q)
        ∂∂V(q) = grad_grads[ix(q)](q)

        return V,∂V,∂∂V
    end

end

