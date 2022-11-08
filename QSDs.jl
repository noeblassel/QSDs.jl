function P1_element(qi_m1::Float64,qi::Float64,qi_p1::Float64)
    function P1(q::Float64)
        if qi_m1 < q < qi_p1
            return (q < qi) ? 1 - ( qi - q ) / ( qi - qi_m1 ) : 1 - ( q - qi ) /  ( qi_p1 - qi )
        else
            return 0.0
        end
    end
    return P1
end

function boundary_P1_element(q_m1::Float64,q_periodic_right::Float64,q_periodic_left::Float64,q_p1::Float64)
    function P1_diff(q::Float64)
        if q_m1 < q < q_periodic_right
            return 1 - (q_periodic_right-q)/(q_periodic_right - q_m1)
        elseif q_periodic_left < q < q_p1
            return 1 - (q-q_periodic_left)/(q_p1-q_periodic_left)
        else
            return 0.0
        end
    end
end

function P1_element_diff(qi_m1::Float64,qi::Float64,qi_p1::Float64)
    function P1_diff(q::Float64)
        if qi_m1 < q < qi_p1
            return (q < qi) ? inv( qi - qi_m1 ) : -inv( qi_p1 - qi )
        else
            return 0.0
        end
    end
    return P1_diff
end

function boundary_P1_element_diff(q_m1::Float64,q_periodic_right::Float64,q_periodic_left::Float64,q_p1::Float64)
    function P1_diff(q::Float64)
        if q_m1 < q < q_periodic_right
            return inv(q_periodic_right-q_m1)
        elseif q_periodic_left < q < q_p1
            return - inv(q_p1 - q_periodic_left)
        else
            return 0.0
        end
    end
end


"""
Precompute weights for the M and δM matrices in the SQSD_1D_FEM function, to avoid redudant quadrature calculations.

Arguments:  mu::Function, the invariant measure
            D::Vector{Float64}, the mesh points. The last element of D is the periodic image of the first element.
"""
function calc_weights_periodic(mu::Function,D::Vector{Float64})

    N = length(D)-1 # D= [q_1, ... , q_{N}, q_{N+1}] with q_1 ≅ q_{N+1}
    elements = [P1_element(D[i],D[i+1],D[i+2]) for i=1:N-1]
    boundary_element = boundary_P1_element(D[N],1.0,0.0,D[2]) # P1 element straddling the boundary for periodic boundary conditions

    elements_diff = [P1_element_diff(D[i],D[i+1],D[i+2]) for i=1:N-1]
    boundary_element_diff = boundary_P1_element_diff(D[N],1.0,0.0,D[2])

    diag_weights_diff = zeros(N) # ⟨∇ϕ_i,∇ϕ_i⟩μ on each of the [q_{i},q_{i+2}]
    off_diag_weights_diff = zeros(N) # ⟨∇ϕ_i, ∇ϕ_{i+1}⟩μ on each of the [q_{i+1},q_{i+2}]

    diag_weights = zeros(2N) # ∫ϕ_i^2 dμ on each of the [q_i,q_{i+1}] and [q_{i+1},q_{i+2}]
    off_diag_weights = zeros(N) # ∫ ϕ_i ϕ_{i+1}dμ on each of the [q_{i+1},q_{i+2}]

    # diagonal terms
    for i=1:N-1
        diag_weights_diff[i],err = hquadrature(q->mu(q)*elements_diff[i](q)^2,D[i],D[i+2])
    end

    # periodic boundary diagonal term
    inta,_ = hquadrature(q->mu(q)*boundary_element_diff(q)^2,D[N],D[N+1])
    intb,_ = hquadrature(q->mu(q)*boundary_element_diff(q)^2,D[1],D[2])
    diag_weights_diff[N] = inta + intb

    for i=1:N-2
        off_diag_weights_diff[i],_= hquadrature(q->mu(q)*elements_diff[i](q)*elements_diff[i+1](q),D[i+1],D[i+2])
    end

    # periodic boundary off-diagonal terms
    off_diag_weights_diff[N-1],_ = hquadrature(q->mu(q)*boundary_element_diff(q)*elements_diff[N-1](q),D[N],D[N+1])
    off_diag_weights_diff[N],_ = hquadrature(q->mu(q)*boundary_element_diff(q)*elements_diff[1](q),D[1],D[2])

    for i=1:N-1
        diag_weights[2i-1],_ = hquadrature(q->mu(q)*elements[i](q)^2,D[i],D[i+1])
        diag_weights[2i],_ = hquadrature(q->mu(q)*elements[i](q)^2,D[i+1],D[i+2])
    end

    diag_weights[2N-1],_ = hquadrature(q->mu(q)*boundary_element(q)^2,D[N],D[N+1])
    diag_weights[2N],_ = hquadrature(q->mu(q)*boundary_element(q)^2,D[1],D[2])

    for i=1:N-2
        off_diag_weights[i],_ = hquadrature(q->mu(q)*elements[i](q)*elements[i+1](q),D[i+1],D[i+2])
    end

    off_diag_weights[N-1],_ = hquadrature(q->mu(q)*elements[N-1](q)*boundary_element(q),D[N],D[N+1])
    off_diag_weights[N],_ = hquadrature(q->mu(q)*elements[1](q)*boundary_element(q),D[1],D[2])

    return diag_weights_diff,off_diag_weights_diff,diag_weights,off_diag_weights
end

function QSD_1D_FEM(V::Function, β::Float64,Ωl::Float64,Ωr::Float64,N::Int64) #requires Cubature and LinearAlgebra
    mu(q)=exp(-β*V(q))
    h=(Ωr-Ωl)/N

    #divide the domain Ω into regular subdomains
    # q_1 = Ωl
    #   Ω[i] = [q_i,q_i +h]
    Ω=collect(Float64,range(Ωl,Ωr,N))

    #= 
        Approximate the generator in weak form acting on the Sobolev space H_0^1(Ω) by a Galerkin approximation procedure.
        This is done by computing the action of the bilinear form 
                (u,v) -> ⟨ Lu,v ⟩μ = -1/β ⟨ ∇u, ∇v ⟩μ,
        where the brackets ⟨⋅,⋅⟩μ denote μ-weighted inner products on Ω, 
        on the (N-2)-dimensional subspace generated by the basis functions
            ϕ_i(q)= (1- |q-qi|/h)⁺
        Computing
            Mij= 1/β ⟨ ∇ϕ_i, ∇ϕ_j ⟩μ,
        this gives a symmetric positive-definite matrix whose first eigenvector is an approximation to the unnormalized QSD.
        The coefficients Mij can be computed easily in terms of the 
            μ([q_i,q_i+h]), 1 ≤ i ≤ N-1
     =#

     # Compute the weights, 
     #       weights[i]= μ([q_i,q_i+h])
    
    elements_diff=[P1_element_diff(Ω[i],Ω[i+1],Ω[i+2]) for i=1:N-2]

    # Compute M -- in fact we compute an unnormalized version of M, since we only care about ratios of eigenvalues
    M=zeros(N-2,N-2)

    for i=1:N-2 #diagonal terms
        M[i,i],_ = hquadrature(q->mu(q)*elements_diff[i](q)^2,Ω[i],Ω[i+2])
    end

    for i=2:N-2 #off-diagonal terms
        M[i,i-1],_ = M[i-1,i],_ = hquadrature(q->mu(q)*elements_diff[i](q)*elements_diff[i-1](q),Ω[i-1],Ω[i+2])
    end

    # Note M has zero-sum rows, so that it actually is the generator of a discrete-state jump process
    M = SymTridiagonal(M)
    
    λs,us=eigen(M,1:2)
    #add Dirichlet boundary conditions

    u1=us[:,1]
    push!(u1,0.0) 
    pushfirst!(u1,0.0)
    
    qsd = u1 .* mu.(Ω)
    I = h*sum(qsd)
    qsd ./= I
    return λs,qsd,us
end

QSD_1D_FEM(V, β, domain::Tuple{Float64,Float64},N) = QSD_1D(V,β,domain[1],domain[2],N)

"""
Computes the QSD for the Overdamped Langevin dynamics in a 1D potential with soft killing, using a finite-element method on a regular mesh
Arguments: V, the potential function
           β, the inverse temperature
           α, a vector giving the killing rate on each of the cells of the discretized domain
Keyword argument: weights, optional. The result of calc_weights_periodic(V,β,D) where D is the domain. Allows to recycle quadrature computations.
"""
function SQSD_1D_FEM(V::Function,β::Float64,α::Vector{Float64}; weights = nothing)

    mu_tilde(q)=exp(-β*V(q))
    Z,_=hquadrature(mu_tilde,0,1)
    mu(q)=mu_tilde(q)/Z
    N=length(α)
    h=1/N
    D=collect(Float64,range(0,1,N+1))
    
    if weights === nothing
        diag_weights_diff,off_diag_weights_diff,diag_weights,off_diag_weights = calc_weights_periodic(mu,D)
    else
        diag_weights_diff,off_diag_weights_diff,diag_weights,off_diag_weights = weights
    end

    # FEM discretization of (negative) Fokker-Planck generator, up to a constant

    M=zeros(N,N) 

    for i=1:N
        j = (i == N) ? 1 : i+1
        M[i,i] = diag_weights_diff[i]
        M[i,j] = M[j,i]= off_diag_weights_diff[i]
    end

    M /= β

    # FEM discretization of (negative) Soft-Measure perturbation -- assuming α is constant on each subinterval of the domain

    δM = zeros(N,N)

    for i=1:N
        j = (i == N) ? 1 : i+1
        δM[i,i] = α[i]*diag_weights[2i-1] + α[j]*diag_weights[2i]
        δM[i,j] = δM[j,i] = α[j] * off_diag_weights[i]
    end

    # diagonalize discretized operator

    H = Symmetric(M + δM)
    λs,us=eigen(H,1:2)
    u1=us[:,1]

    qsd = u1 .* mu.(D[2:N+1])

    I= h * sum(qsd)
    qsd /= I
    
    return λs,qsd,us,H
end