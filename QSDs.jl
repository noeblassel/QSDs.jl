module QSDs
export calc_weights_periodic,
        calc_weights_schrodinger_periodic,
        QSD_1D_FEM,
        QSD_1D_FEM_schrodinger,
        SQSD_1D_FEM,
        SQSD_1D_FEM_schrodinger,
        build_FEM_matrices_1D

    using Cubature, LinearAlgebra, SparseArrays, Arpack, TensorOperations

    """
    Precompute weights for the M and δM matrices in the SQSD_1D_FEM function, to avoid redudant quadrature calculations.

    Arguments:  mu::Function, the invariant measure
                D::Vector{Float64}, the mesh points. The last element of D is the periodic image of the first element.
    """
    function calc_weights_periodic(mu::Function,D::AbstractVector{Float64})
        N = length(D)-1 # D= [q_1, ... , q_{N}, q_{N+1}] with q_1 ≅ q_{N+1}
        diag_weights_diff = zeros(N) # ⟨∇ϕ_i,∇ϕ_i⟩μ on each of the [q_{i},q_{i+2}]
        off_diag_weights_diff = zeros(N) # ⟨∇ϕ_i, ∇ϕ_{i+1}⟩μ on each of the [q_{i+1},q_{i+2}]

        diag_weights = zeros(2N) # ∫ϕ_i^2 dμ on each of the [q_i,q_{i+1}] and [q_{i+1},q_{i+2}]
        off_diag_weights = zeros(N) # ∫ ϕ_i ϕ_{i+1}dμ on each of the [q_{i+1},q_{i+2}]
        
        for i=1:N
            if i<N
                a=D[i]
                b=D[i+1]
                c=D[i+1]
                d=D[i+2]
            else
                a=D[N]
                b=D[N+1]
                c=D[1]
                d=D[2]
            end

            f1(t) = mu(a+t*(b-a))
            f2(t) = mu(c + t*(d-c))
            f3(t) = f1(t)*t^2
            f4(t) = f2(t)*(1-t)^2
            f5(t) = f2(t)*t*(1-t)

            int_1,_ = hquadrature(f1,0,1) 
            int_2,_ = hquadrature(f2,0,1) 
            int_3,_ = hquadrature(f3,0,1)
            int_4,_ = hquadrature(f4,0,1)
            int_5,_ = hquadrature(f5,0,1)

            diag_weights_diff[i] = int_1 / (b-a) + int_2 / (d-c)
            diag_weights[2i-1] = int_3 * (b-a)
            diag_weights[2i] = int_4 * (d-c)

            off_diag_weights_diff[i] = -int_2 / (d-c)
            off_diag_weights[i] = int_5 * (d-c)

        end   

        return diag_weights_diff,off_diag_weights_diff,diag_weights,off_diag_weights
    end

    function calc_weights_schrodinger_periodic(W::Function, D::AbstractVector{Float64})
        N = length(D)-1 # D= [q_1, ... , q_{N}, q_{N+1}] with q_1 ≅ q_{N+1}

        diag_weights_diff = zeros(N) # ∫ ∇ϕ_i⋅∇ϕ_i on each of the [q_{i},q_{i+2}]
        off_diag_weights_diff = zeros(N) # ∫∇ϕ_i⋅∇ϕ_{i+1}⟩ on each of the [q_{i+1},q_{i+2}]

        diag_weights_pot = zeros(N) # ∫W ϕ_i^2 on each of the [q_i,q_{i+2}]
        off_diag_weights_pot = zeros(N) # ∫W ϕ_i ϕ_{i+1}dμ on each of the [q_{i+1},q_{i+2}]

        diag_weights = zeros(2N) # ∫ ϕ_i ϕ_i on each of the [q_{i},q_{i+1}] and [q_{i+1},q_{i+2}]
        off_diag_weights = zeros(N) # ∫ ϕ_i ϕ_{i+1} on each of the [q_{i+1},q_{i+2}] 

        for i=1:N
            if i<N
                a=D[i]
                b=D[i+1]
                c=D[i+1]
                d=D[i+2]
            else
                a=D[N]
                b=D[N+1]
                c=D[1]
                d=D[2]
            end

            f1(t) = W(a+t*(b-a))*t^2
            f2(t) = W(c + t*(d-c))*(1-t)^2
            f3(t) = W(c + t*(d-c))*t*(1-t)

            int_1,_ = hquadrature(f1,0,1) 
            int_2,_ = hquadrature(f2,0,1) 
            int_3,_ = hquadrature(f3,0,1)

            diag_weights_diff[i] = 1 / (b-a) + 1 / (d-c)

            diag_weights_pot[i] = int_1 * (b-a) + int_2 * (d-c)

            diag_weights[2i-1] = (b-a) / 3
            diag_weights[2i] = (d-c) / 3

            off_diag_weights_diff[i] = - 1 / (d-c)
            off_diag_weights_pot[i] = int_3 * (d-c)
            off_diag_weights[i] = (d-c) / 6

        end   

        return diag_weights_diff, off_diag_weights_diff, diag_weights_pot, off_diag_weights_pot, diag_weights, off_diag_weights
    end

    function QSD_1D_FEM(mu::Function, β::Float64,Ω::AbstractVector{Float64};weights=nothing) #requires Cubature and LinearAlgebra
        N=length(Ω)-1

        if weights===nothing
            diag_weights_diff,off_diag_weights_diff,diag_weights,off_diag_weights = calc_weights_periodic(mu,Ω)
        else
            diag_weights_diff,off_diag_weights_diff,diag_weights,off_diag_weights = weights
        end

        M=zeros(N-1,N-1)
        B=zeros(N-1,N-1)
        for i=1:N-1 #diagonal terms
            B[i,i] = diag_weights[2i-1] + diag_weights[2i]
            M[i,i] = diag_weights_diff[i]
        end

        for i=1:N-2 #off-diagonal terms
            B[i,i+1] = B[i+1,i] = off_diag_weights[i]
            M[i,i+1] = M[i+1,i] = off_diag_weights_diff[i]
        end

        M ./= β
        M = Symmetric(M)
        B = Symmetric(B)
        
        λs,us=eigen(M,B)

        return λs[1:2],us[:,1:2]
    end

    """ 
    Build FEM matrices associated with the generator of the Overdamped Langevin dynamics on the periodic domain with piecewise constant killing rate α on each cell. 
    Domain is an abstract vector [q_1,…, q_{N+1}] where q_{N+1} and q_1 are identified with each other.
    """
    function build_FEM_matrices_1D(V::Function,β::S,domain::AbstractVector{T}) where {S<:Real,T<:Real,U<:Real}
        N=length(domain)-1

        M=zeros(N,N)
        B=zeros(N,N)

        mus = exp.(-β*V.(domain))
    
        for i=1:N-1
            M[i,i] = inv(2β*(domain[i+1]-domain[i]))*(mus[i]+mus[i+1]) + inv(2β*(domain[i+2]-domain[i+1]))*(mus[i+1]+mus[i+2])
            B[i,i] = (domain[i+1]-domain[i])*(mus[i]/12+mus[i+1]/4) + (domain[i+2]-domain[i+1])*(mus[i+1]/4+mus[i+2]/12)

            M[i,i+1] = M[i+1,i] = -inv(2β*(domain[i+2]-domain[i+1]))*(mus[i+1]+mus[i+2])
            B[i,i+1] = B[i+1,i] = (domain[i+2]-domain[i+1])*(mus[i+1]+mus[i+2])/12
        end
    
        #boundary terms
        M[N,N] = inv(2β*(domain[N+1]-domain[N]))*(mus[N]+mus[N+1]) + inv(2β*(domain[2]-domain[1]))*(mus[1]+mus[2])
        B[N,N] = (domain[N+1]-domain[N])*(mus[N]/12 +mus[N+1]/4) + (domain[2]-domain[1])*(mus[1]/4 +mus[2]/12)
        M[1,N] = M[N,1] = -inv(2β*(domain[2]-domain[1]))*(mus[1]+mus[2])
        B[1,N] = B[N,1] = (domain[2]-domain[1])*(mus[1]+mus[2])/12



        #Mᵢⱼ = β⁻¹∫∇ϕᵢ⋅∇ϕⱼ dμ + ∫ϕᵢϕⱼα dμ
        #Bᵢⱼ = ∫ϕᵢϕⱼ dμ 

        M = Symmetric(M)
        B = Symmetric(B)

        function δM(α)
            ΔM = zeros(N,N)
            for i=1:N-1
                ΔM[i,i] = α[i]*(domain[i+1]-domain[i])*(mus[i]/12+mus[i+1]/4) + α[i+1]*(domain[i+2]-domain[i+1])*(mus[i+1]/4+mus[i+2]/12)
                ΔM[i,i+1] = ΔM[i+1,i] = α[i+1]*(domain[i+2]-domain[i+1])*(mus[i+1]+mus[i+2])/12
            end
            ΔM[N,N] = α[N]*(domain[N+1]-domain[N])*(mus[N]/12 +mus[N+1]/4) + α[1]*(domain[2]-domain[1])*(mus[1]/4 +mus[2]/12)
            ΔM[1,N] = ΔM[N,1] = α[1]*(domain[2]-domain[1])*(mus[1]+mus[2])/12
            return Symmetric(ΔM)
        end


        """
        Derivative of a soft-killing operator eigenvalue with respect to a component of the soft killing rate. 
        The index ix of this component should refer to the killing rate on the ix-th cell of the full mesh.
        In the case of Dirichlet boundary condition, u should be extended with zeros adequately to match the dimensionality of the full periodic FEM generator.
        """
        function ∂λ(α,u,ix)
            if ix>1
                return u[ix]^2 * (domain[ix+1]-domain[ix])*(mus[ix]/12+mus[ix+1]/4) +
                u[ix-1]^2 * (domain[ix+1]-domain[ix])*(mus[ix]/4+mus[ix+1]/12) +
                2*u[ix]*u[ix-1]*(domain[ix+1]-domain[ix])*(mus[ix]+mus[ix+1])/12
            else
                return u[1]^2 * (domain[2]-domain[1])*(mus[1]/12+mus[2]/4) +
                u[N]^2 * (domain[2]-domain[1])*(mus[1]/4+mus[2]/12) +
                2*u[N]*u[1]*(domain[2]-domain[1])*(mus[1]+mus[2])/12
            end

        end

        return M,B,δM,∂λ
    end


    """
    Returns λ1,λ2 and associated gradients with respect to a soft killing rate α

    M,B, diag_weights, off_diag_weights should be the return values of `build_FEM_matrices_1D`.
    log_α is the vector of soft log killing rates on the cells of the mesh 
    This is easier to optimize, while avoiding negative values of α. α must be of size N, 
    Ω_indices correspond to the indices of vertices inside the domain (where α is finite). 
    """
    function calc_soft_killing_grads(M,B,δM,∂λ,log_α,Ω_indices::AbstractVector{T}) where {T<:Integer}

        N = first(size(M))
        α = exp.(log_α)
        ΔM_dirichlet = δM(α)[Ω_indices,Ω_indices]
        M_dirichlet = M[Ω_indices,Ω_indices] + ΔM_dirichlet
        B_dirichlet = B[Ω_indices,Ω_indices]

        λs,us = eigen(M_dirichlet,B_dirichlet)
        u1_dirichlet = us[:,1]
        u2_dirichlet = us[:,2]
        λ1,λ2 = λs[1:2]

        u1 = zeros(N)
        u2 = zeros(N)

        u1[Ω_indices] .= u1_dirichlet
        u2[Ω_indices] .= u2_dirichlet

        ∇λ1 = α .* [∂λ(α,u1,ix) for ix=1:N]
        ∇λ2 = α .* [∂λ(α,u2,ix) for ix=1:N]

        return u1,u2,λ1,λ2,∇λ1,∇λ2

    end

    function build_FEM_schrodinger_1D(V::Function,β::S,domain::AbstractVector{T}) where {S<:Real,T<:Real}
        N=length(domain)-1
        M=zeros(N,N)
        B=zeros(N,N)
    end


    function QSD_1D_FEM_schrodinger(W::Function, β::Float64,Ω::AbstractVector{Float64};weights=nothing , precond=nothing)

        N=length(Ω)-1
        if weights===nothing
            diag_weights_diff, off_diag_weights_diff, diag_weights_pot, off_diag_weights_pot, diag_weights, off_diag_weights = calc_weights_schrodinger_periodic(W,Ω)
        else
            diag_weights_diff, off_diag_weights_diff, diag_weights_pot, off_diag_weights_pot, diag_weights, off_diag_weights = weights
        end

        M=zeros(N-1,N-1)
        B=zeros(N-1,N-1)

        for i=1:N-1 
            M[i,i] = diag_weights_pot[i] + diag_weights_diff[i] / β
            B[i,i] = diag_weights[2i-1] + diag_weights[2i]
        end

        for i=1:N-2 #off-diagonal terms
            M[i,i+1] = M[i+1,i] = off_diag_weights_pot[i] + off_diag_weights_diff[i] / β
            B[i,i+1] = B[i+1,i] = off_diag_weights[i]

        end
        
        (precond !== nothing) && ( M.=precond*M ; B.=precond*B )

        M = Symmetric(M)
        B = Symmetric(B)
        
        λs,us=eigen(M,B)

        return λs[1:2],us[:,1:2]
    end

    """
    Computes the QSD for the Overdamped Langevin dynamics in a 1D potential with soft killing, using a finite-element method on a regular mesh
    Arguments: V, the potential function
            β, the inverse temperature
            α, a vector giving the killing rate on each of the cells of the discretized domain
    Keyword argument: weights, optional. The result of calc_weights_periodic(V,β,D) where D is the domain. Allows to recycle quadrature computations.
    """
    function SQSD_1D_FEM(mu::Function,β::Float64,α::Vector{Float64}; weights = nothing)

        N=length(α)
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
        
        # Compute M -- in fact we compute an unnormalized version of M, since we only care about ratios of eigenvalues
        
        if weights === nothing
            diag_weights_diff,off_diag_weights_diff,diag_weights,off_diag_weights = calc_weights_periodic(mu,range(0,1,N+1))
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

        B=zeros(N,N)
        δM = zeros(N,N)

        for i=1:N
            j = (i == N) ? 1 : i+1
            B[i,i] = diag_weights[2i-1] + diag_weights[2i]
            B[i,j] = off_diag_weights[i]
            δM[i,i] = α[i]*diag_weights[2i-1] + α[j]*diag_weights[2i]
            δM[i,j] = δM[j,i] = α[j] * off_diag_weights[i]
        end

        # diagonalize discretized operator

        H = Symmetric(M + δM)
        B= Symmetric(B)

        λs,us=eigen(H,B)

        return λs[1:2],us[:,1:2]
    end

    """solve the discretized eigenvalue problem associated with the Schrodinger-like eigenproblem Δu/β-Wu-αu = λu. weights should be the output of 
    calc_weights_schrodinger_periodic(W,D) where D is some spatial domain and W is the potential.
    """
    function SQSD_1D_FEM_schrodinger(W::Function,β::Float64,α::Vector{Float64}; weights = nothing , precond= nothing)

        N=length(α)

        if weights === nothing #assume the spatial domain is [0,1]
            diag_weights_diff, off_diag_weights_diff, diag_weights_pot, off_diag_weights_pot, diag_weights, off_diag_weights=calc_weights_schrodinger_periodic(W,range(0,1,N+1))
        else
            diag_weights_diff, off_diag_weights_diff, diag_weights_pot, off_diag_weights_pot, diag_weights, off_diag_weights=weights
        end

    # FEM discretization of eigenvalue problem, yielding a generalized eigenvalue problem, A(α)u = λMu, ( M ≅ matrix of the restriction of (u,v) -> ∫uv to P1 subspace, A(α)≅ matrix of the restriction of (u,v) -> ∫-∇u⋅∇v +Wuv -αuv to P1 subspace)

    M=zeros(N,N)
    B=zeros(N,N) 

    for i=1:N
        j = (i == N) ? 1 : i+1
        M[i,i] = diag_weights_pot[i] + diag_weights_diff[i]/β + α[i]*diag_weights[2i-1] + α[j]*diag_weights[2i]
        M[i,j] = M[j,i]= off_diag_weights_pot[i] + off_diag_weights_diff[i]/β + α[j]*off_diag_weights[i]

        #mass matrix
        B[i,i] = diag_weights[2i-1] + diag_weights[2i]
        B[i,j] = B[j,i] = off_diag_weights[i]
    end

    #diagonalize discretized operator
    (precond !== nothing) && ( M.= precond*M ; B.= precond*B)
    M = Symmetric(M) #to retrieve correct eigenvalues
    B = Symmetric(B)
    #=    println(M)
    println(diag_weights) =#
    λs,us=eigen(M,B)
    
    return λs[1:2],us[:,1:2]
    end

end