module QSDs
export calc_weights_periodic,
        calc_weights_schrodinger_periodic,
        QSD_1D_FEM,
        QSD_1D_FEM_schrodinger,
        SQSD_1D_FEM,
        SQSD_1D_FEM_schrodinger,
        build_FEM_matrices_1D_PBC,
        build_FEM_matrices_1D_Neumann,
        soft_killing_grads,
        build_FEM_matrices_2D,
        apply_bc,
        reverse_bc,
        qsd_2d

    using Cubature, Arpack, SparseArrays, LinearAlgebra, Triangulate

    Base.length(nothing) = 0 # for convenience

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
    Build FEM matrices in 1D associated with the overdamped Langevin generator eigenproblem with periodic boundary conditions (PBC).
    Domain is an abstract vector [q_1,…, q_{N+1}] where q_{N+1} and q_1 are identified with each other by PBC.
    V should be a periodic function, ie V(q_1)=V(q_{N+1})
    """
    function build_FEM_matrices_1D_PBC(V::Function,β::S,domain::AbstractVector{T}) where {S<:Real,T<:Real}
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
    Returns λ1,λ2 and associated gradients with respect to a soft killing log rate log_α

    M,B, diag_weights, off_diag_weights should be the return values of `build_FEM_matrices_1D`.
    log_α is the vector of soft killing log rates on the cells of the mesh.
    This is easier to optimize, while avoiding negative values of α. α must be of size N, 
    Ω_indices correspond to the indices of vertices inside the domain (where α is finite). 
    """
    function soft_killing_grads(M,B,δM,∂λ,log_α,Ω_indices::AbstractVector{T}) where {T<:Integer}

        N = first(size(M))
        Nα=length(log_α)
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

        ∇λ1 = α .* [∂λ(α,u1,ix) for ix=1:Nα]
        ∇λ2 = α .* [∂λ(α,u2,ix) for ix=1:Nα]

        return u1,u2,λ1,λ2,∇λ1,∇λ2

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
    
    """ 
    Build FEM matrices in 1D associated with the overdamped Langevin generator eigenproblem, with homogeneous Neumann boundary conditions. 
    Domain is an abstract vector [q_1,…, q_{N}]
    """
    function build_FEM_matrices_1D_Neumann(V::Function,β::S,domain::AbstractVector{T}) where {S<:Real,T<:Real}
        N=length(domain)

        M=spzeros(N,N)
        B=spzeros(N,N)

        mus = exp.(-β*V.(domain))
    
        for i=2:N-1
            M[i,i] = inv(2β*(domain[i]-domain[i-1]))*(mus[i-1]+mus[i]) + inv(2β*(domain[i+1]-domain[i]))*(mus[i]+mus[i+1])
            B[i,i] = (domain[i]-domain[i-1])*(mus[i-1]/12+mus[i]/4) + (domain[i+1]-domain[i])*(mus[i]/4+mus[i+1]/12)

            M[i,i+1] = M[i+1,i] = -inv(2β*(domain[i+1]-domain[i]))*(mus[i]+mus[i+1])
            B[i,i+1] = B[i+1,i] = (domain[i+1]-domain[i])*(mus[i]+mus[i+1])/12
        end
    
        #boundary terms
        M[1,1] = inv(2β*(domain[2]-domain[1]))*(mus[1]+mus[2])
        B[1,1]= (domain[2]-domain[1])*(mus[1]/4+mus[2]/12)

        M[1,2] = M[2,1] = -inv(2β*(domain[2]-domain[1]))*(mus[1]+mus[2])
        B[1,2] = B[2,1] =(domain[2]-domain[1])*(mus[1]+mus[2])/12

        M[N,N]=inv(2β*(domain[N]-domain[N-1]))*(mus[N-1]+mus[N])
        B[N,N]=(domain[N]-domain[N-1])*(mus[N-1]/12+mus[N]/4)


        #Mᵢⱼ = β⁻¹∫∇ϕᵢ⋅∇ϕⱼ dμ + ∫ϕᵢϕⱼα dμ
        #Bᵢⱼ = ∫ϕᵢϕⱼ dμ 

        M = Symmetric(M)
        B = Symmetric(B)

        function δM(α)
            ΔM = spzeros(N,N)
            for i=2:N-1
                ΔM[i,i] = α[i-1]*(domain[i]-domain[i-1])*(mus[i-1]/12+mus[i]/4) + α[i]*(domain[i+1]-domain[i])*(mus[i]/4+mus[i+1]/12)
                ΔM[i,i+1] = ΔM[i+1,i] = α[i]*(domain[i+1]-domain[i])*(mus[i]+mus[i+1])/12
            end

            #boundary terms
            ΔM[1,1] = α[1]*(domain[2]-domain[1])*(mus[1]/4+mus[2]/12)
            ΔM[1,2] = ΔM[2,1] = α[1]*(domain[2]-domain[1])*(mus[1]+mus[2])/12
            ΔM[N,N] = α[N-1]*(domain[N]-domain[N-1])*(mus[N-1]/12 +mus[N]/4)

            return Symmetric(ΔM)
        end


        """
        Derivative of a soft-killing operator eigenvalue with respect to a component of the soft killing rate. 
        The index ix of this component should refer to the killing rate on the ix-th cell of the full mesh.
        In the case of Dirichlet boundary condition, u should be extended with zeros adequately to match the dimensionality of the full periodic FEM generator.
        """
        function ∂λ(u,ix)
            return (domain[ix+1]-domain[ix]) *( u[ix]^2*(mus[ix]/4 +mus[ix+1]/12) +
                                                u[ix+1]^2*(mus[ix]/12 + mus[ix+1]/4)+
                                                u[ix]*u[ix+1]*(mus[ix]+mus[ix+1])/6)
        end

        return M,B,δM,∂λ
    end


    @inline tri_area(xi,yi,xj,yj,xk,yk) = (xi*yj - xj*yi + xj*yk - xk*yj + xk*yi - yk*xi)/2 # oriented area of a triangle ( + is clockwise orientation)
    @inline off_diag_M_int(xi,yi,xj,yj,xk,yk,μi,μj,μk,Aijk) = -((xi-xk)*(xj-xk)+(yi-yk)*(yj-yk))*(μi+μj+μk)/12Aijk #∫∇ϕᵢ⋅∇ϕⱼdμ on triangle Tᵢⱼₖ
    @inline diag_M_int(xi,yi,xj,yj,xk,yk,μi,μj,μk,Aijk) = ((xj-xk)^2+(yj-yk)^2)*(μi+μj+μk)/12Aijk # ∫|∇ϕᵢ|²dμ on triangle Tᵢⱼₖ
    @inline off_diag_B_int(μi,μj,μk,Aijk) = Aijk * ((μi+μj)/30 + μk/60) #∫ϕᵢϕⱼdμ on triangle Tᵢⱼₖ
    @inline diag_B_int(μi,μj,μk,Aijk) = Aijk * (μi/10 + (μj+μk)/30) #∫ϕᵢ²dμ on triangle Tᵢⱼₖ

    """ 
    Build FEM matrices in 2D associated with the overdamped Langevin generator eigenproblem with possible soft killing perturbation.
    V should be a function V(x,y)::Float64,
    β is the inverse thermodynamic temperature
    domain should be a TriangulateIO object from Triangulate.jl (for instance obtained from calling QSDs.triangulate_domain)
    periodic_images is a vector of 2 x N indices where N is the number of points in the triangulation, where periodic_images[:,k] = [i,j] means that point i is identified with point j.
    In the case more than two points are identified, the user should make sure that the chosen image is unique.
    The issue of orientation-reversing identifications is ignored.
    """
    function build_FEM_matrices_2D(V::Function,β::S,domain::TriangulateIO) where {S<:Real}
        N = numberofpoints(domain)
        M=spzeros(N,N)
        B=spzeros(N,N)

        mus = zeros(N)

        for i=1:N
            xi,yi = domain.pointlist[:,i]
            mus[i] = exp(-β * V(xi,yi)) # vertex values of unnormalized Gibbs measure    
        end

        T = size(domain.trianglelist)[2]
        tmp_M = tmp_B = 0.0
        tri_areas = zeros(T)

        for n=1:T
            i,j,k = domain.trianglelist[:,n]
            xi,yi = domain.pointlist[:,i]
            xj,yj = domain.pointlist[:,j]
            xk,yk = domain.pointlist[:,k]

            μi,μj,μk= mus[i],mus[j],mus[k]

            Aijk = tri_area(xi,yi,xj,yj,xk,yk)
            tri_areas[n]=Aijk

            M[i,i] += diag_M_int(xi,yi,xj,yj,xk,yk,μi,μj,μk,Aijk)/β
            B[i,i] += diag_B_int(μi,μj,μk,Aijk)

            M[j,j] += diag_M_int(xj,yj,xk,yk,xi,yi,μj,μk,μi,Aijk)/β
            B[j,j] += diag_B_int(μj,μk,μi,Aijk)

        
            M[k,k] += diag_M_int(xk,yk,xi,yi,xj,yj,μk,μi,μj,Aijk)/β
            B[k,k] += diag_B_int(μk,μi,μj,Aijk)

            tmp_M = off_diag_M_int(xi,yi,xj,yj,xk,yk,μi,μj,μk,Aijk)/β
            tmp_B = off_diag_B_int(μi,μj,μk,Aijk)

            if i<j
                M[i,j] += tmp_M
                B[i,j] += tmp_B
            else
                M[j,i] += tmp_M
                B[j,i] += tmp_B
            end
            
            tmp_M = off_diag_M_int(xj,yj,xk,yk,xi,yi,μj,μk,μi,Aijk)/β
            tmp_B = off_diag_B_int(μj,μk,μi,Aijk)

            if j<k
                M[j,k] += tmp_M
                B[j,k] += tmp_B
            else
                M[k,j] += tmp_M
                B[k,j] += tmp_B
            end

            tmp_M = off_diag_M_int(xk,yk,xi,yi,xj,yj,μk,μi,μj,Aijk)/β
            tmp_B = off_diag_B_int(μk,μi,μj,Aijk)

            if i<k
                M[i,k] += tmp_M
                B[i,k] += tmp_B
            else
                M[k,i] += tmp_M
                B[k,i] += tmp_B
            end
            

        end

        M=Symmetric(M)
        B=Symmetric(B)

        function δM(α)
            ΔM = spzeros(N,N)
            tmp = 0.0

            for n=1:T
                i,j,k = domain.trianglelist[:,n]
                μi,μj,μk= mus[i],mus[j],mus[k]

                Aijk = tri_areas[n]

                ΔM[i,i] += α[n] * diag_B_int(μi,μj,μk,Aijk)

                ΔM[j,j] += α[n] * diag_B_int(μj,μk,μi,Aijk)

                ΔM[k,k] += α[n] * diag_B_int(μk,μi,μj,Aijk)

                tmp = off_diag_B_int(μi,μj,μk,Aijk)

                if i<j
                    ΔM[i,j] += α[n] * tmp
                else
                    ΔM[j,i] += α[n] * tmp
                end

                tmp = off_diag_B_int(μj,μk,μi,Aijk)

                if j<k
                    ΔM[j,k] += α[n] * tmp
                else
                    ΔM[k,j] += α[n] * tmp
                end

                tmp = off_diag_B_int(μk,μi,μj,Aijk)

                if i<k
                    ΔM[i,k] += α[n] * tmp
                else
                    ΔM[k,i] += α[n] * tmp
                end

            end
                return Symmetric(ΔM)
        end

        function ∂λ(u,n)
            i,j,k = domain.trianglelist[:,n]

            μi,μj,μk= mus[i],mus[j],mus[k]

            Aijk = tri_areas[n]

            grad = u[i]^2 * diag_B_int(μi,μj,μk,Aijk) +
                u[j]^2 * diag_B_int(μj,μk,μi,Aijk) +
                u[k]^2 * diag_B_int(μk,μi,μj,Aijk) +
                2u[i] * u[j] * off_diag_B_int(μi,μj,μk,Aijk) +
                2u[j]*u[k]* off_diag_B_int(μj,μk,μi,Aijk) +
                2u[k]*u[i]* off_diag_B_int(μk,μi,μj,Aijk)
            
            return grad
        end

        return M,B,δM,∂λ
    end

    """
    Apply Periodic and Dirichlet boundary conditions to FEM matrices in 2D.
    """
    function apply_bc(M,B,periodic_images,dirichlet_boundary_points)#,dirichlet_boundary_points)
        Mper = sparse(M)
        Bper = sparse(B)

        for ix=1:size(periodic_images)[2]
            i,j = periodic_images[:,ix]
            Mper[j,:] += Mper[i,:]
            Mper[:,j] += Mper[:,i]
           # Mper[j,j] -= Mper[i,i] # correct double counting

            Bper[j,:] += Bper[i,:]
            Bper[:,j] += Bper[:,i]
          #  Bper[j,j] -= Bper[i,i] # correct double counting

        end
        
        N = first(size(M))

        reduced_points = Set(vcat(periodic_images[1,:],dirichlet_boundary_points))
        unreduced_points = setdiff(1:N,reduced_points)

        Mper=Mper[unreduced_points,unreduced_points]
        Bper=Bper[unreduced_points,unreduced_points]

        return Symmetric(Mper),Symmetric(Bper)
    end

    """
    Reconstruct eigenvectors that have been reduced by boundary conditions
    to a format compatible with the triangulation of the domain.
    """
    function reverse_bc(u_reduced,N,periodic_images,dirichlet_boundary_points)#,dirichlet_boundary_points)
        reduced_points = Set(vcat(periodic_images[1,:],dirichlet_boundary_points))
        unreduced_points=setdiff(1:N,reduced_points)
        u = zeros(N)
        u[unreduced_points] .= u_reduced

        for ix=1:size(periodic_images)[2]
            i,j = periodic_images[:,ix]
            u[i] = u[j]
        end

        u[dirichlet_boundary_points] .= 0
        return u
    end

    """
    Returns the qsd associated with a discretized right eigenvector of the generator, with optional soft killing.
    """
    function qsd_2d(u,V,β,domain::TriangulateIO)
        mu(x,y) = exp(-β*V(x,y))
        X = domain.pointlist[1,:]
        Y = domain.pointlist[2,:]
        mus = mu.(X,Y)
        Z = 0.0
        
        for n=1:numberoftriangles(domain)
            (i,j,k) = domain.trianglelist[:,n]
            
            xi,yi = domain.pointlist[:,i]
            xj,yj = domain.pointlist[:,j]
            xk,yk = domain.pointlist[:,k]

            Z += tri_area(xi,yi,xj,yj,xk,yk) * (u[i]+u[j]+u[k])*(mus[i]+mus[j]+mus[k])/6
        end
        return (u .* mus)/Z
    end

    """
    Computes the exterior normal derivative of u, which is a piecewise affine function on the domain, 
    defined by its values at each vertex.
    boundary_triangles is a (3 × M) Int32 - valued array,
     where boundary_triangles[:,n] = [i,j,k] means that the vertices i and j are succesive vertices on the boundary, oriented clockwise,
     and vertex k is an interior vertex such that triangle [i,j,k] is in domain.
     It is assumed u[i] = u[j] = 0.
    """
    function sq_normal_derivatives(u,domain::TriangulateIO,boundary_triangles)
        _, n_boundary_triangles = size(boundary_triangles)

        sq_derivatives = zeros(n_boundary_triangles)
        for n=1:n_boundary_triangles
            i,j,k = boundary_triangles[:,n]
            xi,yi=domain.pointlist[:,i]
            xj,yj=domain.pointlist[:,j]
            xk,yk=domain.pointlist[:,k]
            Aijk=tri_area(xi,yi,xj,yj,xk,yk)
            r_ij2 = (xi-xj)^2 + (yi-yj)^2
            sq_derivatives[n] = 4 * r_ij2 * u[k]^2 / Aijk^2

        end

        return sq_derivatives
    end

end # module QSDs