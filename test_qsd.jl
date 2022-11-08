using Plots, LinearAlgebra, Cubature, ProgressMeter

#β=5.0

include("QSDs.jl")

Ωl,Ωr=(0.25,0.75)
#N=10

εrange=0.0:0.01:2.0
αrange=0.0:1.0:100.0
βrange=0.0:0.05:10.
ratios=Float64[]

anim = @animate for N=1000:2000
    Ω=range(Ωl,Ωr,N)
    qs=range(0,1,N)
    β=0.2
    println(N)
   # V(q)=sin(4π*q-π/2)+ε*cos(24π*q)
   V(q)=0
    #∇V(q)=4π*cos(4π*q-π/2)-24π*ε*sin(24π*q)

    (Z,errZ)=hquadrature(q -> exp(-β*V(q)),0,1)
    mu(q)=exp(-β*V(q))/Z

    λ1,λ2,qsd=QSD_1D_FEM(V,β,Ωl,Ωr,N)
    #@assert (minimum(u1) >= 0) # check for positivity
    push!(ratios,(λ2-λ1)/λ1)
    analytic_qsd(q) = sin(π*(q-Ωl)/(Ωr-Ωl))*π/(2*(Ωr-Ωl))
    plot(mu,0,1,label="μ")
    plot!(Ω,qsd,label="numerical (N= $N)")
    plot!(analytic_qsd,Ωl,Ωr,label="analytic")
    #plot!(V,0,1,label="V")
    vline!([Ωl,Ωr],label="",linestyle=:dot,color=:red)

end

mp4(anim, "numerical.mp4")
#= plot(βrange,ratios,xlabel="α",ylabel="(τ corr)/(τ exit)",label="")
savefig("perturbation_hot.pdf") =#