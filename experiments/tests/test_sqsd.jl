using Plots,Cubature,LinearAlgebra

include("QSDs.jl")

N=300
#V(q) = 2cos(2π*q) - cos(4π*q)
V(q)=0
βrange=0.1:0.1:5.0
β=0.8

Ωl = 1/6
Ωr = 5/6
mu_no(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_no,0,1)
mu(q) = mu_no(q)/Z


anim=@animate for r= 10.0 .^ (-1:8)
    rate(q) = (Ωl < q < Ωr) ? 0.0 : r
    D = collect(Float64,range(0,1,N+1))
    α = rate.(range(0,1,N+1)[1:N])
    W=calc_weights_periodic(mu,D)
    #λs,qsd,us = QSD_1D_FEM(V,β,Ωl,Ωr,N)
    λs,qsd,us=SQSD_1D_FEM(V,β,α;weights=W)
    Ω=range(Ωl,Ωr,N)
    plot(D[2:end],qsd,label="soft qsd (α = $r)")
    plot!(D,mu.(D),label="mu")
    vline!([1/6,5/6])
    f(q)=sin(π*(q-Ωl)/(Ωr-Ωl))
    z,_=hquadrature(f,Ωl,Ωr)
    plot!(q->f(q)/z,Ωl,Ωr,label="qsd")
end

mp4(anim,"test.mp4",fps=1)
#= ratios = Float64[]
lg_r_range=range(-2,5,500)
anim = @animate for lg_r=lg_r_range
    r= 10^lg_r
    println(r)
    plot(D,mu.(D),label="μ (β = $β)",xaxis="q",yaxis="likelihood",color=:red,linestyle=:dot)
    plot!(Ω,qsd,color=:blue,label="qsd")

    rate(q) = (Ωl < q < Ωr) ? 0.0 : r
    α = rate.(range(0,1,N+1)[1:N])
    λs,sqsd,us = SQSD_1D_FEM(V,β,α)
    plot!(D[2:end],sqsd,color=:green,label="sqsd (α=$(round(r,digits=3)))")

    vline!([Ωr,Ωl],color=:black,linestyle=:dot,label="")
    λ1,λ2 = λs
    push!(ratios,(λ2-λ1)/λ1)
    println("$(last(ratios))")
end

mp4(anim,"lerp_r.mp4") =#