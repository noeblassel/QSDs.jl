#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature#, Plots

N=parse(Int64,ARGS[1])
dβ=parse(Float64,ARGS[2])
dh=parse(Float64,ARGS[3])

include("../QSDs.jl")

V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)

saddle_points = [0.350482,0.649518]
Ωl,Ωr=saddle_points

h0=-0.05
hmax=first(saddle_points)
hrange=h0:dh:hmax
V_rel(h)=V(last(saddle_points)+h)
V_rels=V_rel.(hrange)

βrange=0:dβ:5

hstars_classic=Float64[]
hstars_schrodinger=Float64[]

for β=βrange
    println(β)
    W(q) = (β*dV(q)^2/2-d2V(q))/2

    mu_tilde(q) = exp(-β * V(q))
    Z,_ = hquadrature(mu_tilde,0,1)
    mu(q) = mu_tilde(q) / Z
    
    λ1s_classic=Float64[]
    gaps_classic=Float64[]

    λ1s_schrodinger=Float64[]
    gaps_schrodinger=Float64[]

    for h=hrange
        println("\t",h)
        λs_classic,_=QSD_1D_FEM(mu,β,Ωl-h,Ωr+h,N)
        λs_schrodinger,_=QSD_1D_FEM_schrodinger(W,β,Ωl-h,Ωr+h,N)

        λ1,λ2=λs_classic
        push!(λ1s_classic,λ1)
        push!(gaps_classic,λ2-λ1)

        λ1,λ2=λs_schrodinger
        push!(λ1s_schrodinger,λ1)
        push!(gaps_schrodinger,λ2-λ1)
    end

    ratios_classic = gaps_classic ./ λ1s_classic
    ratios_schrodinger = gaps_schrodinger ./ λ1s_schrodinger

    hstar_classic=hrange[argmax(ratios_classic)]
    hstar_schrodinger=hrange[argmax(ratios_schrodinger)]

    push!(hstars_classic,hstar_classic)
    push!(hstars_schrodinger,hstar_schrodinger)
end

f=open("max_ratios.out","w")
println(f,"βs=[",join(βrange,","),"]")
println(f,"hstars_classic=[",join(hstars_classic,","),"]")
println(f,"hstars_schrodinger=[",join(hstars_schrodinger,","),"]")
close(f)
#= 
plot(xlabel="h*",ylabel="β")
plot!(hstars_classic,βrange,linestyle=:dash,color=:red,label="classic")
plot!(hstars_schrodinger,βrange,linestyle=:dot,color=:blue,label="schrodinger")

plot!(twinx(),hrange,V_rels,label="",color=:black,linestyle=:dot)

savefig("./figures/h_stars.pdf") =#