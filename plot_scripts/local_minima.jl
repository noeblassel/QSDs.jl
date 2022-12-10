#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature

N=parse(Int64,ARGS[1])
β=parse(Float64,ARGS[2])

include("../QSDs.jl")

V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)


W(q) = (β*dV(q)^2/2-d2V(q))/2

mu_tilde(q) = exp(-β * V(q))
Z,_ = hquadrature(mu_tilde,0,1)
mu(q) = mu_tilde(q) / Z

saddle_points = [0.350482,0.649518]
Ωl,Ωr=saddle_points

h0=-0.05
hmax=first(saddle_points)

λ1s_classic=Float64[]
gaps_classic=Float64[]

λ1s_schrodinger=Float64[]
gaps_schrodinger=Float64[]

hrange=h0:0.001:hmax
V_rel(h)=V(first(saddle_points)+h)
V_rels=V_rel.(hrange)

maxima_classic = Float64[]
maxima_schrodinger = Float64[]
heights_classic = Float64[]
heights_schrodinger = Float64[]
hs_classic=Float64[]
hs_schrodinger=Float64[]

for h=hrange
    println(h)
    domain = range(Ωl-h,Ωr+h,N)
    _,us_classic=QSDs.QSD_1D_FEM(mu,β,domain)
    _,us_schrodinger=QSDs.QSD_1D_FEM_schrodinger(W,β,domain)

    w_classic = us_classic[:,1]
    push!(w_classic,0.0)
    pushfirst!(w_classic,0.0)

    w_schrodinger = us_schrodinger[:,1]
    push!(w_schrodinger,0.0)
    pushfirst!(w_schrodinger,0.0)

    w_classic .*=mu_tilde.(domain)
    w_schrodinger .*=exp.(-β*V.(domain)/2)

    w_classic /= sum(w_classic) * (Ωr-Ωl +2h)/(N-1)
    w_schrodinger /= sum(w_schrodinger) *(Ωr-Ωl +2h)/(N-1)

    extrema_classic = [domain[i] for i=2:N-1 if (w_classic[i-1]<w_classic[i])&&(w_classic[i+1]<w_classic[i])]
    extrema_schrodinger = [domain[i] for i=2:N-1 if (w_schrodinger[i-1]<w_schrodinger[i])&&(w_schrodinger[i+1]<w_schrodinger[i])]

    append!(maxima_classic, extrema_classic)
    append!(maxima_schrodinger, extrema_schrodinger)
    append!(hs_classic,[h for i=1:length(extrema_classic)])
    append!(hs_schrodinger,[h for i=1:length(extrema_schrodinger)])
    append!(heights_classic,[w_classic[i] for i=2:N-1 if (w_classic[i-1]<w_classic[i])&&(w_classic[i+1]<w_classic[i])])
    append!(heights_schrodinger,[w_schrodinger[i] for i=2:N-1 if (w_schrodinger[i-1]<w_schrodinger[i])&&(w_schrodinger[i+1]<w_schrodinger[i])])
end

f=open("result_files/extrema.out_$β","w")
println(f,"β = $β")
println(f,"""
Ωl=$Ωl
V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)
""")
println(f,"hs=[",join(hrange,","),"]")
println(f,"hs_classic=[",join(hs_classic,","),"]")
println(f,"hs_schrodinger=[",join(hs_classic,","),"]")
println(f,"maxima_classic=[",join(maxima_classic,","),"]")
println(f,"maxima_schrodinger=[",join(maxima_schrodinger,","),"]")
println(f,"heights_classic=[",join(heights_classic,","),"]")
println(f,"heights_schrodinger=[",join(heights_schrodinger,","),"]")
close(f)

