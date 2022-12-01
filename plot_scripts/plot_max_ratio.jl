#!/libre/blasseln/julia-1.8.2/bin/julia

using LinearAlgebra, Cubature#, Plots

N=parse(Int64,ARGS[1])
dβ=parse(Float64,ARGS[2])

include("../QSDs.jl")


V(q)=cos(6π*q) + cos(4π*q)/2
dV(q)= -6π*sin(6π*q) -2π*sin(4π*q)
d2V(q) = -36*π^2*cos(6π*q) - 8π^2*cos(4π*q)

saddle_points = [0.350482,0.649518]
Ωl,Ωr=saddle_points

domain=range(0,1,N+1)

βrange=dβ:dβ:5

hstars_classic=Float64[]
hstars_schrodinger=Float64[]

for β=βrange
    println(β)

    mu_tilde(q) = exp(-β * V(q))
    Z,_ = hquadrature(mu_tilde,0,1)
    mu(q) = mu_tilde(q) / Z
    
    W(q) = (β*dV(q)^2/2-d2V(q))/2
    
    weights_classic = calc_weights_periodic(mu,domain)
    diag_weights_diff_classic,off_diag_weights_diff_classic,diag_weights_classic,off_diag_weights_classic = weights_classic

    weights_schrodinger = calc_weights_schrodinger_periodic(W,domain)
    diag_weights_diff_schrodinger,off_diag_weights_diff_schrodinger,diag_weights_schrodinger,off_diag_weights_schrodinger = weights_schrodinger

    trunc_weights_classic(i)=(weights_classic[1][i:end-i+1],weights_classic[2][i:end-i+1],weights_classic[3][2i-1:end-2i+2],weights_classic[4][i:end-i+1])
    trunc_weights_schrodinger(i)=(weights_schrodinger[1][i:end-i+1],weights_schrodinger[2][i:end-i+1],weights_schrodinger[3][i:end-i+1],weights_schrodinger[4][i:end-i+1],weights_schrodinger[5][2i-1:end-2i+2],weights_schrodinger[6][i:end-i+1])
    
    λ1s_classic=Float64[]
    gaps_classic=Float64[]

    λ1s_schrodinger=Float64[]
    gaps_schrodinger=Float64[]

    for i=1:floor(Int64,N/2-1)
        println("\t",i)
        λs_classic,_=QSD_1D_FEM(mu,β,domain[i:end-i+1];weights=trunc_weights_classic(i))
        λs_schrodinger,_=QSD_1D_FEM_schrodinger(W,β,domain[i:end-i+1];weights=trunc_weights_schrodinger(i))

        λ1,λ2=λs_classic
        push!(λ1s_classic,λ1)
        push!(gaps_classic,λ2-λ1)

        λ1,λ2=λs_schrodinger
        push!(λ1s_schrodinger,λ1)
        push!(gaps_schrodinger,λ2-λ1)
    end

    ratios_classic = gaps_classic ./ λ1s_classic
    ratios_schrodinger = gaps_schrodinger ./ λ1s_schrodinger

    hstar_classic=domain[argmax(ratios_classic)]
    hstar_schrodinger=domain[argmax(ratios_schrodinger)]

    push!(hstars_classic,hstar_classic)
    push!(hstars_schrodinger,hstar_schrodinger)
    flush(stdout)
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