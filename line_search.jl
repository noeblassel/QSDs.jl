using Cubature,LinearAlgebra,Plots

include("QSDs.jl")
include("optimisation.jl")

#= N,β,Niter,cutoff = ARGS

N,Niter=parse.(Int64,[N,Niter])
β,cutoff=parse.(Float64,[β,cutoff]) =#

N=1000
cutoff=1000000.0

V(q) = 2cos(2π*q) - cos(4π*q)
dV(q) = -4π*sin(2π*q)+4π*sin(4π*q)
d2V(q)= -8π^2*cos(2π*q)+16π^2*cos(4π*q)


D=collect(Float64,range(0,1,N+1))
vs=V.(D)
vmin=minimum(vs)
vmax=maximum(vs)



plot_f=plot(V,0,1,label="V",xlabel="q",ylims=(vmin,vmax))
plot_λ=plot(V,0,1,label="V",xlabel="q",ylims=(vmin,vmax))
plot_λ1=plot(xlims=(0,1),xlabel="q",yaxis=:log)
plot_λ2=plot(xlims=(0,1),xlabel="q",yaxis=:log)
vline!(plot_f,[1/6,5/6],label="",linestyle=:dot)
vline!(plot_λ,[1/6,5/6],label="",linestyle=:dot)

γ=0.0001

f(λ1,λ2)=λ2/λ1
g(λ1,λ2)=1/λ1
hrange=0.01:0.005:0.4

for β=1.0:2.0:10.0

    W_pot(q) = -(0.5d2V(q)-β*dV(q)^2/4)


    fs=Float64[]
    λ1s=Float64[]
    λ2s=Float64[]

    weights=calc_weights_schrodinger_periodic(W_pot,D)

    for h=hrange
        println(β," :",h)
        λs,us=QSD_1D_FEM_schrodinger(W_pot,β,h,1-h,N)
        λ1,λ2=λs
        push!(fs,f(λ1,λ2))
        push!(λ1s,λ1)
        push!(λ2s,λ2)
    end

    plot!(plot_f,hrange,vmin.+(vmax-vmin)*fs/maximum(fs),label="β=$β")
    plot!(plot_λ,hrange,vmin .+ (vmax-vmin)*λ1s/maximum(λ1s),label="λ1(β=$β)")
    plot!(plot_λ,hrange,vmin .+ (vmax-vmin)*λ2s/maximum(λ2s),label="λ2(β=$β)")
    plot!(plot_λ1,hrange,λ1s,label="β=$β")
    plot!(plot_λ2,hrange,λ2s,label="β=$β")
    println(minimum(λ2s))
end

vline!(twinx(plot_λ1),[1/6,5/6],label="",linestyle=:dot,color=:black)
vline!(twinx(plot_λ2),[1/6,5/6],label="",linestyle=:dot,color=:black)
plot!(twinx(plot_λ1),V,0,1,label="",linestyle=:dot,color=:black)
plot!(twinx(plot_λ2),V,0,1,label="",linestyle=:dot,color=:black)


savefig(plot_f,"norm_ratios.pdf")
savefig(plot_λ,"norm_lambdas.pdf")
savefig(plot_λ1,"lambda1.pdf")
savefig(plot_λ2,"lambda2.pdf")