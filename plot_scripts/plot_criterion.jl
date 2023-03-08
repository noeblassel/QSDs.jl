#!/bin/env julia

using Plots

dir=ARGS[1]
βmin,dβ,βmax=parse.(Float64,ARGS[2:4])
output_file_prefix=ARGS[5]
ix_start = parse(Int64,ARGS[6])
ix_end = parse(Int64,ARGS[7])
println("usage dir  βmin dβ βmax output_file_prefix ix_start ix_end")

include(joinpath("result_files/eigen",dir,"potential.out"))

for β=βmin:dβ:βmax
    println(β)
    include(joinpath("result_files/eigen/normal_derivatives","dirichlet_normal_derivatives_β$(β)_$(dir).out"))
    pl=plot(xlabel="h",ylabel="")
    obj = (λ2s - λ1s) ./ λ1s
    crit = hcat(λ2s .* r_normal_derivative_u1.^2,λ1s .* r_normal_derivative_u2.^2)
    n = length(obj)
    plot!(pl,domain[ix_start:ix_end],obj,color=:blue,label="(λ₂-λ₁)/λ₁")
    #hstar = [domain[ix_start + argmax(obj) -1]]
    #vline!(pl,hstar,color=:black,label="",linestyle=:dot)
    plot!(twinx(pl),domain[ix_start:ix_end],crit,label=hcat("λ₂(∂u₁/∂n)²", "λ₁(∂u₂/∂n)²"),color=hcat(:green, :red),yaxis=:log,legend=:bottomleft)
    savefig(pl,"figures/dirichlet_eigen/criterion/$(output_file_prefix)_β$(β)_$(dir).pdf")
end
