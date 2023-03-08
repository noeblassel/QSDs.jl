#!/bin/env julia

using Plots

dir=ARGS[1]
ix=parse(Int64,ARGS[2])
βmin,dβ,βmax=parse.(Float64,ARGS[3:5])
output_file_prefix=ARGS[6]
legend=parse(Bool,ARGS[7])

println("usage dir ix βmin dβ βmax output_file_prefix legend::Bool")

include(joinpath("result_files/eigen",dir,"potential.out"))
pl_u1=plot(xlabel="h",ylabel="u1")
pl_u2=plot(xlabel="h",ylabel="u2")

for β=βmin:dβ:βmax
    println(β)
    include(joinpath("result_files/eigen",dir,"us","beta$(β)_N1000_ix$(ix).out"))
    u1=us[:,1]
    u1 *= 1000/sqrt(sum(abs2,u1))
    (u1[10]<0) && (u1=-u1)
    u2 =us[:,2]
    u2 *=1000/sqrt(sum(abs,u2))
    (u2[10]<0) && (u2=-u2)

    if legend
        plot!(pl_u1,domain[1:length(u1)],u1,label="β=$(β)")
        plot!(pl_u2,domain[1:length(u2)],u2,label="β=$(β)")
    else
        plot!(pl_u1,domain[1:length(u1)],u1,label="",linewidth=1,color=:black)
        plot!(pl_u2,domain[1:length(u2)],u2,label="",linewidth=1,color=:black)
    end
end

plot!(twinx(pl_u1),domain,Vs,linestyle=:dot,color=:red,label="")
plot!(twinx(pl_u2),domain,Vs,linestyle=:dot,color=:red,label="")

savefig(pl_u1,joinpath("figures/","$(output_file_prefix)_u1_$(dir).pdf"))
savefig(pl_u2,joinpath("figures/","$(output_file_prefix)_u2_$(dir).pdf"))