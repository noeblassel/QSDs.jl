#!/bin/env julia

using Plots


istart=100
iend = 800

βs=0.01:0.01:5.0
hstars = zero(βs)

for dir=["1_1","1_2","2_1","2_2"]
    plt = plot()
    include(joinpath("result_files/eigen/",dir,"potential.out"))
    include(joinpath("result_files","bifurcations_$(dir).out"))
    for i=1:length(βs)
        include(joinpath("result_files/eigen",dir,"eigen_β$(βs[i])_N1000.out"))
        hstars[i] = domain[istart + argmax(ratios[1:iend])]
    end

    plot(hstars,βs,xlabel="h",ylabel="β",label="h⋆",color=:red)
    plot!(hs,β_biffs,label="biffurcation",color=:green)
    plot!(twinx(),domain,Vs,label="",linestyle=:dot,color=:black,ylabel="V")

    savefig(joinpath("figures/dirichlet_eigen/","max_ratios_$(dir)_.pdf"))
end