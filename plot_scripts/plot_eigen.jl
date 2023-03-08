#!/bin/env julia

using Plots

dir=ARGS[1]
βmin,dβ,βmax=parse.(Float64,ARGS[2:4])
output_file_prefix=ARGS[5]
ix_start = parse(Int64,ARGS[6])
n_discard = parse(Int64,ARGS[7])

println("usage dir βmin dβ βmax output_file_prefix ix_start n_discard")

include(joinpath("result_files/eigen",dir,"potential.out"))
for β=βmin:dβ:βmax
    println(β)
    λ1_plot =plot(xlabel="h",ylabel="λ1")
    inv_λ1_plot=plot(xlabel="h",ylabel="1/λ1")
    gaps_plot=plot(xlabel="h",ylabel="λ2-λ1")
    inv_gaps_plot=plot(xlabel="h",ylabel="1/(λ2-λ1)")
    ratios_plot=plot(xlabel="h",ylabel="(λ2-λ1)/λ1")

    include(joinpath("result_files/eigen",dir,"eigen_β$(β)_N1000.out"))
    l = length(λ1s) - n_discard
    #plot!(λ1_plot,domain[ix_start:ix_start+l-1],λ1s[1:l],label="",color=:blue)
    plot!(inv_λ1_plot,domain[ix_start:ix_start+l-1],inv.(λ1s[1:l]),label="",color=:blue)
    plot!(gaps_plot,domain[ix_start:ix_start+l-1],gaps[1:l],label="",color=:blue)
    #plot!(inv_gaps_plot,domain[ix_start:ix_start+l-1],inv.(gaps[1:l]),label="",color=:blue)
    plot!(ratios_plot,domain[ix_start:ix_start+l-1],gaps[1:l] ./ λ1s[1:l],label="",color=:blue)

    for pl in [λ1_plot,inv_λ1_plot,gaps_plot,inv_gaps_plot,ratios_plot]
        plot!(twinx(pl),domain,Vs,label="",color=:red,linestyle=:dot)
    end
    #savefig(λ1_plot, joinpath("figures/dirichlet_eigen/","$(output_file_prefix)_l1_$(β)_$(dir).pdf"))
    savefig(inv_λ1_plot, joinpath("figures/dirichlet_eigen/","$(output_file_prefix)_invl1_$(β)_$(dir).pdf"))
    savefig(gaps_plot, joinpath("figures/dirichlet_eigen/","$(output_file_prefix)_gaps_$(β)_$(dir).pdf"))
    #savefig(inv_gaps_plot, joinpath("figures/dirichlet_eigen/","$(output_file_prefix)_invgaps_$(β)_$(dir).pdf"))
    savefig(ratios_plot, joinpath("figures/dirichlet_eigen/","$(output_file_prefix)_ratios_$(β)_$(dir).pdf"))
end
    