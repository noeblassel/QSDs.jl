#!/bin/env julia

using Plots

β=parse(Float64,ARGS[1])
N=parse(Int64,ARGS[2])
l_well_depth,well_depth,r_well_depth=parse.(Float64,ARGS[3:5])

include("../SplinePotentials.jl")

data_dir = ARGS[6]
output_dir = ARGS[7]

n_contour_levels = parse(Int64,ARGS[8])

println("Usage β N l_well_depth well_depth r_well_depth data_dir output_dir n_contour_levels")

include(joinpath(data_dir,"full_landscape_β$(β)_N$(N)_$(l_well_depth)_$(well_depth)_$(r_well_depth).out"))
l_coreset,r_coreset = 0.1,0.9 # q of coresets are 0.1 and 0.9

critical_pts=[0.0,l_coreset,(0.5+l_coreset)/2,0.5,(0.5+r_coreset)/2,r_coreset,1-1/N]
potential_heights = [0.0,-l_well_depth,0.0,-well_depth,0.0,-r_well_depth,0.0]
V,dV,d2V = SplinePotentials.spline_potential_derivatives(critical_pts,potential_heights,1.0)

domain = range(l_coreset,r_coreset,N) 
obj = (λ2s -λ1s) ./ λ1s
criterion = @. sqrt((λ2s*l_normal_derivative_u1^2 - λ1s*l_normal_derivative_u2^2)^2 + (λ2s*r_normal_derivative_u1^2 - λ1s*r_normal_derivative_u2^2)^2)

obj[isnan.(obj)] .= 0.0
hstar_ix = argmax(obj)
hstar = [domain[hstar_ix[1]],domain[hstar_ix[2]]]

criterion[iszero.(criterion)] .= Inf
argmin_crit_ix = argmin(criterion)
argmin_crit=[domain[argmin_crit_ix[1]],domain[argmin_crit_ix[2]]]

println("Maximizer of λ₂/λ₁: $hstar; Maximal value: $(obj[hstar_ix]) ")
println("Minimzer of |λ₂(∂u₁/∂n)²-λ₁(∂u₂/∂n)²|: $argmin_crit, Minimal value: $(criterion[argmin_crit_ix])")

pl_obj = plot(xlabel="b",ylabel="a",title="ln[(λ₂-λ₁)/λ₁]")
contour!(pl_obj,domain,domain,log.(obj),c=:hsv,levels=n_contour_levels)
savefig(pl_obj,joinpath(output_dir,"objective_β$(β)_N$(N)_$(l_well_depth)_$(well_depth)_$(r_well_depth).pdf"))

pl_crit  = plot(xlabel="b",ylabel="a",title = "ln|λ₂(∂u₁/∂n)²-λ₁(∂u₂/∂n)²|")
contour!(pl_crit,domain,domain,log.(criterion),c=:hsv,levels=n_contour_levels)
savefig(pl_crit,joinpath(output_dir,"criterion_β$(β)_N$(N)_$(l_well_depth)_$(well_depth)_$(r_well_depth).pdf"))

pl_domain = plot(xlabel="q",ylabel="V",title="Optimal domain")
plot!(pl_domain,V,0,1,label="",color=:black)
vline!(pl_domain,hstar,color=:red,label="direct")
vline!(pl_domain,argmin_crit,color=:blue,linestyle=:dot,label="first-order condition")
savefig(pl_domain,joinpath(output_dir,"domain_β$(β)_N$(N)_$(l_well_depth)_$(well_depth)_$(r_well_depth).pdf"))