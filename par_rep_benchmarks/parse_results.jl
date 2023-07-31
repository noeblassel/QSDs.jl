using Plots, Statistics

log_dir ="logs_cold"

types = [Bool,Int64,Int64,Float64,Float64]
files = ["is_metastable.bool","state_from.int64","state_to.int64","transition_time.f64","exit_configuration.vec2f64"]

is_metastable,state_from,state_to,transition_time,exit_configuration = [reinterpret(T,read(joinpath(log_dir,f))) for (T,f)=zip(types,files)]
exit_configuration = reshape(exit_configuration,2,length(exit_configuration)÷2)

cdf_comp(t,series) = begin m = mean(series .> t); return (m>0) ? m : NaN end

# tmax = 20

grid = [(i!=j) ? plot(t-> cdf_comp(t,transition_time[@. (state_from == i) && (state_to == j)] ),yaxis=:log,xlabel="t",ylabel="prob",0,maximum(transition_time[@. (state_from == i) && (state_to == j)]),title="$i → $j",label="") : plot(axis=false,grid=false) for i=1:3,j=1:3]
plot(collect(grid)...,layout = (3,3),size=(1000,1000))
savefig("hists_cold.pdf")

# function entropic_switch(x, y)
#     tmp1 = x^2
#     tmp2 = (y - 1 / 3)^2
#     return 3 * exp(-tmp1) * (exp(-tmp2) - exp(-(y - 5 / 3)^2)) - 5 * exp(-y^2) * (exp(-(x - 1)^2) + exp(-(x + 1)^2)) + 0.2 * tmp1^2 + 0.2 * tmp2^2
# end

# xlims = -2.5, 2.5
# ylims = -1.75, 2.5
# xrange = range(xlims..., 200)
# yrange = range(ylims..., 200)

# contourf(xrange, yrange, entropic_switch, levels=50, cmap=:hsv)

# scatter!(exit_configuration[1,:],exit_configuration[2,:])
# savefig("exits.pdf")