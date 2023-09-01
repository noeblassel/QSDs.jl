using Plots, Statistics

dirs = ["logs_cold","logs_dns"]
labels = ["tol=0.05","dns"]
types = [Bool,Int64,Int64,Float64,Float64]
files = ["is_metastable.bool","state_from.int64","state_to.int64","transition_time.f64","exit_configuration.vec2f64"]
colors = [:red,:blue]

is_metastable,state_from,state_to,transition_time,exit_configuration = [[reinterpret(T,read(joinpath(dir,f))) for dir in dirs] for (T,f)=zip(types,files)]
exit_configuration = [reshape(conf,2,length(conf)÷2) for conf in exit_configuration]

cdf_comp(t,series;alpha = 0.05,pmin = 1e-5) = begin m = mean(series .>= t); n = length(series); e = sqrt(log(2/α)/2n); return (m>0) ? (max(pmin,m-e),m,m+e) : (NaN,NaN) end

# tmax = 20
α = 0.05 # confidence level
pmin = 1e-5

#,ribbon = sqrt(log(2/α)/2sum(@. (state_from == i) && (state_to == j))),ylims=(pmin,1)
#,ribbon = sqrt(log(2/α)/2sum(@. (state_from == i) && (state_to == j)))

series = [[(i !=j ) ? (s = transition_time[k][@. (state_from[k] == i) && (state_to[k] == j)] ; tmax = maximum(s); trange = range(0,tmax,10000); X = cdf_comp.(trange,(s,)); [first.(X) getindex.(X,2) last.(X) trange] ) : [0.0] for i=1:3,j=1:3] for k=1:length(dirs)]


# show(series)
grid = [(i!=j) ? plot(yaxis=:log,xlabel="t",ylabel="prob",title="$i → $j",ylims=(pmin,1)) : plot(axis=false,grid=false) for i=1:3,j=1:3]


for k=1:length(dirs)
    for i=1:3,j=1:3
        if i!=j
            plot!(grid[i,j],series[k][i,j][:,4],series[k][i,j][:,3],fillrange = series[k][i,j][:,1],fillcolor=colors[k],fillalpha=0.1,alpha=0.0,label="")
            plot!(grid[i,j],series[k][i,j][:,4],series[k][i,j][:,2],label=labels[k],color=colors[k])
        end
    end
end
plot(collect(grid)...,layout = (3,3),size=(1000,1000))
savefig("hists.pdf")

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

# for k =1:length(dirs)
#     scatter!(exit_configuration[k][1,:],exit_configuration[k][2,:],color=colors[k],markersize=0.5,markerstrokewidth=0,markeralpha=0.1,label="")
# end
# savefig("exits_dns.pdf")
