using Plots, Colors

input_file = ARGS[1]

include(input_file)

V(x,y)= cos(2π*x)-cos(2π*(y-x))

cmap = colormap("RdBu")

pl = plot(aspect_ratio=1,size=(800,800))

(_,Ntri) = size(T)

for n=1:Ntri
    (i,j,k) = T[:,n]
    log_α_n = log_α_star[n]
    c = cmap[round(Int64,length(cmap)*(log_α_n-log_α_min)/(log_α_max))]
    plot!(pl,Shape(X[[i,j,k]],Y[i,j,k]),label="",fillcolor=c,lw=0)
end

plot(pl)