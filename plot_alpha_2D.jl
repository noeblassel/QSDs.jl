using Plots, Colors

input_file = "output_alpha_3.out"

include(input_file)

V(x,y)= cos(2π*x)-cos(2π*(y-x))
grad_norm_V(x,y) = 2π*sqrt((sin(2π*x)+sin(2π*(y-x)))^2 + sin(2π*(y-x))^2)
cmap = colormap("RdBu")

pl = plot(aspect_ratio=1,size=(800,800),xlabel="x",ylabel="y",)

(_,Ntri) = size(T)
log_α_min = -8
log_α_max = 8

log_α_star[ log_α_star .< log_α_min ] .= log_α_min
log_α_star[ log_α_star .> log_α_max] .= log_α_max

for n=1:Ntri
    (i,j,k) = T[:,n]

    if (i ∈ dirichlet_boundary_points) && (j ∈ dirichlet_boundary_points) && (k ∈ dirichlet_boundary_points)
        log_α_star[n] = log_α_max
    end

    log_α_n = log_α_star[n]
    c = cmap[1+floor(Int64,(length(cmap)-1)*(log_α_n-log_α_min)/(log_α_max-log_α_min))]
    plot!(pl,Shape(X[[i,j,k]],Y[[i,j,k]]),label="",fillcolor=c,lw=0,color=c)
end

plot(pl)