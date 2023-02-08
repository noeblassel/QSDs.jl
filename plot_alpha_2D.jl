using Plots, Colors

input_file = "opt_alpha_5.0.out"

include(input_file)

V(x,y)= cos(2π*x)-cos(2π*(y-x))

function plot_trif(X,Y,f,t,fmin,fmax; crange=colormap("RdBu"),kwargs...)
    pl =plot(;kwargs...)
    tmp_f = copy(f)
    clamp!(tmp_f,fmin,fmax)
    (_,Ntri) = size(t)

    for n=1:Ntri

        (i,j,k) = T[:,n]

        c = crange[1+floor(Int64,(length(crange)-1)*(tmp_f[n]-fmin)/(fmax-fmin))]

        plot!(pl,Shape(X[[i,j,k]],Y[[i,j,k]]),label="",fillcolor=c,linecolor=nothing)
    end

    return pl
end

function to_trif(u,t)
    (_,Ntri) = size(t)
    return [sum(u[t[:,n]])/3 for n=1:Ntri]
end