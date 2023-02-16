using Plots, Colors, TriplotRecipes

β = ARGS[1]

input_dir = "results_optim"
input_filename = "opt_alpha_$(β)"

include(joinpath(input_dir,input_filename*".out"))

V(x, y) = cos(2π * x) - cos(2π * (y - x))
Vmin = -2
Vmax = 2
α_max = 20000
Vnorm(x,y) = α_max*(V(x,y)-Vmin) /(Vmax-Vmin)

function plot_trif(X, Y, f, t, fmin, fmax; crange=colormap("RdBu"), kwargs...)

    pl = plot(;kwargs...)

    tmp_f = copy(f)
    clamp!(tmp_f, fmin, fmax)
    (_, Ntri) = size(t)

     for n = 1:Ntri

        (i, j, k) = T[:, n]

        c = crange[1+floor(Int64, (length(crange) - 1) * (tmp_f[n] - fmin) / (fmax - fmin))]

        plot!(pl, Shape(X[[i, j, k]], Y[[i, j, k]]), label="", fillcolor=c, linecolor=nothing)
    end

    return pl
end

function to_trif(u, t)
    (_, Ntri) = size(t)
    return [sum(u[t[:, n]]) / 3 for n = 1:Ntri]
end

function to_verf(α,X,Y,t)
    N=length(X)
    weights=zeros(N)
    verf=zeros(N)
    _,Ntri=size(t)

    for n=1:Ntri
        i,j,k = t[:,n]

        cx = sum(X[[i,j,k]])/3
        cy = sum(Y[[i,j,k]])/3

        di =sqrt((X[i]-cx)^2+(Y[i]-cy)^2)
        dj=sqrt((X[j]-cx)^2+(Y[j]-cy)^2)
        dk =sqrt((X[k]-cx)^2+(Y[k]-cy)^2)
        verf[i] += di*α[n]
        verf[j] += dj*α[n]
        verf[k] += dk*α[n]
        
        weights[i] += di
        weights[j] += dj
        weights[k] += dk
        
    end
    return verf ./ weights
end

verf_α = to_verf(α_star,X,Y,T)

tripcolor(X,Y,verf_α,T,xlabel="x",ylabel="y",title="β=$(β)",aspectratio=1,size=(800,800))
q=range(-1,1,1000)
contour!(q,q,Vnorm,label="",cmap=:hsv,colorbar_entry=false)
output_dir = "results_alpha_2D"
savefig(joinpath(output_dir,input_filename*".pdf"))