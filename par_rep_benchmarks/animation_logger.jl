
# xlims = -1.2,0.78
# ylims = -0.3,1.9

# # xrange = range(xlims...,200)
# yrange = range(ylims...,200)
# contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv)


# Base.@kwdef mutable struct AnimationLogger2D
#     anim = Plots.Animation()
# end

# function ParRep.log_state!(logger::AnimationLogger2D,step; kwargs...)
#     if step == :initialization
#         f=contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv,title="Initialize",xlims=xlims,ylims=ylims)
#         ref_walker = kwargs[:algorithm].reference_walker
#         scatter!(f,[ref_walker[1]],[ref_walker[2]],color=:blue,label="",markersize=4)
#         frame(logger.anim,f)
#     elseif step == :dephasing
#         f=contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv,title="Dephase/decorrelate",xlims=xlims,ylims=ylims)
#         reps = kwargs[:algorithm].replicas
#         ref_walker = kwargs[:algorithm].reference_walker
#         scatter!(f,[ref_walker[1]],[ref_walker[2]],color=:blue,label="",markersize=4)
#         xs,ys = [rep[1] for rep in reps],[rep[2] for rep in reps]
#         scatter!(f,xs,ys,color=[(i in kwargs[:killed_ixs]) ? :red : :blue for i in 1:kwargs[:algorithm].N],markersize=2,label="")
#         frame(logger.anim,f)
#     elseif step == :parallel
#         f=contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv,title="Parallel exit",xlims=xlims,ylims=ylims)
#         reps = kwargs[:algorithm].replicas
#         ref_walker = kwargs[:algorithm].reference_walker
#         scatter!(f,[ref_walker[1]],[ref_walker[2]],color=:blue,label="",markersize=4)
#         xs,ys = [rep[1] for rep in reps],[rep[2] for rep in reps]
#         scatter!(f,xs,ys,color=:blue,markersize=2,label="")
#         frame(logger.anim,f)
#     end
# end

### steepest descent state checker

# Base.@kwdef mutable struct SteepestDescentState{X}
#     η = 0.1
#     dist_tol = 1e-2
#     grad_tol = 1e-3
#     minima = X[]
#     energies = Float64[]
#     ∇V::Function
#     V::Function
# end

# function ParRep.get_macrostate!(checker::SteepestDescentState,microstate,current_macrostate)
#     x = copy(microstate)
#     grad_norm = Inf
#     min_iter,iter = 20,1
#     while true
#         grad = checker.∇V(x)
#         x .-= checker.η * grad # gradient descent step

#         for (i,m)=enumerate(checker.minima)
#             dist = √sum(abs2,m-x)
#             if dist < checker.dist_tol
#                 return i
#             end
#         end

#         grad_norm = √sum(abs2,grad)
#         (grad_norm < checker.grad_tol) && (iter > min_iter) && break
#         iter +=1
#     end

#     push!(checker.minima,x)
#     push!(checker.energies,checker.V(x))

#     return length(checker.minima)
# end
# const A = [-200,-100,-170,15]
# const a = [-1,-1,-6.5,0.7]
# const b = [0,0,11,0.6]
# const c = [-10,-10,-6.5,0.7]
# const x0 = [1,0,-0.5,-1]
# const y0 = [0,0.5,1.5,1]

# mueller_brown(x,y) = sum(@. A*exp(a*(x-x0)^2 + b*(x-x0)*(y-y0)+c*(y-y0)^2))
# mueller_brown(X) = mueller_brown(X...)

# function grad_mueller_brown(x,y)
#     v = @. A*exp(a*(x-x0)^2 + b*(x-x0)*(y-y0)+c*(y-y0)^2)
#     return [sum(@. (2a*(x-x0)+b*(y-y0))*v), sum(@. (2c*(y-y0)+b*(x-x0))*v) ]
# end

# function grad_mueller_brown!(x,y,grad)
#     v = @. A*exp(a*(x-x0)^2 + b*(x-x0)*(y-y0)+c*(y-y0)^2)
#     grad[1] = sum(@. (2a*(x-x0)+b*(y-y0))*v)
#     grad[2] = sum(@. (2c*(y-y0)+b*(x-x0))*v)
# end

# grad_mueller_brown(X) = grad_mueller_brown(X...)