include("ParRep.jl"); using .ParRep, Base.Threads,Random


# n_transitions = parse(Int64,ARGS[1])
# freq_checkpoint = parse(Int64,ARGS[2])
# n_replicas = parse(Int64,ARGS[3])
# pot_type = parse(ARGS[4])


### Gelma-Rubin convergence test 

Base.@kwdef mutable struct GelmanRubinDiagnostic{F}
    observables::F
    means = Matrix{Float64}(undef,1,1)
    sq_means = Matrix{Float64}(undef,1,1)
    burn_in = 1
    tol = 5e-2
end

function ParRep.check_dephasing!(checker::GelmanRubinDiagnostic,replicas,current_macrostate,step_n)
    
    if step_n == 1 # initialize running mean and running square_mean buffer
        checker.means = zeros(length(checker.observables),length(replicas))
        checker.sq_means = copy(checker.means)
    end
    
    @threads for i=1:length(replicas)
        r = replicas[i]
        for (j,f)=enumerate(checker.observables)
            val = f(r,current_macrostate)
            sq_val = val^2

            checker.means[j,i] += (val-checker.means[j,i]) / step_n # online way to compute the running mean
            checker.sq_means[j,i] += (sq_val-checker.sq_means[j,i]) / step_n

        end
    end

    (step_n < checker.burn_in) && return false

    Obar = sum(checker.means;dims = 2) / length(replicas)

    numerator = sum(@. (checker.sq_means -2checker.means*Obar + Obar^2);dims=2)
    denominator = sum(checker.sq_means - checker.means .^ 2;dims=2)

    R = maximum(numerator ./ denominator) - 1
    return (R < checker.tol)
end


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


### Entropic switch


"""
    entropic_switch(x, y)

Potential energy function.
"""
function entropic_switch(x, y)
    tmp1 = x^2
    tmp2 = (y - 1 / 3)^2
    return 3 * exp(-tmp1) * (exp(-tmp2) - exp(-(y - 5 / 3)^2)) - 5 * exp(-y^2) * (exp(-(x - 1)^2) + exp(-(x + 1)^2)) + 0.2 * tmp1^2 + 0.2 * tmp2^2
end

function entropic_switch(q)
    return entropic_switch(q...)
end

"""
    ∇entropic_switch(x, y)

Gradient of the potential energy function.
"""
function grad_entropic_switch(x, y)

    tmp1 = exp(4*x)
    tmp2 = exp(-x^2 - 2*x - y^2 - 1)
    tmp3 = exp(-x^2)
    tmp4 = exp(-(y-1/3)^2)
    tmp5 = exp(-(y-5/3)^2)

    dx = 0.8*x^3 + 10*(tmp1*(x - 1) + x + 1)*tmp2 - 6*tmp3*x*(tmp4 - tmp5)

    dy = 10*(tmp1 + 1)*y*tmp2 + 3*tmp3*(2*tmp5*(y - 5/3) - 2*tmp4*(y - 1/3)) + 0.8*(y - 1/3)^3

    return [dx, dy]
end

grad_entropic_switch(q) = grad_entropic_switch(q[1],q[2])


###


### Polyhedral states
const minima = [-1.0480549928242202 0.0 1.0480549928242202; -0.042093666306677734 1.5370820044494633 -0.042093666306677734]
const saddles = [-0.6172723078764598 0.6172723078764598 0.0; 1.1027345175080963 1.1027345175080963 -0.19999999999972246]
const pos_eigvecs = [0.6080988038706289 -0.6080988038706289 -1.0; 0.793861351075306 0.793861351075306 0.0]


struct PolyhedralStateChecker end

function ParRep.get_macrostate!(checker::PolyhedralStateChecker,walker,current_state)
    l1,l2,l3 =[(walker-saddles[:,i])'pos_eigvecs[:,i] for i=1:3]
    (l1 <= 0) && (l3 >= 0) && return 1
    (l1 >= 0) && (l2 >= 0) && return 2
    (l3 <= 0) && (l2 <= 0) && return 3
end
###

Base.@kwdef struct OverdampedLangevinSimulator{F,R}
    dt::Float64
    β::Float64
    ∇V::F
    n_steps=1
    σ = √(2dt/β)
    rng::R = Random.GLOBAL_RNG
end

function ParRep.update_microstate!(X,simulator::OverdampedLangevinSimulator)
    for k=1:simulator.n_steps
        X .-= simulator.∇V(X)*simulator.dt
        X[1] += simulator.σ*randn(simulator.rng)
        X[2] += simulator.σ*randn(simulator.rng)
    end
end

struct ExitEventKiller end
ParRep.check_death(::ExitEventKiller,state_a,state_b,_) = (state_a != state_b)

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

Base.@kwdef mutable struct TransitionLogger{X}
    state_from = Int[]
    state_to = Int[]
    transition_time = Int[]
    is_metastable = Bool[]
    exit_configuration = X[]
end

function ParRep.log_state!(logger::TransitionLogger,step; kwargs...)
    if step == :transition
        alg = kwargs[:algorithm]
        state_b = kwargs[:new_macrostate]
        state_a = kwargs[:current_macrostate]
        exit_time = kwargs[:exit_time]
        dephasing_time = kwargs[:dephasing_time]
        push!(logger.state_from,state_a)
        push!(logger.state_to,state_b)
        push!(logger.transition_time,exit_time+dephasing_time)
        push!(logger.is_metastable,exit_time != 0)
        push!(logger.exit_configuration,copy(alg.reference_walker))
    end
end

# logger = AnimationLogger2D()

function main()
    logger = TransitionLogger{Vector{Float64}}()
    ol_sim = OverdampedLangevinSimulator(dt = 1e-3,β = 3.0,∇V = grad_entropic_switch,n_steps=1)
    state_check = PolyhedralStateChecker()
    # gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->(x[1]-state_check.minima[i][1]),(x,i)->(x[2]-state_check.minima[i][2])],tol=0.05)
    gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->sum(abs,x-minima[:,i])],tol=0.05)

    n_replicas = 32

    alg = GenParRepAlgorithm(N=n_replicas,
                            simulator = ol_sim,
                            dephasing_checker = gelman_rubin,
                            macrostate_checker = state_check,
                            replica_killer = ExitEventKiller(),
                            logger = logger,
                            reference_walker = minima[:,1])

    log_dir = "logs_cold"


    if !isdir(log_dir)
        mkdir(log_dir)
    end

    n_transitions = 1_000_000
    freq_checkpoint = 100

   @time for k=1:(n_transitions ÷ freq_checkpoint)
        println(k)
        ParRep.simulate!(alg,freq_checkpoint;verbosity=0)

        write(open(joinpath(log_dir,"state_from.int64"),"w"),logger.state_from)
        write(open(joinpath(log_dir,"state_to.int64"),"w"),logger.state_to)
        write(open(joinpath(log_dir,"transition_time.f64"),"w"),logger.transition_time*ol_sim.dt*ol_sim.n_steps)
        write(open(joinpath(log_dir,"is_metastable.bool"),"w"),logger.is_metastable)
        write(open(joinpath(log_dir,"exit_configuration.vec2f64"),"w"),stack(logger.exit_configuration))

        alg.n_transitions = 0
    end
 

    # mp4(logger.anim,"anims/movie_short.mp4",fps=4)



    ## TODO: log transition events (state_from , state_to, elapsed_time from previous transition event, )
    ## → get histograms of transition times
    ## → interface with Molly for LJ clusters
    ## → 
end

main()