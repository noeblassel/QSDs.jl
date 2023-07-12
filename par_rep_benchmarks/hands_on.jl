include("ParRep.jl"); using .ParRep, Base.Threads,Random, Plots

Base.@kwdef mutable struct GelmanRubinDiagnostic{F}
    observables::F
    means = Matrix{Float64}(undef,1,1)
    sq_means = Matrix{Float64}(undef,1,1)
    burn_in = 1
    tol = 5e-2
end

function ParRep.check_dephasing!(checker::GelmanRubinDiagnostic,replicas::Vector,current_macrostate,step_n)
    
    if step_n == 1 # initialize running mean and running square_mean buffer
        checker.means = zeros(length(checker.observables),length(replicas))
        checker.sq_means = copy(checker.means)
    end
    
    @threads for i=1:length(replicas)
        r = replicas[i]
        for (j,f)=enumerate(checker.observables)
            val = f(r,current_macrostate)
            sq_val = val^2

            checker.means[j,i] += (val-checker.means[j,i]) / step_n
            checker.sq_means[j,i] += (sq_val-checker.sq_means[j,i]) / step_n

        end
    end

    (step_n < checker.burn_in) && return false

    Obar = sum(checker.means;dims = 2) / length(replicas)

    numerator = sum(@. (checker.sq_means -2checker.means*Obar + Obar^2);dims=2)
    denominator = sum(checker.sq_means - checker.means .^ 2;dims=2)

    R = maximum(numerator ./ denominator) - 1
    # push!(checker.gr_hist,R)
    return (R < checker.tol)
end
Base.@kwdef mutable struct SteepestDescentState{X}
    η = 0.1
    dist_tol = 1e-2
    grad_tol = 1e-3
    minima = X[]
    ∇V::Function
end

function ParRep.get_macrostate!(checker::SteepestDescentState,microstate,current_macrostate)
    x = copy(microstate)
    grad_norm = Inf
    min_iter,iter = 20,1
    while true
        grad = checker.∇V(x)
        x .-= checker.η * grad # gradient descent step

        for (i,m)=enumerate(checker.minima)
            dist = √sum(abs2,m-x)
            if dist < checker.dist_tol
                return i
            end
        end

        grad_norm = √sum(abs2,grad)
        (grad_norm < checker.grad_tol) && (iter > min_iter) && break
        iter +=1
    end

    push!(checker.minima,x)
    return length(checker.minima)
end
const A = [-200,-100,-170,15]
const a = [-1,-1,-6.5,0.7]
const b = [0,0,11,0.6]
const c = [-10,-10,-6.5,0.7]
const x0 = [1,0,-0.5,-1]
const y0 = [0,0.5,1.5,1]

mueller_brown(x,y) = sum(@. A*exp(a*(x-x0)^2 + b*(x-x0)*(y-y0)+c*(y-y0)^2))
mueller_brown(X) = mueller_brown(X...)

function grad_mueller_brown(x,y)
    v = @. A*exp(a*(x-x0)^2 + b*(x-x0)*(y-y0)+c*(y-y0)^2)
    return [sum(@. (2a*(x-x0)+b*(y-y0))*v), sum(@. (2c*(y-y0)+b*(x-x0))*v) ]
end

grad_mueller_brown(X) = grad_mueller_brown(X...)
grad_mueller_brown(0,0)

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
        X .= X-simulator.∇V(X)*simulator.dt + simulator.σ*randn(simulator.rng,size(X)...)
    end
end

struct ExitEventKiller end
ParRep.check_death(::ExitEventKiller,state_a,state_b,_) = (state_a != state_b)

xlims = -1.2,0.78
ylims = -0.3,1.9

xrange = range(xlims...,200)
yrange = range(ylims...,200)
contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv)


Base.@kwdef mutable struct AnimationLogger2D
    anim = Plots.Animation()
end

function ParRep.log_state!(logger::AnimationLogger2D,step; kwargs...)
    if step == :initialization
        f=contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv,title="Initialize",xlims=xlims,ylims=ylims)
        ref_walker = kwargs[:algorithm].reference_walker
        scatter!(f,[ref_walker[1]],[ref_walker[2]],color=:blue,label="",markersize=4)
        frame(logger.anim,f)
    elseif step == :dephasing
        f=contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv,title="Dephase/decorrelate",xlims=xlims,ylims=ylims)
        reps = kwargs[:algorithm].replicas
        ref_walker = kwargs[:algorithm].reference_walker
        scatter!(f,[ref_walker[1]],[ref_walker[2]],color=:blue,label="",markersize=4)
        xs,ys = [rep[1] for rep in reps],[rep[2] for rep in reps]
        scatter!(f,xs,ys,color=[(i in kwargs[:killed_ixs]) ? :red : :blue for i in 1:kwargs[:algorithm].N],markersize=2,label="")
        frame(logger.anim,f)
    elseif step == :parallel
        f=contourf(xrange,yrange,mueller_brown,levels=50,clims=(-150,200),cmap=:hsv,title="Parallel exit",xlims=xlims,ylims=ylims)
        reps = kwargs[:algorithm].replicas
        ref_walker = kwargs[:algorithm].reference_walker
        scatter!(f,[ref_walker[1]],[ref_walker[2]],color=:blue,label="",markersize=4)
        xs,ys = [rep[1] for rep in reps],[rep[2] for rep in reps]
        scatter!(f,xs,ys,color=:blue,markersize=2,label="")
        frame(logger.anim,f)
    end
end

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
        push!(logger.exit_configuration,alg.reference_walker)
    end
end

# logger = AnimationLogger2D()
logger = TransitionLogger{Vector{Float64}}()
ol_sim = OverdampedLangevinSimulator(dt = 1e-4,β = 0.1,∇V = grad_mueller_brown,n_steps=10)
state_check = SteepestDescentState{Vector{Float64}}(η=1e-4,∇V = grad_mueller_brown)
# gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->(x[1]-state_check.minima[i][1]),(x,i)->(x[2]-state_check.minima[i][2])],tol=0.05)
gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->mueller_brown(x)],tol=0.05)

alg = GenParRepAlgorithm(N=64,
                        simulator = ol_sim,
                        dephasing_checker = gelman_rubin,
                        macrostate_checker = state_check,
                        replica_killer = ExitEventKiller(),
                        logger = logger,
                        reference_walker = [-0.5,1.5])

ParRep.simulate!(alg,5000)

f=open("log.txt","w")
println(f,"state_from=",logger.state_from)
println(f,"state_to=",logger.state_to)
println(f,"transition_time=",logger.transition_time*ol_sim.dt*ol_sim.n_steps)
println(f,"is_metastable=",logger.is_metastable)
println(f,"exit_configuration=",logger.exit_configuration)
close(f)
# mp4(logger.anim,"anims/movie_short.mp4",fps=4)



## TODO: log transition events (state_from , state_to, elapsed_time from previous transition event, )
## → get histograms of transition times
## → interface with Molly for LJ clusters
## → 