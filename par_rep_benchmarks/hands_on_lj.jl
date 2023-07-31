include("ParRep.jl"); using .ParRep, Base.Threads,Random


n_transitions = parse(Int64,ARGS[1])
freq_checkpoint = parse(Int64,ARGS[2])
n_replicas = parse(Int64,ARGS[3])



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

Base.@kwdef mutable struct SteepestDescentStateSym{X,E,F} # determine state by steepest descent (case of symetric energy landscape) 
    η = 5e-4
    grad_tol = 1e-2
    e_tol = 1e-1
    minima = X[]
    minimums = Float64[]
    V::E
    ∇V::F
end

function ParRep.get_macrostate!(checker::SteepestDescentStateSym,walker,current_macrostate)
    x = copy(walker)
    grad_norm = Inf

    while (grad_norm > checker.grad_tol) #for k=1:checker.steps
        grad = checker.∇V(x)
        x -= checker.η * grad # gradient descent step
        grad_norm = maximum(abs.(grad))
    end
    
    if grad_norm < checker.grad_tol
        
        pot = checker.V(x)

        if isempty(checker.minimums)

            push!(checker.minima,x)
            push!(checker.minimums,pot)

            return 1
        else
            ΔEs = @. abs(pot - checker.minimums)
            imin = argmin(ΔEs)

            if ΔEs[imin] < checker.e_tol
                return imin
            else
                push!(checker.minima,x)
                push!(checker.minimums,pot)

                return length(checker.minima)
            end 
        end
    else # steepest descent has not converged in alloted number of iterations
        return nothing
    end
end


###### Define the interaction ######

Base.@kwdef struct LJClusterInteraction2D{N}
    σ=1.0
    σ6=σ^6
    σ12=σ6^2
    ε=1.0
    α=1.0 # sharpness of harmonic confining potential
end

function lj_energy(X,inter::LJClusterInteraction2D{N}) where {N}
    V = 0.0
    for i=1:N-1
        for j=i+1:N
            inv_r6=inv(sum(abs2,X[:,i]-X[:,j]))^3
            V += (inter.σ12*inv_r6-inter.σ6)*inv_r6
        end
    end
    return 4inter.ε*V+inter.α*sum(abs2,X)/2 # add confining potential
end

function lj_grad(X,inter::LJClusterInteraction2D{N}) where {N}
    F = zeros(2,N)
    for i=1:N-1
        r = zeros(2)
        f = zeros(2)
        for j=i+1:N
            r = X[:,i]-X[:,j]
            inv_r2 = inv(sum(abs2,r))
            inv_r4 = inv_r2^2
            inv_r8 = inv_r4^2
            
            f = (6inter.σ6 - 12inter.σ12*inv_r2*inv_r4)*inv_r8*r

            F[:,i] .+= f
            F[:,j] .-= f
        end
    end

    return 4inter.ε*F + inter.α*X #add confining potential
end


N_cluster = 7
inter = LJClusterInteraction2D{N_cluster}()


###### "Crystaline" initial configuration ######
X = zeros(2,N_cluster)

k = ceil(Int,sqrt(N_cluster))

for i=1:N_cluster
    X[1,i] = i ÷ k
    X[2,i] = i % k
end

X .-= (k-1)/2

###### Define the simulator ######

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
        X .+= simulator.σ*randn(simulator.rng,size(X)...)
    end
end

struct ExitEventKiller end
ParRep.check_death(::ExitEventKiller,state_a,state_b,_) = (state_a != state_b)

###### Define the logging output method ######

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

logger = TransitionLogger{typeof(X)}()

###### Declare everything ######

pot(x) = lj_energy(x,inter)
grad_pot(x) = lj_grad(x,inter)



sim = OverdampedLangevinSimulator(dt = 5e-4,β = 5.0,∇V = grad_pot,n_steps=state_check_freq)
state_checker = SteepestDescentStateSym{typeof(X),typeof(pot),typeof(grad_pot)}(V=pot,∇V = grad_pot,η=5e-3,grad_tol = 5e-2,e_tol=8e-2)
gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->x[j] for j=1:2N_cluster],tol=0.05)



alg = GenParRepAlgorithm(N=n_replicas,
                        simulator = sim,
                        dephasing_checker = gelman_rubin,
                        macrostate_checker = state_checker,
                        replica_killer = ExitEventKiller(),
                        logger = logger,
                        reference_walker = X)


if !isdir("logs")
    mkdir("logs")
end

ParRep.simulate!(alg,n_transitions)

# for k=1:(n_transitions ÷ freq_checkpoint)
#     println(k)
#     ParRep.simulate!(alg,freq_checkpoint)

#     write(open(joinpath("logs","state_from.int64"),"w"),logger.state_from)
#     write(open(joinpath("logs","state_to.int64"),"w"),logger.state_to)
#     write(open(joinpath("logs","transition_time.f64"),"w"),logger.transition_time*ol_sim.dt*ol_sim.n_steps)
#     write(open(joinpath("logs","is_metastable.bool"),"w"),logger.is_metastable)
#     write(open(joinpath("logs","exit_configuration.vec2f64"),"w"),stack(logger.exit_configuration))

#     alg.n_transitions = 0
# end

# mp4(logger.anim,"anims/movie_short.mp4",fps=4)



## TODO: log transition events (state_from , state_to, elapsed_time from previous transition event, )
## → get histograms of transition times
## → interface with Molly for LJ clusters
## → 