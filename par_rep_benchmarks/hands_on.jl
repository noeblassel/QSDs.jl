include("ParRep.jl"); using .ParRep, Base.Threads,Random

Base.@kwdef mutable struct GelmanRubinDiagnostic{F}
    observables::F
    means = Matrix{Float64}(undef,1,1)
    sq_means = Matrix{Float64}(undef,1,1)
    burn_in = 1
    tol = 5e-2
end

function ParRep.check_dephasing!(checker::GelmanRubinDiagnostic,replicas::Vector{X},current_macrostate,step_n) where {X}
    
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

function ParRep.get_macrostate!(checker::SteepestDescentState,microstate::X,current_macrostate) where {X}
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

ol_sim = OverdampedLangevinSimulator(dt = 1e-4,β = 0.09,∇V = grad_mueller_brown,n_steps=1)
state_check = SteepestDescentState{Vector{Float64}}(η=1e-4,∇V = grad_mueller_brown)
gelman_rubin = GelmanRubinDiagnostic(observables=[(x,i)->(x[1]-state_check.minima[i][1]),(x,i)->(x[2]-state_check.minima[i][2])],tol=0.05)

alg = GenParRepAlgorithm(N=32,
                        simulator = ol_sim,
                        dephasing_checker = gelman_rubin,
                        macrostate_checker = state_check,
                        replica_killer = ExitEventKiller(),
                        reference_walker = [-0.5,1.5])

ParRep.simulate!(alg,1000)