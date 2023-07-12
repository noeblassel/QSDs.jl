### Author: Noé Blassel
### Date: 11 Jul. 2023


using Plots, Base.Threads, Random


## overdamped langevin simulator

Base.@kwdef struct OverdampedLangevinSimulator{F}
    dt::Float64
    β::Float64
    ∇V::F
    n_steps=1 #simulation steps per iteration
    σ = √(2dt/β)
end

function update_state!(X,simulator::OverdampedLangevinSimulator;rng=Random.GLOBAL_RNG)
    for k=1:simulator.n_steps
        X .= X-simulator.∇V(X)*simulator.dt + simulator.σ*randn(rng,size(X)...)
    end
end


## Lennard-Jones with confining potential

Base.@kwdef struct LJClusterInteraction2D{N}
    σ=1.0
    σ6=σ^6
    σ12=σ6^2
    ε=1.0
    α=1.0 # sharpness of harmonic confining potential
     # for multithread force computation
end

function lj_energy(X,inter::LJClusterInteraction2D{N}) where {N}
    V = 0.0
    for i=1:N-1
        for j=2:N
            inv_r6=inv(sum(abs2,X[:,i]-X[:,j]))^3
            V += (inter.σ12*inv_r6^2-inter.σ6*inv_r6)
        end
    end
    return 4inter.ε*V+inter.α*sum(abs2,X)/2 # add confining potential
end

function lj_grad(X,inter::LJClusterInteraction2D{N}) where {N}
    F_threaded = fill(zeros(2,N),nthreads())
    @threads for i=1:N-1
        r = zeros(2)
        f = zeros(2)
        for j=i+1:N
            r = X[:,i]-X[:,j]
            inv_r2 = inv(sum(abs2,r))
            inv_r4 = inv_r2^2
            inv_r8 = inv_r4^2
            
            f = (6inter.σ6 - 12inter.σ12*inv_r2*inv_r4)*inv_r8*r

            F_threaded[threadid()][:,i] .+= f
            F_threaded[threadid()][:,j] .-= f
        end
    end

    return 4inter.ε*sum(F_threaded) + inter.α*X #add confining potential
end

## Cluster test

using Plots

N_cluster = 7
inter = LJClusterInteraction2D{N_cluster}()
sim = OverdampedLangevinSimulator(dt = 1e-4,β = 1.0,∇V = (x->lj_grad(x,inter)),n_steps=100)
X = zeros(2,N_cluster)

k = ceil(Int,sqrt(N_cluster))

for i=1:N_cluster
    X[1,i] = i ÷ k
    X[2,i] = i % k
end

X .-= [sum(X[1,:]);sum(X[2,:])]/N_cluster

## visualize

nframes = 1000
anim= @animate for i=1:nframes
    (i%10 == 0) && println(i,"/",nframes)
    update_state!(X,sim)
    scatter(X[1,:],X[2,:],label="",xlims=(-k,k),ylims=(-k,k))
end

mp4(anim,"lj_cluster.mp4")

## test speed
niter = 1000 # note this is actually 100000 simulation steps since sim.n_steps=100
println("speed test:")
@time for i=1:niter
    (i%10 == 0) && println(i,"/",niter)
    update_state!(X,sim)
end

## lots of room for optimization