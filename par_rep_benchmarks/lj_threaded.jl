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