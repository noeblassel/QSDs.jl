using LinearAlgebra, Plots


function random_tridiag_eigens(N)
    X=zeros(N,N)
    X[diagind(X)] .= randn(N) .^ 2
    X[1,1]=X[N,N]=1
    u= -X[1,1]*rand()
    for i=1:N-1
        X[i,i+1]=u
        X[i+1,i]=u
        u= -(X[i+1,i+1]+u)
    end
    
    X=SymTridiagonal(X)
    λ,vs = eigen(X)
    return λ,vs
end

N=1000

 _λ,vs=random_tridiag_eigens(N)
 plot(vs[:,N-3:N],label="")