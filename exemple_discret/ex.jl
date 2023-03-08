using LinearAlgebra


function get_eigen(n1,n12,n2,α)#two states
    N1=n1+n12
    N2=n2+n12
    N=n1+n2+n12
    P=[n1-N1 n1 0; n12 n12-N-α n12; 0 n2 n2-N2-α]
    return eigen(-P)
end

n1 = 1000
γ = 1/n1
ϕ = 