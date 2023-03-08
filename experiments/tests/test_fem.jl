using Plots , LinearAlgebra, Cubature

include("QSDs.jl")

plot()
N=5
h=1/N
D=collect(Float64,range(0,1,N+1))
elements=[P1_element_diff(D[i],D[i+1],D[i+2]) for i=1:N-1]
boundary_element=boundary_P1_element_diff(D[N],1.0,0.0,D[2]) #P1 element on the boundary
for (i,f)= enumerate(elements)
    plot!(f,0,1,label="$i")
end
plot!(boundary_element,0,1,label="$N")
xlims!(0,1)
