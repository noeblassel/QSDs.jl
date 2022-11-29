"""
A piecewise-constant killing rate.
values : the values of the killing rate
domain : a representation of the subsets on which the kill rate is constant
fixed_indices : the indices of the parameters which are fixed
fixed_values : the corresponding fixed values
"""
struct SoftKillingRate{S,T,U}
    values::S
    domain::T
    fixed_indices::Int64
    fixed_values::U
end


"""
    Returns the derivative of eigenvalue λ with respect to α[ix] of the Overdamped Langevin generator with soft killing α.

    u is the associated eigenvector.

    ix is the index of the parameter with respect to which we compute the derivative.

    The diag_weights correspond to the concatenations of the integrals ∫ϕi^2 dm, on [q_i,q_i+1] and [q_i+1,q_i+2]
    where m is the Lebesgue measure in the Schrodinger formulation, or the Gibbs measure
    in the standard formulation.

    The off_diag_weights argument corresponds to the integrals ∫ϕiϕi+1 dm.
"""
function ∂λ(u,ix; diag_weights,off_diag_weights)
    N = length(u)
    j = (ix == 1) ? N : ix-1
    return u[ix]^2 * diag_weights[2ix-1] + u[j]^2 * diag_weights[2j] + 2off_diag_weights[j]*u[ix]*u[j]
end

function ∇λ(u,α::SoftKillingRate; diag_weights,off_diag_weights)
    gradient=zero(α.values)
    N=length(gradient)

    for ix=1:N
        gradient[i]=∂λ(u,ix; diag_weights,off_diag_weights)
    end

    for ix=α.fixed_indices
        gradient[ix]=zero(gradient[ix])
    end

    return gradient
end


"""
The gradient of a scalar function of λ1 and λ2 with respect to α.
u1 and u2 are the eigenvectors corresponding respectively to λ1 and λ2.
f is the objective function,
∂1f and ∂2f are the partial derivatives of f with respect to λ1 and λ2 respectively.
α is the killing rate.
"""
function ∇f(f,∂1f, ∂2f,λ1,λ2,u1,u2,α::SoftKillingRate; diag_weights,off_diag_weights)
    ∇λ1 = ∇λ(u1,α; diag_weights=diag_weights,off_diag_weights=off_diag_weights)
    ∇λ2 = ∇λ(u2,α; diag_weights=diag_weights, off_diag_weights=off_diag_weights)

    return ∂1f(λ1,λ2)*∇λ1 + ∂2f(λ1,λ2)*∇λ2
end

