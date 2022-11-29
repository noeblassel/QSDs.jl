

"""
A struct representing a trigonometric polynomial of the form p(x)= a0+ sum(k=1::n) a_k*cos(kπx)+b_k*sin(kπx)
"""
struct TrigPolynomial{T,S<:AbstractVector{T}}
    a0::T
    as::S
    bs::S
end