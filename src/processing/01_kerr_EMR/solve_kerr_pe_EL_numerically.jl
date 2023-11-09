cd(@__DIR__)
using Pkg; Pkg.activate("../../"); Pkg.instantiate();
using NLsolve

e = 0.1
p = 10.0
M = 1.0

function E(p::Float64, e::Float64,M::Float64)::Float64
    """
    Energy Schwarzschild time-like geodesic
    """
    res = sqrt(((p-2-2*e)*(p-2+2*e))/(p*(p-3-e^2)))
    return res
end

function L(p,e, M)
    """
    Angular momentum Schwarzschild time-like geodesic
    """
    res = p*M/sqrt(p-3-e^2)
    return res
end


function f!(F, x)
    # L: x[1]
    # E: x[2]
    F[1] = 1 + x[1]^2 /(p^2 * M^2) * (1 + e)^2 - 2/p * (1 + e) - 2*x[1]^2/(p^3*M^2)*(1 + e)^3 - x[2]^2
    F[2] = 1 + x[1]^2 /(p^2 * M^2) * (1 - e)^2 - 2/p * (1 - e) - 2*x[1]^2/(p^3*M^2)*(1 - e)^3 - x[2]^2
end

function j!(J, x)
    J[1, 1] = 2*x[1] / (p^2 * M^2) * (1 + e)^2 - 4*x[1]/(p^3*M^2)*(1 + e)^3
    J[1, 2] = -2*x[2]
    J[2, 1] = 2*x[1] / (p^2 * M^2) * (1 - e)^2 - 4*x[1]/(p^3*M^2)*(1 - e)^3
    J[2, 2] = -2*x[2]
end

println("L: " * string(L(p,e,M)))
println("E: " * string(E(p,e,M)))
x = nlsolve(f!, j!, [ 3; 0.6])
println(x)
println(x.zero)