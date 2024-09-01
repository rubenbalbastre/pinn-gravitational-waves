# TODO: not in use

using NLsolve


function E_kerr(p::Float64, e::Float64, M::Float64, a::Float64)::Float64
    """
    Energy Kerr time-like geodesic
    """
    res = sqrt((M^4*p^3*(-2 - 2*e + p)*(-2 + 2*e + p)*(-3 - e^2 + p) - 
    a^2*(-1 + e^2)^2*M^2*p^2*(-5 + e^2 + 3*p) - 
    2*sqrt(a^2*(-1 + e^2)^4*M^2*p^3*(a^4*(-1 + e^2)^2 + 
        M^4*(-4*e^2 + (-2 + p)^2)*p^2 + 
        2*a^2*M^2*p*(-2 + p + e^2*(2 + p)))))/(M^2*p^3*(-4*a^2*(-1 + 
         e^2)^2 + M^2*(3 + e^2 - p)^2*p)))
    return res
end


function L_kerr(p::Float64, e::Float64, M::Float64, a::Float64)::Float64
    """
    Angular momentum Kerr time-like geodesic
    """
    res = sqrt((M^4*p^3*(-2 - 2*e + p)*(-2 + 2*e + p)*(-3 - e^2 + p) - 
        a^2*(-1 + e^2)^2*M^2*p^2*(-5 + e^2 + 3*p) - 
        2*sqrt(a^2*(-1 + e^2)^4*M^2*p^3*(a^4*(-1 + e^2)^2 + 
        M^4*(-4*e^2 + (-2 + p)^2)*p^2 + 
        2*a^2*M^2*p*(-2 + p + e^2*(2 + p)))))/(M^2*p^3*(-4*a^2*(-1 + e^2)^2 + 
        M^2*(3 + e^2 - p)^2*p)))*(a^4*(-1 + e^2)^4 + 
        a^2*(-1 + e^2)^2*M^2*p*(-4 + 3*p + e^2*(4 + p)) - sqrt(
        a^2*(-1 + e^2)^4*M^2*p^3*(a^4*(-1 + e^2)^2 + 
        M^4*(-4*e^2 + (-2 + p)^2)*p^2 + 
        2*a^2*M^2*p*(-2 + p + e^2*(2 + p)))))/(a^3*(-1 + e^2)^4 - 
        a*(-1 + e^2)^2*M^2*(-4*e^2 + (-2 + p)^2)*p)
    return res
end


function compute_drdτ_in_kerr_metric(e::Float64, χ::Float64, p::Float64, L::Float64, E::Float64, r::Float64)::Float64
    """
    Compute in a way such that numerical issues are avoided [not sure why this happened]

    Original formula:
    x = (r^2*E^2 + 2*M*(a*E-L)^2/r + (a^2*E^2 - L^2) - Δ)/r^2
    """

    x_1 = e * sin(χ) * sqrt( (p-6-2*e*cos(χ))/(p*(p-3-e^2)))

    x1 = x_1^2

    x2 = (2*M/r*(2*a*E*L+a^2*E^2) + a^2*(E^2 - 1))/r^2
    x = x1 + x2
    drdτ = real(sqrt(x))


    return drdτ

end


function RelativisticOrbitModel_Kerr_EMR_numerically(u, model_params, t)
    """
    Defines system of odes which describes motion of
    point like particle in schwarzschild background, uses

    u[1] = χ
    u[2] = ϕ
    """
    χ, ϕ = u
    p, M, e, a  = model_params

    E, L = numerically_compute_E_L_kerr(p, M , e)

    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2

    Δ = r^2 - 2*M*r + a^2
    dϕdτ = ((1-2*M/r)*L + 2*M*a*E/r) / Δ
    dtdτ = ((r^2 + a^2 + 2*M*a^2/r)*E - 2*M*a*L/r) / Δ

    # TODO: this was introduced to avoid some numerical issues [not sure why this happened]
    drdτ = compute_drdτ_in_kerr_metric(e, χ, p, L, E, r)

    ϕ̇ = dϕdτ / dtdτ
    χ̇ = drdτ / (dtdτ * drdχ)

    return [χ̇, ϕ̇,]
end


function NNOrbitModel_Kerr_EMR_numerically(u, model_params, t; NN=nothing, NN_params=nothing)
    """
    Defines system of odes which describes motion of
    point like particle with Newtonian physics, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants
    """
    χ, ϕ = u
    p, M, e, a  = model_params

    E, L = numerically_compute_E_L_kerr(p, M , e)

    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2

    Δ = r^2 - 2*M*r + a^2
    dϕdτ = ((1-2*M/r)*L + 2*M*a*E/r) / Δ
    dtdτ = ((r^2 + a^2 + 2*M*a^2/r)*E - 2*M*a*L/r) / Δ

    # TODO: this was introduced to avoid some numerical issues [not sure why this happened]
    drdτ = compute_drdτ_in_kerr_metric(e, χ, p, L, E, r)

    ϕ̇ = dϕdτ / dtdτ
    χ̇ = drdτ / (dtdτ * drdχ)

    return [χ̇, ϕ̇]
end


function f_kerr!(F, x; p, M, e)
    """
    Evaluate the expressions for angular momentum L (=x[1]) and energy E (=x[2]) for Kerr metric
    """
    L, E = x
    F[1] = - a^2 + a^2*E^2 - L^2 + (2 * (1 + e) * (a * E - L)^2)/p + (2 *M^2 *p)/(1 + e) - (M^2* p^2)/(1 + e)^2 + (E^2*M^2*p^2)/(1 + e)^2
    F[2] = - a^2 + a^2*E^2 - L^2 + (2 * (1 - e) * (a * E - L)^2)/p + (2 *M^2 *p)/(1 - e) - (M^2* p^2)/(1 - e)^2 + (E^2*M^2*p^2)/(1 - e)^2
end


function j_kerr!(J, x; p, M, e)
    """ 
    Calculate the Jacobian matrix for angular momentum L (=x[1]) and energy E (=x[2]) for Schwarzschild metric
    """
    L, E = x
    J[1, 1] = -2*L - 4*(1+e)*(a*E-L) / p
    J[1, 2] = 2*a^2*E + 4*a*(1+e)*(a*E-L)/p + 2*E*M^2*p^2/(1+e)^2
    J[2, 1] = -2*L - 4*(1-e)*(a*E-L)/p
    J[2, 2] = 2*a^2*E + 4*a*(1-e)*(a*E-L)/p + 2*E*M^2*p^2/(1-e)^2
end


function numerically_compute_E_L_kerr(p, M, e)
    """
    Numerically compute E and L.
    # obtain numerically E and L from initial values E₀ and L₀. Convergence was difficult so exact calulation is used instead
    # NOTE: not used
    """


    # f_kerr(F, x) = f_kerr!(F, x, p=p, M=M, e=e)
    # j_kerr(F, x) = j_kerr!(F, x, p=p, M=M, e=e)

    # L₀ =  
    # E₀ = 

    # numerical_solution = nlsolve(f_kerr, j_kerr, [ L₀; E₀])

    # E = numerical_solution.zero[2]
    # L = numerical_solution.zero[1]

    # return E, L
end
