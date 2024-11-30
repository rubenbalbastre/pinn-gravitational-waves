

function RelativisticOrbitModel_Kerr_EMR(u, model_params, t)
    """
    Defines system of odes which describes motion of
    point like particle in schwarzschild background, uses

    u[1] = χ
    u[2] = ϕ
    """
    χ, ϕ = u
    p, M, e, a = model_params

    # angular momentum, energy
    L =  L_kerr(p, e, M, a)
    E = E_kerr(p, e, M, a)

    # auxiliary
    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2
    Δ = r^2 - 2*M*r + a^2

    # definition Kerr geodesic
    dϕdτ = ((1-2*M/r)*L + 2*M*a*E/r) / Δ
    dtdτ = ((r^2 + a^2 + 2*M*a^2/r)*E - 2*M*a*L/r) / Δ
    drdτ = compute_drdτ_in_kerr_metric(χ, a, Δ, L, E, r, M)

    ϕ̇ = dϕdτ / dtdτ
    χ̇ = drdτ / (dtdτ * drdχ)

    return [χ̇ , ϕ̇ ]
end


function NNOrbitModel_Kerr_EMR(u, model_params, t; NN=nothing, NN_params=nothing)
    """
    Defines system of odes which describes motion of
    point like particle with Kerr background, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, e and a are constants.
    """

    χ, ϕ = u
    p, M, e, a = model_params

    neural_network_input = [χ, ϕ, a, p, M, e]

    if isnothing(NN)
        nn = [1,1]
    else
        # instead of introducing 'u' as neural network input we 
        # include a new vector that includes spin parameter a
        nn = 1 .+ NN(neural_network_input, NN_params)
    end

    # angular momentum, energy
    L =  L_kerr(p, e, M, a)
    E = E_kerr(p, e, M, a)

    # auxiliary
    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2
    Δ = r^2 - 2*M*r + a^2

    # definition Kerr geodesic
    dϕdτ = ((1-2*M/r)*L + 2*M*a*E/r) / Δ
    dtdτ = ((r^2 + a^2 + 2*M*a^2/r)*E - 2*M*a*L/r) / Δ
    drdτ = compute_drdτ_in_kerr_metric(χ, a, Δ, L, E, r, M)

    ϕ̇ = (dϕdτ / dtdτ) * nn[1]
    χ̇ = (drdτ / (dtdτ * drdχ)) * nn[2]

    return [χ̇, ϕ̇,] 
end


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


function compute_drdτ_in_kerr_metric(χ::Float64, a::Float64, Δ:: Float64, L::Float64, E::Float64, r::Float64, M::Float64)::Float64
    """
    Compute in a way such taht numerical issues are avoided [not sure why this happened]

    Original formula:
    x = (r^2*E^2 + 2*M*(a*E-L)^2/r + (a^2*E^2 - L^2) - Δ)/r^2
    """

    x = (r^2*E^2 + 2*M*(a*E-L)^2/r + (a^2*E^2 - L^2) - Δ)/r^2
    drdτ = sqrt(x)

    # rule based on schwarzschild extreme case where a = 0 -> ask Rubén for more details
    if sin(χ) < 0
        drdτ = - drdτ
    end
    
    return drdτ

end
