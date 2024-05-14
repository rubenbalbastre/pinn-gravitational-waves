# TODO: currently not in use


function f!(F, x)
    """
    Evaluate the expressions for angular momentum L (=x[1]) and energy E (=x[2]) for Schwarzschild metric
    """
    F[1] = 1 + x[1]^2 /(p^2 * M^2) * (1 + e)^2 - 2/p * (1 + e) - 2*x[1]^2/(p^3*M^2)*(1 + e)^3 - x[2]^2
    F[2] = 1 + x[1]^2 /(p^2 * M^2) * (1 - e)^2 - 2/p * (1 - e) - 2*x[1]^2/(p^3*M^2)*(1 - e)^3 - x[2]^2
end


function j!(J, x)
    """ 
    Calculate the Jacobian matrix for angular momentum L (=x[1]) and energy E (=x[2]) for Schwarzschild metric
    """
    J[1, 1] = 2*x[1] / (p^2 * M^2) * (1 + e)^2 - 4*x[1]/(p^3*M^2)*(1 + e)^3
    J[1, 2] = -2*x[2]
    J[2, 1] = 2*x[1] / (p^2 * M^2) * (1 - e)^2 - 4*x[1]/(p^3*M^2)*(1 - e)^3
    J[2, 2] = -2*x[2]
end


function E(p::Float64, e::Float64, M::Float64)::Float64
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



function NNOrbitModel_Schwarzschild_EMR_numerically(u, model_params, t; NN=nothing, NN_params=nothing)
    """
    Defines system of odes which describes motion of
    point like particle with Newtonian physics, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants
    """
    χ, ϕ = u
    p, M, e  = model_params

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(u, NN_params)
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇]

end


function RelativisticOrbitModel_Schwarzschild_EMR_numerically(u, model_params, t)
    """
    Defines system of odes which describes motion of
    point like particle in schwarzschild background, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants
    # NOTE: not used
    """
    χ, ϕ = u
    p, M, e  = model_params

    L₀ = p*M/sqrt(p-3-e^2)
    E₀ = sqrt(((p-2-2*e)*(p-2+2*e))/(p*(p-3-e^2)))

    numerical_solution = nlsolve(f!, j!, [ L₀; E₀])
    E = numerical_solution.zero[2]
    L = numerical_solution.zero[1]

    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2

    dϕdτ = L/r^2
    dtdτ = p/(p-2-2*e*cos(χ))*E
    drdτ = e * sin(χ) * sqrt( (p-6-2*e*cos(χ))/(p*(p-3-e^2)))

    ϕ̇ = dϕdτ / dtdτ #L/r^2 # eq. 11a
    χ̇ = drdτ / (dtdτ * drdχ) # eq. 11b

    return [χ̇, ϕ̇,]

end

