

function RelativisticOrbitModel_Schwarzschild_EMR(u, model_params, t)
    """
    Defines system of odes which describes motion of
    point like particle in schwarzschild background, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants
    """

    χ, ϕ = u
    p, M, e, a = model_params

    numer = (p-2-2*e*cos(χ)) * (1+e*cos(χ))^2
    denom = sqrt( (p-2)^2-4*e^2 )

    ϕ̇ = numer / (M*(p^(3/2))*denom) # eq. 11a
    χ̇ = numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom ) # eq. 11b

    return [χ̇, ϕ̇,] 
end


function NNOrbitModel_Schwarzschild_EMR(u, model_params, t; NN=nothing, NN_params=nothing)
    """
    Defines system of odes which describes motion of
    point like particle with Schwarzschild background, uses

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

    numer = (p-2-2*e*cos(χ)) * (1+e*cos(χ))^2
    denom = sqrt( (p-2)^2-4*e^2 )

    ϕ̇ = (numer / (M*(p^(3/2))*denom)) * nn[1] # eq. 11a
    χ̇ = (numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom )) * nn[2] # eq. 11b

    return [χ̇, ϕ̇,] 
end
