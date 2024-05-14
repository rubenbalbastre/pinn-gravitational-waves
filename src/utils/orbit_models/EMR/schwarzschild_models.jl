

function NNOrbitModel_Schwarzschild_EMR(u, model_params, t; NN=nothing, NN_params=nothing)
    """
    Defines system of odes which describes motion of
    point like particle with Newtonian physics, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants
    """

    χ, ϕ = u
    p, M, e, a = model_params

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(u, NN_params)
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇,] 
end


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
