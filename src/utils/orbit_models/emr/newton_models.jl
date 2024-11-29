

function NewtonianOrbitModel_EMR(u, model_params, t)
    """
    Defines system of odes which describes motion of
    point like particle with Newtonian physics, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants
    """
    χ, ϕ = u
    p, M, e, a  = model_params

    numer = (1+e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = numer / denom
    ϕ̇ = numer / denom

    return [χ̇, ϕ̇]

end


function NNOrbitModel_Newton_EMR(u, model_params, t; NN=nothing, NN_params=nothing)
    """
    Defines system of odes which describes motion of
    point like particle with Newtonian physics, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants
    """

    χ, ϕ = u
    p, M, e, a = model_params

    neural_network_input = [χ, ϕ, a, p, M, e]

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(neural_network_input, NN_params)
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇,] 
end
