

function NNOrbitModel_Schwarzschild_modified_EMR(u, model_params, t; NN=nothing, NN_params=nothing)
    """
    Defines system of odes which describes motion of
    point like particle with Newtonian physics, uses

    u[1] = χ
    u[2] = ϕ

    where, p, M, and e are constants.

    # NOTE: Schwarzschild MODIFIED model
    """

    χ, ϕ = u
    p, M, e, a = model_params

    # neural_network_input = copy(u)
    # push!(neural_network_input, a)
    neural_network_input = [χ, ϕ, a]

    if isnothing(NN)
        nn = [1,1]
    else
        # instead of introducing 'u' as neural network input we 
        # include a new vector that includes spin parameter a
        nn = 1 .+ NN(neural_network_input, NN_params)
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇,] 
end
