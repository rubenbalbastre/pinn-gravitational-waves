

function NNOrbitModel_Schwarzchild(u, model_params, t;
    NN_chiphi=nothing, NN_chiphi_params=nothing,
    NN_pe=nothing, NN_pe_params=nothing)::Vector
    """
    Defines system of ODEs which describes motion of
    point like particle with Newtonian physics.
    Equation of motion => u = [χ, ϕ, p, e]

    Arguments:
        - u: solution at instant t
        - model_params:
        - N_chiphi: NN
        - N_pe: NN
        - N_chiphi: vector of parameters of NN_chiphi_params
        . N_pe_params: vector of parameters of NN_pe params

    Returns:
        - new_u: solution at instant t+1
    """

    # get actual solution at instant t
    χ = u[1]
    ϕ = u[2]
    p = u[3]
    e = u[4]

    M = model_params["M"]

    if p <= 0
        # println("p = ", p)
    end

    if isnothing(NN_chiphi)
        nn_chiphi = [1,1]
    else
        nn_chiphi = 1 .+ NN_chiphi(u, NN_chiphi_params)
    end

    if isnothing(NN_pe)
        nn_pe = [0,0]
    else
        nn_pe = NN_pe(u, NN_pe_params)
    end

    numer = (1+e*cos(χ))^2
    denom = M*(abs(p)^(3/2))

    # get solution in t+1
    χ̇ = (numer / denom) * nn_chiphi[1] # eq. 5a
    ϕ̇ = (numer / denom) * nn_chiphi[2] # eq. 5b
    ṗ = nn_pe[1] # eq. 5c
    ė = nn_pe[2] # eq. 5d

    new_u = [χ̇, ϕ̇, ṗ, ė]

    return new_u

end
