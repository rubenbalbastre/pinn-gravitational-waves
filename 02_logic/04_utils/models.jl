#=
    ODE models for orbital mechanics
=#

function NewtonianOrbitModel(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    numer = (1+e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = numer / denom
    ϕ̇ = numer / denom

    return [χ̇, ϕ̇]

end

function RelativisticOrbitModel(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle in schwarzschild background, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    numer = (p-2-2*e*cos(χ)) * (1+e*cos(χ))^2
    denom = sqrt( (p-2)^2-4*e^2 )

    ϕ̇ = numer / (M*(p^(3/2))*denom) # eq. 11a
    χ̇ = numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom ) # eq. 11b

    return [χ̇, ϕ̇,]

end

function AbstractNNOrbitModel(u, model_params, t; NN=nothing, NN_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
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

function AbstractNROrbitModel(u, model_params, t;
                              NN_chiphi=nothing, NN_chiphi_params=nothing,
                              NN_pe=nothing, NN_pe_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ
        u[3] = p
        u[4] = e	δ

        q is the mass ratio
    =#
    χ, ϕ, p, e = u
    # p = abs(p)
    if p < 0
        p = p^2
    end
    q = model_params[1]
    M=1.0

    if p <= 0
        println("p = ", p)
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

    χ̇ = (numer / denom) * nn_chiphi[1] # eq. 5a
    ϕ̇ = (numer / denom) * nn_chiphi[2] # eq. 5b
    ṗ = nn_pe[1] # eq. 5c
    ė = nn_pe[2] # eq. 5d

    return [χ̇, ϕ̇, ṗ, ė]
end



#------------------- made by Rubén -------------#


function NR_OrbitModel_Ruben(u, model_params, t;
    NN_chiphi=nothing, NN_chiphi_params=nothing,
    NN_pe=nothing, NN_pe_params=nothing)::Vector
    #=
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
    =#

    # get actual solution at instant t
    χ = u[1]
    ϕ = u[2]
    p = u[3]
    e = u[4]

    q = model_params[1]
    M = model_params[2]

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
    new_u = [χ̇, ϕ̇, ṗ, ė, q, M]

    return new_u

end



function AbstractNNOrbitModel_Ruben(u, model_params, t; NN=nothing, NN_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
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

    return [χ̇, ϕ̇, p, M, e]

end

function RelativisticOrbitModel_Ruben(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle in schwarzschild background, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    if length(u) == 2
        χ, ϕ = u
    elseif length(u) == 5
        χ, ϕ, _, _, _ = u
    end
    p, M, e  = model_params

    numer = (p-2-2*e*cos(χ)) * (1+e*cos(χ))^2
    denom = sqrt( (p-2)^2-4*e^2 )

    ϕ̇ = numer / (M*(p^(3/2))*denom) # eq. 11a
    χ̇ = numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom ) # eq. 11b

    return [χ̇, ϕ̇, p, M, e]

end


###############################################
# Kerr 
###############################################

function EMR_Kerr(u:: Vector{Float32}, model_params:: Tuple):: Tuple
    """
    Defines system of odes which describes motion of point like particle in kerr background, where, p, M, e and a are constants
    """
    χ, ϕ, p, M, e, a = u
    p, M, e, a  = model_params

    # numer = 
    # denom = 

    ϕ̇ = numer / (M*(p^(3/2))*denom)
    χ̇ = numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom )

    return [χ̇, ϕ̇, p, M, e, a]
end


function NN_EMR_Kerr(u:: Vector{Float32}, model_params:: Tuple; NN=nothing, NN_params=nothing):: Tuple
    """
    Defines NN system of odes which describes motion of point like particle in kerr background, where, p, M, e and a are constants
    """
    χ, ϕ, p, M, e, a = u
    p, M, e  = model_params

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(u, NN_params)
    end

    # numer = 
    # denom = 

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇, p, M, e, a]

end