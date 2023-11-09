

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


function NN_EMR_Kerr(u::Vector{Float32}, model_params::Tuple; NN=nothing, NN_params=nothing):: Tuple
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