

function get_pinn_EMR_newton(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64, a::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
    """
    Get ODE NN problem in schwarzschild metric modified.
    Enables to introduce spin parameter a to enable notebooks work
    """

    u0 = Float64[χ₀, ϕ₀]
    tspan = (tspan[1], factor*tspan[2])

    tsteps = range(tspan[1], tspan[2], length = datasize*factor)
    model_params = [p, M, e, a]
    dt_data = Float64(tsteps[2] - tsteps[1])

    function ODE_model(u, NN_params, t)
        du = NNOrbitModel_Newton_EMR(u, model_params, t, NN=NN, NN_params=NN_params)
        return du
    end

    prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)

    problem = Dict(
        "nn_problem" => prob_nn, 
        "tsteps"=> tsteps, 
        "model_params"=> model_params, 
        "u0"=> u0, 
        "dt_data"=> dt_data, 
        "tspan" => tspan,
        "q" => 0.0,
        "p" => p,
        "e" => e,
        "a" => a,
        "M" => M,
        "dt" => dt
    )

    return problem
end


function get_pinn_EMR_schwarzschild(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64, a::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
    """
    Get ODE NN problem in schwarzschild metric modified.
    Enables to introduce spin parameter a
    """

    u0 = Float64[χ₀, ϕ₀]
    tspan = (tspan[1], factor*tspan[2])

    tsteps = range(tspan[1], tspan[2], length = datasize*factor)
    model_params = [p, M, e, a]
    dt_data = Float64(tsteps[2] - tsteps[1])

    function ODE_model(u, NN_params, t)
        du = NNOrbitModel_Schwarzschild_EMR(u, model_params, t, NN=NN, NN_params=NN_params)
        return du
    end

    prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)

    problem = Dict(
        "nn_problem" => prob_nn, 
        "tsteps"=> tsteps, 
        "model_params"=> model_params, 
        "u0"=> u0, 
        "dt_data"=> dt_data, 
        "tspan" => tspan,
        "q" => 0.0,
        "p" => p,
        "e" => e,
        "a" => a,
        "M" => M,
        "dt" => dt
    )

    return problem
end


function get_pinn_EMR_kerr(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64, a::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
    """
    Get ODE NN problem in Kerr metric
    """

    u0 = Float64[χ₀, ϕ₀]
    tspan = (tspan[1], factor*tspan[2])

    tsteps = range(tspan[1], tspan[2], length = datasize*factor)
    model_params = [p, M, e, a]
    dt_data = Float64(tsteps[2] - tsteps[1])

    function ODE_model(u, NN_params, t)
        du = NNOrbitModel_Kerr_EMR(u, model_params, t, NN=NN, NN_params=NN_params)
        return du
    end

    prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)

    problem = Dict(
        "nn_problem" => prob_nn, 
        "tsteps"=> tsteps, 
        "model_params"=> model_params, 
        "u0"=> u0, 
        "dt_data"=> dt_data, 
        "tspan" => tspan,
        "q" => 0.0,
        "M" => M,
        "p" => p,
        "e" => e,
        "a" => a,
        "dt" => dt
    )

    return problem
end


function process_datasets(datasets)
    """
    Create set of datasets. Usually train and test
    """

    processed_data = Dict()

    for set in keys(datasets)

        print("Creating "*set*" dataset \n")
        processed_data[set] = []

        for (ind, data) in enumerate(datasets[set])

            data_dictionary_to_add = merge(data, Dict("index" => ind))
            
            push!(processed_data[set], data_dictionary_to_add)
        end
    end

    return processed_data
end


function get_batch(dataset, batch_size)
    """
    Get random subset of data
    """

    if batch_size !== nothing
        if batch_size < length(dataset)
            subset = rand(dataset, batch_size)
        else
            subset = dataset
        end
    else
        subset = dataset
    end

    return subset
end
