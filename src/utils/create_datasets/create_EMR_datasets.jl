

function get_pinn_EMR_schwarzschild_modified(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64, a::Float64,  mass_ratio::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
    """
    Get ODE NN problem in schwarzschild metric modified (includes spin parameter a)
    """

    u0 = Float64[χ₀, ϕ₀]
    tspan = (tspan[1], factor*tspan[2])

    tsteps = range(tspan[1], tspan[2], length = datasize*factor)
    model_params = [p, M, e, a]
    dt_data = Float64(tsteps[2] - tsteps[1])

    function ODE_model(u, NN_params, t)
        du = NNOrbitModel_Schwarzschild_modified_EMR(u, model_params, t, NN=NN, NN_params=NN_params)
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



function get_pinn_EMR_schwarzschild(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64,  mass_ratio::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
    """
    Get ODE NN problem in schwarzschild metric
    """

    u0 = Float64[χ₀, ϕ₀]
    tspan = (tspan[1], factor*tspan[2])

    tsteps = range(tspan[1], tspan[2], length = datasize*factor)
    model_params = [p, M, e, 0.0]
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
        "a" => 0.0,
        "M" => M,
        "dt" => dt
    )

    return problem
end


function get_pinn_EMR_kerr(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64, a::Float64,  mass_ratio::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
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


function get_true_solution_EMR_schwarzschild(u0::Vector{Float64}, model_params::Vector{Float64}, mass_ratio::Float64, total_mass::Float64, tspan, tsteps, dt_data::Float64, dt::Float64)
    """
    Computes true solution of a schwarzschild system in Kerr metric
    """

    exact_problem = ODEProblem(RelativisticOrbitModel_Schwarzschild_EMR, u0, tspan, model_params)
    true_solution = Array(solve(exact_problem, RK4(), saveat = tsteps, dt = dt))
    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return Dict("true_solution" => true_solution, "true_waveform" => true_waveform)
end


function get_true_solution_EMR_kerr(u0::Vector{Float64}, model_params::Vector{Float64}, mass_ratio::Float64, total_mass::Float64, tspan, tsteps, dt_data::Float64, dt::Float64)
    """
    Computes true solution of a EMR system in Kerr metric
    """

    exact_problem = ODEProblem(RelativisticOrbitModel_Kerr_EMR, u0, tspan, model_params)
    true_solution = Array(solve(exact_problem, RK4(), saveat = tsteps, dt = dt, adaptive=false))
    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return Dict("true_solution" => true_solution, "true_waveform" => true_waveform)
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


function get_data_subset(dataset, batch_size)
    """
    Get random subset of data
    """

    if batch_size !== nothing
        if batch_size < length(dataset)
            # subset = StatsBase.sample(dataset, batch_size, replace = false)
            subset = rand(dataset, batch_size)
        else
            subset = dataset
        end
    else
        subset = dataset
    end

    return subset
end
