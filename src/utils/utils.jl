function create_directories(list_directories)
    """
    Check if a set of directories are created. If they are not, they are created.
    """

    for directory in list_directories
        if !isdir(directory)
            mkdir(directory)
        end
    end

end


function import_project_utils(utils_path)

    utils_to_import = (
        "orbital_mechanics_utils.jl", "input_preparation.jl", "models.jl", "metrics.jl", "plots.jl", "loss_functions.jl", "output.jl", "nn_models.jl",
        "create_ode_problems.jl", "plot_conditions.jl"
    )

    for util in utils_to_import
        include(utils_path * util);
    end

end


function process_datasets(datasets)

    true_waveform_array = []
    tsteps_array = []
    tspan_array = []
    model_params_array = []
    u0_array = []
    dt_data_array = []

    for data in datasets
        push!(true_waveform_array, data["waveform"])
        push!(tsteps_array, data["tsteps"])
        push!(tspan_array, data["tspan"])
        push!(model_params_array, data["model_params"])
        push!(u0_array, data["u0"])
        push!(dt_data_array, data["dt_data"])
    end

    processed_data = Dict("true_waveform" => true_waveform_array, "tsteps" => tsteps_array, "tspan" => tspan_array, "model_params" => model_params_array, "u0" => u0_array, "dt_data" => dt_data_array)
    return processed_data
end