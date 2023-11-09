
function process_datasets(datasets)
    """
    Create set of datasets. Usually train and test
    """

    processed_data = Dict()

    for set in keys(datasets)
        print("Creating "*set*" dataset \n")

        true_waveform_array = []
        nn_problems = []
        tsteps_array = []
        tspan_array = []
        model_params_array = []
        u0_array = []
        dt_data_array = []
        index = []

        for (ind, data) in enumerate(datasets[set])
            push!(index, ind)
            push!(true_waveform_array, data["waveform"])
            push!(tsteps_array, data["tsteps"])
            push!(tspan_array, data["tspan"])
            push!(model_params_array, data["model_params"])
            push!(u0_array, data["u0"])
            push!(dt_data_array, data["dt_data"])
            push!(nn_problems, data["nn_problem"])
        end

        processed_data[set] = Dict("index" => index, "true_waveform" => true_waveform_array, "tsteps" => tsteps_array, "tspan" => tspan_array, "model_params" => model_params_array, "u0" => u0_array, "dt_data" => dt_data_array)
    end

    
    return processed_data
end