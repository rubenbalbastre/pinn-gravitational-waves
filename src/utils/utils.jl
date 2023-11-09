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
        "orbital_mechanics/orbital_mechanics_utils.jl", "preprocessing/input_preparation.jl", "models/models.jl", "metrics.jl", "plots/plots.jl", "loss_functions/loss_functions.jl", "output/output.jl", "models/nn_models.jl",
        "preprocessing/create_ode_problems.jl", "plots/plot_conditions.jl", "preprocessing/process_datasets.jl"
    )

    for util in utils_to_import
        include(utils_path * util);
    end

end
