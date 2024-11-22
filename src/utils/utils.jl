

function create_directories(list_directories)
    """
    Check if a set of directories are created. If they are not, they are created.
    """

    for directory in list_directories
        create_directory_if_does_not_exist(directory)
    end

end


function create_directory_if_does_not_exist(directory)

    if ! isdir(directory)
        mkdir(directory)
    end
end


function import_project_utils(utils_path)
    """
    Import all functions from all scripts. This is not the best option
    """

    utils_to_import = (

        "loss_functions/loss_functions.jl", 

        "neural_networks/nn_architectures.jl",
        "neural_networks/nn_architectures_EMR.jl",
        "neural_networks/nn_architectures_modified.jl",

        "orbit_models/case_2/schwarzschild_models.jl",
        "orbit_models/EMR/kerr_models.jl",
        "orbit_models/EMR/schwarzschild_models.jl",
        "orbit_models/EMR/schwarzschild_modified.jl",

        "orbital_mechanics/orbital_mechanics_utils.jl", 

        "create_datasets/create_EMR_datasets.jl", 
        "create_datasets/create_nonEMR_datasets.jl",

        "processing_results/metrics.jl",

        "plots/plot_conditions.jl",
        "plots/plots.jl", 

        "output/output.jl",
        "output/output_utils.jl"
    )

    for util in utils_to_import
        include(utils_path * util);
    end

end
