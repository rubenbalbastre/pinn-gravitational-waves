

function create_directories(list_directories)
    """
    Check if a set of directories are created. If they are not, they are created.
    """
    
    for directory in list_directories
        paths_structure = split(directory, "/")
        for (index, path) in enumerate(paths_structure)
            if path != ".."
                dir = join(paths_structure[1:index], "/")
                create_directory_if_does_not_exist(dir)
            end
        end
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

        "src/utils/create_datasets/problems/emr_problems.jl", 
        "src/utils/create_datasets/problems/non_emr_problems.jl",
        "src/utils/create_datasets/solutions/emr/emr_solutions.jl",

        "src/utils/metrics/metrics.jl",

        "src/utils/loss_functions/loss_functions.jl", 

        "src/utils/neural_networks/nn_architectures_emr.jl",
        "src/utils/neural_networks/nn_architectures_non_emr.jl",

        "src/utils/orbit_models/non_emr/schwarzschild_models.jl",
        "src/utils/orbit_models/emr/kerr_models.jl",
        "src/utils/orbit_models/emr/schwarzschild_models.jl",
        "src/utils/orbit_models/emr/newton_models.jl",

        "src/utils/orbital_mechanics/orbital_mechanics_utils.jl", 

        "src/utils/plots/plot_conditions.jl",
        "src/utils/plots/plots.jl", 

        "src/utils/output/output.jl"
    )

    for util in utils_to_import
        include(utils_path * util);
    end

end
