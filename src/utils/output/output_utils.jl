
function create_outputs_directories(test_name, output_directory)

    output_dir = output_directory*test_name
    solutions_dir = output_dir*"solutions/"
    metrics_dir = output_directory*"metrics/"
    img_dir = output_dir*"train_img_for_gif/"
    
    create_directories([output_dir, solutions_dir, metrics_dir, img_dir])

    return [output_dir, solutions_dir, metrics_dir]
end


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
