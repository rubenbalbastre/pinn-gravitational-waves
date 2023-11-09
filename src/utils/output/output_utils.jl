
function create_outputs_directories(test_name, output_directory)

    output_dir = output_directory*test_name
    solutions_dir = output_dir*"solutions/"
    metrics_dir = output_directory*"metrics/"
    img_dir = output_dir*"train_img_for_gif/"
    
    for dir in [output_dir, solutions_dir, metrics_dir, img_dir]
        if ! isdir(dir)
            mkdir(dir)
        end
    end

    return [output_dir, solutions_dir, metrics_dir]
end
