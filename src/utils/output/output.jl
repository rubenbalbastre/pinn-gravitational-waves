
function write_dataframes_and_plot(
    plt,
    df_solution::DataFrame,
    df_trajectories::DataFrame,
    df_waveforms::DataFrame,
    df_losses::DataFrame,
    df_parameters::DataFrame,
    IDs; # ::Vector{String15}
    save_data::Bool = true,
    output_directory::String = "./output/")::Nothing
    #=
    Writes dataframes in given directory inside a folder named by the waveform ID

    Arguments:
        * save_data: whether to write or not
        * output_directory: directory to place outputs
        * plt: plot figure
        * df_solution: 
        * df_trajectories:
        * df_waveforms:
        * df_losses
        * df_parameters
        * IDS: IDs vector 
    
    Returns:
        * nothing
    =#
    if ! isdir(output_directory)
        mkdir(output_directory)
    end

    if save_data
        @info "Saving results...\n"
        CSV.write(output_directory*IDs[1]*"/solution.csv", df_solution)
        CSV.write(output_directory*IDs[1]*"/trajectories.csv", df_trajectories)
        CSV.write(output_directory*IDs[1]*"/waveforms.csv", df_waveforms)
        CSV.write(output_directory*IDs[1]*"/losses.csv", df_losses)
        CSV.write(output_directory*IDs[1]*"/parameters.csv", df_parameters)
        savefig(plt, output_directory*IDs[1]*"/plot.png")
    end

    return Nothing

end

function save_in_df(tsteps, waveform_real, waveform_imag, prediction_data)
    #=
    Save all data in dataframes
    =#
    pred_waveform_real, pred_waveform_imag, x, y, x2, y2, optimized_solution, orbit_nn1, orbit_nn2 = prediction_data
    df_solution = DataFrame(time = tsteps[1:length(optimized_solution)],
                            χ = optimized_solution[1,:],
                            ϕ = optimized_solution[2,:],
                            p = optimized_solution[3,:],
                            e = optimized_solution[4,:])

    df_trajectories = DataFrame(time = tsteps,
                            true_orbit_x1 = x,
                            true_orbit_y1 = y,
                            true_orbit_x2 = x2,
                            true_orbit_y2 = y2,
                            pred_orbit_x1 = orbit_nn1[1,:],
                            pred_orbit_y1 = orbit_nn1[2,:],
                            pred_orbit_x2 = orbit_nn2[1,:],
                            pred_orbit_y2 = orbit_nn2[2,:])

    df_waveforms = DataFrame(time = tsteps,
                            true_waveform_real = waveform_real,
                            true_waveform_imag = waveform_imag,
                            pred_waveform_real = pred_waveform_real,
                            pred_waveform_imag = pred_waveform_imag,
                            error_real = waveform_real .- pred_waveform_real,
                            error_imag = waveform_imag .- pred_waveform_imag)

    df_losses = DataFrame(losses = losses)
    df_parameters = DataFrame(parameters = θ)

    return df_solution, df_trajectories, df_waveforms, df_losses, df_parameters
end
