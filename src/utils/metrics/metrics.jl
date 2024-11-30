using Statistics;


function compute_metrics(
    test_name::String,
    datasize::Int64,
    df_predicted_waveforms, df_predicted_trajectories, df_learned_waveforms, df_learned_trajectories
    )::DataFrame
    """
    Compute some metrics for both train and test
    """

    # train
    RMSE_train = sqrt.(Statistics.mean((df_learned_waveforms[!, "true_waveform"] .- df_learned_waveforms[!, "pred_waveform"]).^2))
    Δx = df_learned_trajectories[!, "true_orbit_x"] .- df_learned_trajectories[!, "pred_orbit_x"]
    Δy = df_learned_trajectories[!, "true_orbit_y"] .- df_learned_trajectories[!, "pred_orbit_y"]
    RMSE_orbits_train = Statistics.mean(sqrt.(Δx.^2 + Δy.^2))
    
    # test
    RMSE_test = sqrt.(Statistics.mean((df_predicted_waveforms[datasize:end, "true_waveform"] .- df_predicted_waveforms[datasize:end, "pred_waveform"]).^2))
    Δx = df_predicted_trajectories[datasize:end, "true_orbit_x"] .- df_predicted_trajectories[datasize:end, "pred_orbit_x"]
    Δy = df_predicted_trajectories[datasize:end, "true_orbit_y"] .- df_predicted_trajectories[datasize:end, "pred_orbit_y"]
    RMSE_orbits_test = Statistics.mean(sqrt.(Δx.^2 + Δy.^2))
    
    metrics_df = DataFrame(
        test=test_name,
        RMSE_train = RMSE_train, RMSE_orbits_train=RMSE_orbits_train,
        RMSE_test=RMSE_test, RMSE_orbits_test = RMSE_orbits_test
    )

    return metrics_df
end


function compute_metrics_case2(
    test_name::String,
    df_trajectories_train,
    df_waveforms_train,
    df_trajectories_test,
    df_waveforms_test
    )::DataFrame
    """
    Compute metrics for case 2,3 experiments
    """

    # train
    RMSE_train = sqrt.(Statistics.mean((df_waveforms_train[!, "true_waveform_real"] .- df_waveforms_train[!, "pred_waveform_real"]).^2))
    Δx = df_trajectories_train[!, "true_orbit_x1"] .- df_trajectories_train[!, "pred_orbit_x2"]
    Δy = df_trajectories_train[!, "true_orbit_y1"] .- df_trajectories_train[!, "pred_orbit_y2"]
    RMSE_orbits_train = Statistics.mean(sqrt.(Δx.^2 + Δy.^2))
    
    # test
    RMSE_test = sqrt.(Statistics.mean((df_waveforms_test[!, "true_waveform_real"] .- df_waveforms_test[!, "pred_waveform_real"]).^2))
    Δx = df_trajectories_test[!, "true_orbit_x1"] .- df_trajectories_test[!, "pred_orbit_x2"]
    Δy = df_trajectories_test[!, "true_orbit_y1"] .- df_trajectories_test[!, "pred_orbit_y2"]
    RMSE_orbits_test = Statistics.mean(sqrt.(Δx.^2 + Δy.^2))

    metrics_df = DataFrame(
        test=test_name,
        RMSE_train = RMSE_train, RMSE_orbits_train=RMSE_orbits_train,
        RMSE_test=RMSE_test, RMSE_orbits_test = RMSE_orbits_test
    )

    return metrics_df
end