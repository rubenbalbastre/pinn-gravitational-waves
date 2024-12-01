"""
This script contains the loss functions of the different experiments.
"""


function loss_function_case1_single_waveform(
    pred_sol::Matrix{Float64},
    true_waveform::Vector{Float64},
    dt_data, 
    model_params::Vector{Float64},
    NN_params::Vector{Float64} = nothing;
    tsteps=nothing,
    loss_function::String = "mae",
    coef_data::Float64 = 1.0,
    coef_weights::Float64 = 0.01,
    subset::Int64 = 250
    )
    """
    Calculate loss function for a single EMR system
    """

    mass_ratio = 0
    _, M, _, _ = model_params
    pred_waveform, _ = compute_waveform(dt_data, pred_sol, mass_ratio, M, model_params)

    if loss_function == "mae"
        loss = coef_data*Flux.Losses.mae(pred_waveform[1:subset], true_waveform[1:subset]) + coef_weights*sum(abs2, NN_params)
    elseif loss_function == "mse"
        loss = Flux.Losses.mse(pred_waveform, true_waveform)
    elseif loss_function == "huber"
        loss = Flux.Losses.huber_loss(pred_waveform, true_waveform, δ=0.01)
    elseif loss_function == "original"
        loss = sum(abs2, true_waveform .- pred_waveform)
    end

    metric = Flux.Losses.mae(pred_waveform[1:subset], true_waveform[1:subset])

    loss_information = Dict{String, Any}()
    loss_information["loss"] = loss
    loss_information["metric"] = metric
    loss_information["pred_waveform"] = pred_waveform
    loss_information["true_waveform"] = true_waveform
    loss_information["tsteps"] = tsteps
    loss_information["model_params"] = model_params

    return loss_information
end


function loss_function_case1(NN_params::Vector{Float64}; processed_data, batch_size::Int64 = nothing, loss_function_name::String = "mae", subset::Int64 = 250)
    """
    Loss function for a set of EMR systems
    """

    train_loss = 0
    train_metric = 0
    test_loss = 0
    test_metric = 0

    local train_loss_information, test_loss_information

    train_subset = get_batch(processed_data["train"], batch_size)
    test_subset = get_batch(processed_data["test"], batch_size)

    for train_item in train_subset

        prob_nn_train = train_item["nn_problem"]
        exact_waveform_train = train_item["true_waveform"]
        tsteps_train = train_item["tsteps"]
        tspan_train = train_item["tspan"]
        model_params_train = train_item["model_params"]
        u0_train = train_item["u0"]
        dt_data_train = train_item["dt_data"]

        pred_sol_train = Array(solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
        train_loss_information = loss_function_case1_single_waveform(pred_sol_train, exact_waveform_train, dt_data_train, model_params_train, NN_params, tsteps=tsteps_train, loss_function=loss_function_name, subset=subset)

        train_loss += abs(train_loss_information["loss"])
        train_metric += abs(train_loss_information["metric"])
    end

    for test_item in reverse(test_subset)
        prob_nn_test = test_item["nn_problem"]
        exact_waveform_test = test_item["true_waveform"]
        tsteps_test = test_item["tsteps"]
        tspan_test = test_item["tspan"]
        model_params_test = test_item["model_params"]
        u0_test = test_item["u0"]
        dt_data_test = test_item["dt_data"]

        pred_sol_test = Array(solve(remake(prob_nn_test, u0=u0_test, p = NN_params, tspan=tspan_test), RK4(), saveat = tsteps_test, dt = dt, adaptive=false))
        test_loss_information = loss_function_case1_single_waveform(pred_sol_test, exact_waveform_test, dt_data_test, model_params_test, NN_params, tsteps=tsteps_test, loss_function=loss_function_name, subset=subset)

        test_loss += abs(test_loss_information["loss"])
        test_metric += abs(test_loss_information["metric"])
    end

    train_loss = train_loss / length(train_subset)
    train_metric = train_metric / length(train_subset)
    test_metric = test_metric / length(test_subset)
    test_loss = test_loss / length(test_subset)
    
    agregated_metrics = Dict("train_loss" => train_loss, "test_loss" => test_loss, "train_metric" => train_metric, "test_metric" => test_metric)

    # we must give the loss value as first argument
    return [train_loss, agregated_metrics, train_loss_information, test_loss_information]
end


function loss_function_case2_single_waveform(
    pred_sol, waveform_real, dt_data, NN_params, model_params;
    reg_term=1.0f-1, stability_term=1.0f0, pos_ecc_term=1.0f1,
    dt2_term=1.0f2, dt_term=1.0f3, data_term=1.0,
    orbits_penalization=false,
    mass1_train=nothing, mass2_train=nothing,
    train_x_1=nothing, train_x_2=nothing
    )::Dict{String, Any}
    """
    Loss function of general systems
    """

    mass_ratio = model_params["q"]
    M = model_params["M"]
    pred_waveform_real, _ = compute_waveform(dt_data, pred_sol, mass_ratio, M, model_params)
    p = pred_sol[3,:]
    e = pred_sol[4,:]
    N = length(pred_waveform_real)

    # some tests to check if this improved performance
    if orbits_penalization & length(pred_waveform_real) > 1500*0.95
        pred_orbit = soln2orbit(pred_sol, model_params)
        orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1_train, mass2_train)
        orbit_loss = rmse_term*sum(abs2, sqrt.( (train_x_1[1:N].- orbit_nn1[1,1:end]).^2 + (train_x_2[1:N]  .- orbit_nn1[2,1:end]).^2 )  )
    else
        orbit_loss = 0
    end

    loss =

        + 1/N * (
        
            # rmse_term*sum(abs2, waveform_real[1:N] .- pred_waveform_real)
            # rmse_term*sum(abs, waveform_real[1:N] .- pred_waveform_real)
            # rmse_term*Flux.Losses.mse(pred_waveform_real, waveform_real[1:N])
            # rmse_term*Flux.Losses.huber_loss(pred_waveform_real, waveform_real[1:N])

            data_term*Flux.Losses.mae(pred_waveform_real, waveform_real[1:N])
            + dt_term*sum(abs2, max.(d_dt(p,dt_data),0.0))
            + dt2_term*sum(abs2, max.(d2_dt2(p,dt_data),0.0))
            + pos_ecc_term*sum(abs2, max.(-e,0.0))
            + pos_ecc_term*sum(abs2, max.(-p, 0.0))
            + stability_term*sum(abs2, max.(e[p .>= 6 + 2*e[1]] .- e[1],0.0))
            + orbit_loss
        )
            + reg_term*sum(abs2, NN_params)
            
    metric = sqrt(Statistics.mean( (pred_waveform_real .- waveform_real[1:N]).^2 ))

    results = Dict{String, Any}(
        "metric" => metric,
        "loss" => loss,
        "pred_waveform" => pred_waveform_real,
        "pred_solution" => pred_sol
    )

    return results
end


function merge_info(a, b)

    for (key, value) in b
        a[key] = value
    end
    return a
end


function loss_function_case2(NN_params; tsteps_increment_bool, dataset_train, dataset_test)
    """
    Compute loss function as the sum of loss functions of the several waveforms.
    # TODO: decide how to aggregate loss function from n datasets
    """

    local train_loss = 0
    local train_metric = 0
    local train_loss_complete = 0
    local train_metric_complete = 0
    local test_loss = 0
    local test_metric = 0

    for (wave_id, wave) in dataset_train

        pred_sol_train_complete = solve(
            remake(wave["nn_problem"], u0=wave["u0"], p = NN_params, tspan=wave["tspan"]), 
            RK4(), 
            saveat = wave["tsteps"], 
            dt = wave["dt_data"], 
            adaptive=false
        )

        train_results_i = loss_function_case2_single_waveform(
            pred_sol_train_complete[:, tsteps_increment_bool], 
            wave["true_waveform"][tsteps_increment_bool], 
            wave["dt_data"], 
            NN_params, 
            wave["model_params"]
        )
        train_results_i_complete = loss_function_case2_single_waveform(
            pred_sol_train_complete, 
            wave["true_waveform"], 
            wave["dt_data"], 
            NN_params, 
            wave["model_params"]
        )
        
        global train_results_i = merge_info(train_results_i, wave)
        global train_results_i_complete = merge_info(train_results_i_complete, wave)

        train_loss += abs(train_results_i["loss"])
        train_loss_complete += abs(train_results_i_complete["loss"])
        train_metric += abs(train_results_i["metric"])
        train_metric_complete += abs(train_results_i_complete["metric"])
    end

    for (wave_id, wave) in dataset_test

        pred_sol_test_complete = solve(
            remake(wave["nn_problem"], u0=wave["u0"], p = NN_params, tspan=wave["tspan"]), 
            RK4(), 
            saveat = wave["tsteps"], 
            dt = wave["dt_data"], 
            adaptive=false
        )

        test_results_i_complete = loss_function_case2_single_waveform(
            pred_sol_test_complete, 
            wave["true_waveform"], 
            wave["dt_data"], 
            NN_params, 
            wave["model_params"]
        )

        global test_results_i_complete = merge_info(test_results_i_complete, wave)

        test_loss += abs(test_results_i_complete["loss"])
        test_metric += abs(test_results_i_complete["metric"])

    end

    metrics = Dict(
        "train_loss" => train_loss / length(dataset_train),
        "test_loss" => test_loss / length(dataset_test),
        "test_metric" => test_metric  / length(dataset_test),
        "train_loss_complete" => train_loss_complete,
        "train_metric" => train_metric  / length(dataset_train),
        "train_metric_complete" => train_metric_complete
    )

    # we must give the loss value as first argument to work with the external library 
    return [train_loss, metrics, train_results_i, train_results_i_complete, test_results_i_complete]
end