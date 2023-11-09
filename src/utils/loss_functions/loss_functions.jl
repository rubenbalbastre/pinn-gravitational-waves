"""
This script contains the loss functions of the different experiments.
"""

function loss_function_EMR(NN_params; processed_data, loss_function_name = "mae")
    """
    Loss function of EMR case
    """

    train_loss = 0
    train_metric = 0
    test_loss = 0
    test_metric = 0

    for train_item in processed_data["train"]["index"]
        prob_nn_train = processed_data["train"]["true_waveform"][train_item]
        exact_waveform_train = processed_data["train"]["true_waveform"][train_item]
        tsteps_train = processed_data["train"]["tsteps"][train_item]
        tspan_train = (tsteps_train[1], tsteps_train[end])
        model_params_train = processed_data["train"]["model_params"][train_item]
        u0_train = processed_data["train"]["u0"][train_item]
        dt_data_train = processed_data["train"]["dt_data"][train_item]

        pred_sol_train = Array(solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
        global train_loss_information = loss_function_case1(pred_sol_train, exact_waveform_train, dt_data_train, model_params_train, NN_params, loss=loss_function_name)
        print("A")
        train_loss_information["tsteps"] = tspan_train
        print("A")

        train_loss += abs(train_loss_information["loss"])
        train_metric += abs(train_loss_information["metric"])
    end

    for test_item in processed_data["test"]

        exact_waveform_test = processed_data["test"]["true_waveform"]
        tsteps_test = processed_data["test"]["tsteps"][test_item]
        tspan_test = (tspan_test[1], tspan_test[end])
        model_params_test = processed_data["test"]["model_params"][test_item]
        u0_test = processed_data["test"]["u0"][test_item]
        dt_data_test = processed_data["test"]["dt_data"][test_item]

        pred_sol_test = Array(solve(remake(prob_nn_test, u0=u0_test, p = NN_params, tspan=tspan_test), RK4(), saveat = tsteps_test, dt = dt, adaptive=false))
        global test_loss_information = loss_function_case1(pred_sol_test, exact_waveform_test, dt_data_test, model_params_test, NN_params, loss=loss_function_name)
        test_loss_information["tsteps"] = tsteps_test

        test_loss += abs(test_loss_information["loss"])
        test_metric += abs(test_loss_information["metric"])
    end

    train_loss = train_loss / length(keys(processed_data["train"]))
    train_loss = train_loss / length(keys(processed_data["train"]))
    train_loss = train_loss / length(keys(processed_data["train"]))
    train_loss = train_loss / length(keys(processed_data["train"]))
    agregated_metrics = Dict("train_loss" => train_loss, "test_loss" => test_loss, "train_metric" => train_metric, "test_metric" => test_metric)

    # we must give the loss value as first argument
    return [train_loss, agregated_metrics, train_loss_information, test_loss_information]
end


function loss_function_case1(
    pred_sol,
    true_waveform,
    dt_data, 
    model_params,
    NN_params=nothing;
    loss = "mae",
    coef_data = 1,
    coef_weights = 0
    )
    """
    Calculate loss function for EMR systems
    """
    mass_ratio = 0
    pred_waveform = compute_waveform(dt_data, pred_sol, mass_ratio, model_params)[1]

    if loss == "mae"
        loss = coef_data*Flux.Losses.mae(pred_waveform, true_waveform) + coef_weights*sum(abs2, NN_params)
    elseif loss == "mse"
        loss = Flux.Losses.mse(pred_waveform, true_waveform)
    elseif loss == "huber"
        loss = Flux.Losses.huber_loss(pred_waveform, true_waveform)
    elseif loss == "original"
        loss = sum(abs2, true_waveform .- pred_waveform)
    end

    metric = Flux.Losses.mse(pred_waveform, true_waveform)

    loss_information = Dict("loss" => loss, "metric" => metric, "pred_waveform" => pred_waveform, "true_waveform" => true_waveform)

    return loss_information
end


function loss_function_case2(
    pred_sol, waveform_real, dt_data, NN_params, model_params;
    reg_term=1.0f-1, stability_term=1.0f0, pos_ecc_term=1.0f1,
    dt2_term=1.0f2, dt_term=1.0f3, data_term=1.0,
    orbits_penalization=false,
    mass1_train=nothing, mass2_train=nothing,
    train_x_1=nothing, train_x_2=nothing
    )
    """
    Loss function of general systems
    """

    mass_ratio = model_params[1]
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_sol, mass_ratio, model_params)
    p = pred_sol[3,:]
    e = pred_sol[4,:]
    N = length(pred_waveform_real)
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

    return [loss, metric, pred_waveform_real, pred_waveform_imag, waveform_real, pred_sol]
end