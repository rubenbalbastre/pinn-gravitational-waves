"""
This script contains the loss functions of the different experiments.
"""

function loss_function_EMR(NN_params; processed_data, train_info, test_info, loss_function_name = "mae")
    """
    Loss function of EMR case
    """

    prob_nn_train=train_info["nn_problem"]
    exact_waveform_train_array=processed_data["true_waveform"]
    u0_train_array=processed_data["u0"]
    tsteps_train_array=processed_data["tsteps"]
    dt_data_train_array=processed_data["dt_data"]
    model_params_train_array=processed_data["model_params"]

    prob_nn_test=test_info["nn_problem"]
    exact_waveform_test=processed_data["true_waveform"][2]
    u0_test=processed_data["u0"][2]
    dt_data_test=processed_data["dt_data"][2]
    tspan_test=processed_data["tspan"][2]
    tsteps_test=processed_data["tsteps"][2]
    model_params_test=processed_data["model_params"][2]

    # train loss 
    train_loss = 0
    number_of_waveforms = length(exact_waveform_train_array[1])

    for wave_index in [1]# range(2, number_of_waveforms)

        exact_waveform_train = exact_waveform_train_array[wave_index]
        tsteps_train = tsteps_train_array[wave_index]
        tspan_train = (tsteps_train[1], tsteps_train[end])
        model_params_train = model_params_train_array[wave_index]
        u0_train = u0_train_array[wave_index]
        dt_data_train = dt_data_train_array[wave_index]

        pred_sol_train = Array(solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
        global train_res_i = loss_function_case1(pred_sol_train, exact_waveform_train, dt_data_train, model_params_train, NN_params, loss=loss_function_name)

        # train_loss += abs(train_res_i[1])
        train_loss += train_res_i[1]
    end

    # Test loss
    pred_sol_test = Array(solve(remake(prob_nn_test, u0=u0_test, p = NN_params, tspan=tspan_test), RK4(), saveat = tsteps_test, dt = dt, adaptive=false))
    test_res = loss_function_case1(pred_sol_test, exact_waveform_test, dt_data_test, model_params_test, NN_params, loss=loss_function_name)
    train_loss = train_loss / number_of_waveforms

    # global custom_act_function_coef = NN_params[1:4]

    # we must give the loss value as first argument
    return [train_loss, train_res_i, test_res, processed_data]
end



function loss_function_case1(
    pred_sol,
    exact_waveform,
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
        loss = coef_data*Flux.Losses.mae(pred_waveform, exact_waveform) + coef_weights*sum(abs2, NN_params)
    elseif loss == "mse"
        loss = Flux.Losses.mse(pred_waveform, exact_waveform)
    elseif loss == "huber"
        loss = Flux.Losses.huber_loss(pred_waveform, exact_waveform)
    elseif loss == "original"
        loss = sum(abs2, exact_waveform .- pred_waveform)
    end

    metric = Flux.Losses.mse(pred_waveform, exact_waveform)

    return loss, metric, pred_waveform
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