"""
This script contains the loss functions of the different experiments.
"""

function loss_function_case1(
    pred_sol,
    exact_waveform,
    dt_data, 
    model_params,
    NN_params=nothing;
    coef_data = 1.0,
    coef_weights = 0.0
    )
    #=
    Calculate loss function
    =#
    mass_ratio = 0
    pred_waveform = compute_waveform(dt_data, pred_sol, mass_ratio, model_params)[1]
    # loss = sum(abs2, exact_waveform .- pred_waveform)
    # loss = Flux.Losses.huber_loss(pred_waveform, exact_waveform)
    # loss = Flux.Losses.mse(pred_waveform, exact_waveform)
    loss = coef_data*Flux.Losses.mae(pred_waveform, exact_waveform) + coef_weights*sum(abs2, NN_params)
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
    #=
    Loss function of case 2 without hyperparameter tuning. Default values.
    =#

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


function loss_function_case3(
    pred_sol, waveform_real, dt_data, NN_params, model_params;
    reg_term=1.0f-1, stability_term=1.0f0, pos_ecc_term=1.0f1,
    dt2_term=1.0f2, dt_term=1.0f3, reg_term_l1=0.0, rmse_term=1.0,
    orbits_penalization=false,
    mass1_train=nothing, mass2_train=nothing,
    train_x_1=nothing, train_x_2=nothing
    )
    #=
    Loss function of case 2 without hyperparameter tuning. Default values.
    =#

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

    loss = 1/N * (
        rmse_term*sum(abs2, waveform_real[1:N] .- pred_waveform_real)
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



# function loss(θ, mass1, q, prob_nn, dt_data; saveat=tsteps, increment::Int64 = 0)
#     #=
#     Definition of loss function to optimize
#     =#

#     # get params from vector θ
#     e0=θ[1]
#     χ0=θ[2]
#     ϕ0=0.0
#     p0 = x₀ * (1+e0*cos(χ0)) / mass1
#     NN_params = θ[3:end]
#     tspan = (saveat[1], saveat[end])

#     # remake ODE problem
#     u0 = [χ0, ϕ0, p0, e0]
#     prob_remake = remake(prob_nn, u0=u0,  p=NN_params, tspan=tspan)
#     # solve ODE problem
#     pred_soln = solve(
#         prob_remake,
#         RK4(),
#         saveat = saveat, dt = dt, adaptive=false,
#         sensealg = BacksolveAdjoint(checkpointing=true)
#     )

#     # compute loss function
#     out = compute_loss_function_case1(waveform_real, dt_data, pred_soln, NN_params, q)
#     loss, pred_waveform_real, pred_waveform_imag = out

#     return loss, pred_waveform_real, pred_waveform_imag, pred_soln
# end
