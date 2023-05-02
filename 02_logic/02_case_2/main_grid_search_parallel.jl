"""
Improvement case 2.

* Try training with batches
* Try transfer learning: train individially and join in a big neural network
* Try Extrema Machine Learning (ELM)
* Try big neural network and drop out
* Try modulus activation aproximation: https://arxiv.org/pdf/2301.05993.pdf
"""

cd(@__DIR__) 
using Distributed;
addprocs(2)
@everywhere using Pkg; 
@everywhere Pkg.activate("../../"); 
@everywhere Pkg.instantiate()
@everywhere using OrdinaryDiffEq
@everywhere using Optim
@everywhere using LineSearches
@everywhere using DiffEqFlux
@everywhere using DiffEqSensitivity
@everywhere using Plots
@everywhere using DataFrames
@everywhere using CSV
@everywhere using BSON: @save, @load
@everywhere using Flux
@everywhere import Logging
@everywhere include("../04_utils/orbital_mechanics_utils.jl");
@everywhere include("../04_utils/input_preparation.jl");
@everywhere include("../04_utils/models.jl");
@everywhere include("../04_utils/metrics.jl");
@everywhere include("../04_utils/plots.jl")
@everywhere include("../04_utils/loss_functions.jl")
@everywhere include("../04_utils/output.jl")
@everywhere include("../04_utils/nn_models.jl")
@everywhere Logging.disable_logging(Logging.Warn)

# script conditions
@everywhere show_plots = false
@everywhere save_plots_gif = false
@everywhere save_data = true
@everywhere model_name = "test/"
@everywhere test_name = "grid_search/"
@everywhere output_directory = "../../01_data/02_output/02_case_2/grid_search/"
@everywhere output_dir = output_directory*test_name
# naming dirs
@everywhere solutions_dir = output_dir*"solutions/"
@everywhere img_dir = output_dir*"train_img_for_gif/"
@everywhere metrics_dir = output_directory*"metrics/"

# time range
@everywhere datasize = 1500
@everywhere dt = 10.0

#  data
@everywhere wave_id = [
    "SXS:BBH:0211",
    "SXS:BBH:0217",
]

@everywhere train_array, test, train, wave_id_dict = load_data(wave_id)
@everywhere test_x_1, test_x_2, test_y_1, test_y_2, exact_test_wf_real, tsteps_test, tspan_test, model_params_test, u0_test, dt_data_test, mass1_test, mass2_test = test
@everywhere train_x_1, train_x_2, train_y_1, train_y_2, exact_train_wf_real, tsteps_train, tspan_train, model_params_train, u0_train, dt_data_train, mass1_train, mass2_train = train
@everywhere train_x_1_array, train_x_2_array, train_y_1_array, train_y_2_array, exact_train_wf_real_array, tsteps_train_array, tspan_train_array,model_params_array, u0_array, dt_data_array, mass1_train_array, mass2_train_array = train_array

# MODEL
@everywhere nn_output = nn_model_case2(model_name)
@everywhere NN_params, NN_chiphi, NN_chiphi_params, NN_pe, NN_pe_params, chain_phichi, chain_pe, re_chiphi, re_pe = nn_output
@everywhere l1 = length(NN_chiphi_params)

@everywhere function ODE_model_train(u, NN_params, t)
    NN_params1 = NN_params[1:l1]
    NN_params2 = NN_params[l1+1:end]
    du = NR_OrbitModel_Ruben(u, model_params_train, t,
                              NN_chiphi=NN_chiphi, NN_chiphi_params=NN_params1,
                              NN_pe=NN_pe, NN_pe_params=NN_params2)
    return du
end

@everywhere function ODE_model_test(u, NN_params, t)
    NN_params1 = NN_params[1:l1]
    NN_params2 = NN_params[l1+1:end]
    du = NR_OrbitModel_Ruben(u, model_params_test, t,
                              NN_chiphi=NN_chiphi, NN_chiphi_params=NN_params1,
                              NN_pe=NN_pe, NN_pe_params=NN_params2)
    return du
end

# TRAINING

# ode problem train
@everywhere prob_nn_train = ODEProblem(ODE_model_train, u0_train, tspan_train, NN_params)
# ode problem test
@everywhere prob_nn_test = ODEProblem(ODE_model_test, u0_test, tspan_test, NN_params)

@everywhere reg_term_list = [1.0f-2, 1.0f-1, 1.0f0, 1.0f1]
@everywhere stability_term_list = [1.0f-2, 1.0f-1, 1.0f0, 1.0f1]
@everywhere pos_ecc_term_list = [1.0f-1, 1.0f0, 1.0f1, 1.5f1, 1.0f2]
@everywhere dt2_term_list = [5f1, 1.0f2, 5f2, 1.0f3]
@everywhere dt_term_list = [1.0f2, 5.0f2, 1.0f3, 5.0f3, 1.0f4]
@everywhere data_term_list = [1.0f-2, 1.0f-1, 1.0f0, 1.0f1]


for reg_term in reg_term_list
    for stability_term in stability_term_list
        for pos_ecc_term in pos_ecc_term_list
            for dt2_term in dt2_term_list
                for dt_term in dt_term_list

                    @everywhere df = DataFrame(CSV.File(metrics_dir*"losses.csv", types=Dict("test_name" => String255)))

                    @distributed (append!) for data_term in data_term_list

                        test_name_i = "regterm_" * string(reg_term) * "__stability_term_" * string(stability_term) * "__pos_ecc_term_" * string(pos_ecc_term) * "__dt2_term_" * string(dt2_term) * "__dt_term_" * string(dt_term) * "__data_term_" * string(data_term)

                        if isfile(metrics_dir*"losses.csv")
                            x = DataFrame(CSV.File(metrics_dir*"losses.csv", types=Dict("test_name" => String255)))
                            if size(filter(:test_name => ==(test_name_i), x),1) == 0

                                global NN_params = NN_params .*0
                                println("Iteration of ", test_name_i)

                                function loss(
                                    NN_params;
                                    tsteps_increment_bool,
                                    exact_waveform_train_array,
                                    exact_waveform_test,
                                    prob_nn_train,
                                    prob_nn_test,
                                    u0_train_array,
                                    u0_test,
                                    tspan_train_array,
                                    tsteps_train_array,
                                    tspan_test,
                                    tsteps_test,
                                    dt_data_train_array,
                                    dt_data_test,
                                    model_params_train_array,
                                    model_params_test
                                    )
                                    """
                                    Compute loss function as the sum of loss functions of the several waveforms.
                                    """

                                    # train loss 
                                    local train_loss = 0
                                    local train_metric = 0
                                    local train_loss_complete = 0
                                    local train_metric_complete = 0
                                    local number_of_waveforms = length(wave_id)

                                    for wave_index in range(2, number_of_waveforms)

                                        global train_x_1 = train_x_1_array[wave_index][tsteps_increment_bool]
                                        global train_y_1 = train_y_1_array[wave_index][tsteps_increment_bool]
                                        global train_x_2 = train_x_2_array[wave_index][tsteps_increment_bool]
                                        global train_y_2 = train_y_2_array[wave_index][tsteps_increment_bool]

                                        global exact_waveform_train = exact_waveform_train_array[wave_index][tsteps_increment_bool]
                                        global exact_waveform_train_complete = exact_waveform_train_array[wave_index]
                                        global tsteps_train = tsteps_train_array[wave_index][tsteps_increment_bool]
                                        global tsteps_train_complete = tsteps_train_array[wave_index]
                                        global tspan_train = (tsteps_train[1], tsteps_train[end])
                                        global tspan_train_complete = (tsteps_train_complete[1], tsteps_train_complete[end])
                                        global model_params_train = model_params_train_array[wave_index]
                                        global u0_train = u0_train_array[wave_index]
                                        global dt_data_train = dt_data_train_array[wave_index]

                                        global pred_sol_train = solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false)
                                        global pred_sol_train_complete = solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train_complete), RK4(), saveat = tsteps_train_complete, dt = dt, adaptive=false)
                                        global train_res_i = loss_function_case2(
                                            pred_sol_train_complete[:,tsteps_increment_bool], exact_waveform_train_complete[tsteps_increment_bool], dt_data_train, NN_params, model_params_train,
                                            reg_term=reg_term,
                                            stability_term=stability_term, pos_ecc_term=pos_ecc_term,
                                            dt2_term=dt2_term, dt_term=dt_term,
                                            data_term=data_term
                                        )
                                        global train_res_i_complete = loss_function_case2(
                                            pred_sol_train_complete, exact_waveform_train_complete, dt_data_train, NN_params, model_params_train,
                                            reg_term=reg_term,
                                            stability_term=stability_term, pos_ecc_term=pos_ecc_term,
                                            dt2_term=dt2_term, dt_term=dt_term,
                                            data_term=data_term
                                        )

                                        train_loss += abs(train_res_i[1])
                                        train_loss_complete += abs(train_res_i_complete[1])
                                        train_metric += abs(train_res_i[2])
                                        train_metric_complete += abs(train_res_i_complete[2])
                                    end

                                    # Test loss
                                    pred_sol_test = solve(remake(prob_nn_test, u0=u0_test, p = NN_params, tspan=tspan_test), RK4(), saveat = tsteps_test, dt = dt, adaptive=false) # sensealg=BacksolveAdjoint(checkpointing=true)
                                    test_res = loss_function_case2(pred_sol_test, exact_waveform_test, dt_data_test, NN_params, model_params_test,
                                        reg_term=reg_term,
                                        stability_term=stability_term, pos_ecc_term=pos_ecc_term,
                                        dt2_term=dt2_term, dt_term=dt_term,
                                        data_term=data_term)
                                    
                                    train_loss = train_loss / number_of_waveforms

                                    return [train_loss, train_metric, train_res_i, train_res_i_complete, test_res] # we must give the loss value as first argument
                                end


                                # callback

                                train_losses = []
                                train_losses_complete = []
                                train_metrics = []
                                train_metrics_complete = []
                                test_losses = []
                                test_metrics = []

                                callback(θ, train_loss, train_metric, train_res_i, train_res_i_complete, test_res; show_plots = show_plots, save_plots_gif=save_plots_gif) = begin

                                    # unpackage training results
                                    train_loss, train_metric, pred_waveform_real_train, pred_waveform_imag_train, exact_waveform_train, pred_sol_train = train_res_i
                                    train_loss_complete, train_metric_complete, pred_waveform_real_train_complete, _, pred_sol_train_complete = train_res_i_complete
                                    test_loss, test_metric, pred_waveform_real_test, pred_waveform_imag_test, exact_waveform_test, pred_sol_test = test_res

                                    # add losses
                                    push!(train_losses, train_loss)
                                    push!(test_losses, test_loss)
                                    push!(train_losses_complete, train_loss_complete)

                                    # add metrics
                                    push!(train_metrics, train_metric)
                                    push!(train_metrics_complete, train_metric_complete)
                                    push!(test_metrics, test_metric)

                                    return false
                                end

                                @info "Begin Progressive training..."
                                num_optimization_increments = 100
                                optimization_increments = [collect(40:10:num_optimization_increments-5)..., num_optimization_increments-1,  num_optimization_increments]
                                n = length(optimization_increments)
                                epochs_increments = [50,50,50,50,50,50,100,150]
                                @assert length(optimization_increments) == length(epochs_increments)

                                for (index, i) in enumerate(optimization_increments)

                                    println("optimization increment :: ", i, " of ", num_optimization_increments)
                                    tsteps_increment_bool = tsteps_test .<= tspan_test[1] + i*(tspan_test[2]-tspan_test[1])/num_optimization_increments
                                    max_epochs = round(epochs_increments[index])
                                    println("Training ", max_epochs, " epochs")

                                    tmp_loss(p) = loss(
                                        p,
                                        tsteps_increment_bool=tsteps_increment_bool,
                                        exact_waveform_train_array=exact_train_wf_real_array,
                                        exact_waveform_test=exact_test_wf_real,
                                        prob_nn_train=prob_nn_train,
                                        prob_nn_test=prob_nn_test,
                                        u0_train_array=u0_array,
                                        u0_test=u0_test, 
                                        tspan_train_array=tspan_train_array,
                                        tsteps_train_array=tsteps_train_array,
                                        tspan_test=tspan_test,
                                        tsteps_test=tsteps_test,
                                        dt_data_train_array=dt_data_array,
                                        dt_data_test=dt_data_test,
                                        model_params_train_array=model_params_array,
                                        model_params_test=model_params_test
                                    )

                                    if index < n-3
                                        global NN_params = NN_params + Float64(1e-6)*randn(eltype(NN_params), size(NN_params)) #1e-7
                                        local res = DiffEqFlux.sciml_train(tmp_loss, NN_params,  BFGS(initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), cb=callback, maxiters = max_epochs, allow_f_increases=true)
                                    else
                                        global NN_params = NN_params + Float64(1e-6)*randn(eltype(NN_params), size(NN_params))
                                        local res = DiffEqFlux.sciml_train(tmp_loss, NN_params, BFGS(initial_stepnorm=2.5e-2, linesearch = LineSearches.BackTracking()), cb=callback, maxiters = max_epochs, allow_f_increases=true)
                                    end
                                    global NN_params = res.minimizer
                                end
                                println("Finished training")

                                # save data
                                niter = range(1, length(train_losses))                                
                                df_losses = DataFrame(niter=niter, test_name=test_name_i, train_losses = train_losses_complete, test_losses=test_losses, train_metrics=train_metrics_complete, test_metrics=test_metrics)
                                df_losses
                            else 
                                println(test_name_i * " already run")
                                nothing
                            end

                            CSV.write(metrics_dir*"losses.csv", df)
                        end
                    end
                end
            end
        end
    end
end

