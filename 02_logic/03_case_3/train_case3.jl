"""
Improvement case 3
"""

cd(@__DIR__)
using Pkg; Pkg.activate("../../"); Pkg.instantiate()
using OrdinaryDiffEq
using Optim
using LineSearches
using DiffEqFlux
using DiffEqSensitivity
using Plots
using DataFrames
using CSV
using BSON: @save, @load
using Flux
import Logging
include("../04_utils/orbital_mechanics_utils.jl");
include("../04_utils/input_preparation.jl");
include("../04_utils/models.jl");
include("../04_utils/metrics.jl");
include("../04_utils/plots.jl")
include("../04_utils/loss_functions.jl")
include("../04_utils/messages.jl")
include("../04_utils/output.jl")
include("../04_utils/nn_models.jl")
Logging.disable_logging(Logging.Warn) # or e.g. Logging.Info

# script conditions
show_plots = true
save_plots_gif = false
save_data = true
model_name = "test/"
test_name = "test/"
output_directory = "../../01_data/02_output/03_case_3/"
output_dir = output_directory*test_name

# time range
datasize = 1500
dt = 10.0

#  data
wave_id = [
    "SXS:BBH:1356",
    "SXS:BBH:1357"
]
# filter!(x -> !(x in ["info.csv", "SXS:BBH:0007", "SXS:BBH:0593", "SXS:BBH:0055"]), readdir("../../01_data/01_input/02_case_2/"))
train_array, test, train, wave_id_dict = load_data(wave_id, source_path="03_case_3")
test_x_1, test_x_2, test_y_1, test_y_2, exact_test_wf_real, tsteps_test, tspan_test, model_params_test, u0_test, dt_data_test, mass1_test, mass2_test = test
train_x_1, train_x_2, train_y_1, train_y_2, exact_train_wf_real, tsteps_train, tspan_train, model_params_train, u0_train, dt_data_train, mass1_train, mass2_train = train
train_x_1_array, train_x_2_array, train_y_1_array, train_y_2_array, exact_train_wf_real_array, tsteps_train_array, tspan_train_array,model_params_array, u0_array, dt_data_array, mass1_train_array, mass2_train_array = train_array

# MODEL
@info "Defining model"
nn_output = nn_model_case2(model_name)
NN_params, NN_chiphi, NN_chiphi_params, NN_pe, NN_pe_params, chain_phichi, chain_pe, re_chiphi, re_pe = nn_output
l1 = length(NN_chiphi_params)

function ODE_model_train(u, NN_params, t)
    NN_params1 = NN_params[1:l1]
    NN_params2 = NN_params[l1+1:end]
    du = NR_OrbitModel_Ruben(u, model_params_train, t,
                              NN_chiphi=NN_chiphi, NN_chiphi_params=NN_params1,
                              NN_pe=NN_pe, NN_pe_params=NN_params2)
    return du
end

function ODE_model_test(u, NN_params, t)
    NN_params1 = NN_params[1:l1]
    NN_params2 = NN_params[l1+1:end]
    du = NR_OrbitModel_Ruben(u, model_params_test, t,
                              NN_chiphi=NN_chiphi, NN_chiphi_params=NN_params1,
                              NN_pe=NN_pe, NN_pe_params=NN_params2)
    return du
end

# TRAINING

# ode problem train
prob_nn_train = ODEProblem(ODE_model_train, u0_train, tspan_train, NN_params)
soln_nn = Array(solve(prob_nn_train, RK4(), u0 = u0_train, p = NN_params, saveat = tsteps_train, dt = dt, adaptive=false))
pred_waveform_real_train, pred_waveform_imag_train = compute_waveform(dt_data_train, soln_nn, model_params_train[1], model_params_train)

plt1 = plot(tsteps_train, exact_train_wf_real, markershape=:none, markeralpha = 0.25,
linewidth = 2, alpha = 0.5, label="wform data (Re)", legend_position=:topleft, title= "Train progress "*wave_id_dict[1], titlefontsize = 8, legend_font_pointsize = 6)
plot!(plt1, tsteps_train, pred_waveform_real_train, markershape=:none, markeralpha = 0.25,
linewidth = 2, alpha = 0.5, label="wform NN (Re)")

@assert length(tsteps_train) === length(exact_train_wf_real) && length(pred_waveform_real_train) == length(exact_train_wf_real)

# ode problem test
prob_nn_test = ODEProblem(ODE_model_test, u0_test, tspan_test, NN_params)
reg_term= 1.0f-1
reg_term_l1=0.0
stability_term=1.0f0
pos_ecc_term=1.0f1 # 1.0f1
dt2_term=1.0f2
dt_term=1.0f3
rmse_term=1.0f0

@info "Defining loss function"
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
    #=
    Compute loss function as the sum of loss functions of the several waveforms.
    =#

    # train loss 
    # local train_loss_step = 0
    local train_loss = 0
    local train_metric = 0
    local number_of_waveforms = length(wave_id)
    # local w_to_iter = rand(2:number_of_waveforms, number_of_waveforms-3)

    for wave_index in range(1, number_of_waveforms)

        global train_x_1 = train_x_1_array[wave_index][tsteps_increment_bool]
        global train_y_1 = train_y_1_array[wave_index][tsteps_increment_bool]
        global train_x_2 = train_x_2_array[wave_index][tsteps_increment_bool]
        global train_y_2 = train_y_2_array[wave_index][tsteps_increment_bool]

        global exact_waveform_train = exact_waveform_train_array[wave_index][tsteps_increment_bool]
        global tsteps_train = tsteps_train_array[wave_index][tsteps_increment_bool]
        global tspan_train = (tsteps_train[1], tsteps_train[end])
        global model_params_train = model_params_train_array[wave_index]
        global u0_train = u0_train_array[wave_index]
        global dt_data_train = dt_data_train_array[wave_index]

        global pred_sol_train = solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false)
        global train_res_i = loss_function_case2(pred_sol_train, exact_waveform_train, dt_data_train, NN_params, model_params_train,
            reg_term=reg_term,
            stability_term=stability_term, pos_ecc_term=pos_ecc_term,
            dt2_term=dt2_term, dt_term=dt_term, reg_term_l1=reg_term_l1,
            rmse_term=rmse_term)

        train_loss += abs(train_res_i[1]) # train_res_i[1]^2
        # train_loss_step += abs(train_res_i_step[1])
        train_metric += abs(train_res_i[2])
    end

    # Test loss
    pred_sol_test = solve(remake(prob_nn_test, u0=u0_test, p = NN_params, tspan=tspan_test), RK4(), saveat = tsteps_test, dt = dt, adaptive=false) # sensealg=BacksolveAdjoint(checkpointing=true)
    test_res = loss_function_case2(pred_sol_test, exact_waveform_test, dt_data_test, NN_params, model_params_test,
        reg_term=reg_term,
        stability_term=stability_term, pos_ecc_term=pos_ecc_term,
        dt2_term=dt2_term, dt_term=dt_term, reg_term_l1=reg_term_l1,
        rmse_term=rmse_term)
    
    train_loss = train_loss / number_of_waveforms

    return [train_loss, train_metric, train_res_i, test_res] # we must give the loss value as first argument
end


# callback

const train_losses = []
const train_metrics = []
const test_losses = []
const test_metrics = []

callback(θ, train_loss, train_metric, train_res_i, test_res; show_plots = show_plots, save_plots_gif=save_plots_gif) = begin

    # list to save plots -> make a gif to project presentation
    if length(train_losses) == 0
        global plot_list = []
    end

    # unpackage training results
    train_loss, train_metric_i, pred_waveform_real_train, pred_waveform_imag_train, exact_waveform_train, pred_sol_train = train_res_i
    test_loss, test_metric, pred_waveform_real_test, pred_waveform_imag_test, exact_waveform_test, pred_sol_test = test_res
    N = length(tsteps_train)

    # add losses
    push!(train_losses, train_loss)
    push!(test_losses, test_loss)

    # add metrics
    push!(train_metrics, train_metric)
    push!(test_metrics, test_metric)

    # train waveform
    plt1 = plot(tsteps_train, exact_waveform_train, markershape=:none, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5, label="wform data (Re)", legend_position=:topleft, title= "Train progress: "*wave_id_dict[length(wave_id)], titlefontsize = 8, legend_font_pointsize = 6)
    plot!(plt1, tsteps_train[1:N], pred_waveform_real_train[1:N], markershape=:none, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5, label="wform NN (Re)")
    # test waveform
    plt12 = plot(tsteps_test, exact_waveform_test, markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform data (Re)", legend=:topleft, title= "Test predictions: "*wave_id_dict[1], titlefontsize = 8, legend_font_pointsize = 6)
    plot!(plt12, tsteps_test[1:end], pred_waveform_real_test[1:end], markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform NN (Re)")
        plot!(plt12, tsteps_test[1:N], pred_waveform_real_test[1:N], markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform NN (Re)")
    # losses plot
    plt3 = plot(train_losses, label="train", title="Loss functions")
    xlabel!(plt3, "Epochs")
    plot!(plt3, test_losses, label="test")

    # save plots
    l = @layout [[a; b] a]
    global plt = plot(plt1, plt12, plt3, layout=l) # plt4 plt2, plt3
    if save_plots_gif
        push!(plot_list, plt)
    end

    if show_plots
        display(plot(plt))
    end

    # Tell sciml_train to not halt the optimization. If return true, then optimization stops.
    return false
end

@info "Begin Progressive training..."
num_optimization_increments = 100
optimization_increments = [collect(40:10:num_optimization_increments-15)..., 85, 90, num_optimization_increments-5, num_optimization_increments-1,  num_optimization_increments]
n = length(optimization_increments)
epochs_increments = [40,40,40,40, 40, 60, 70, 150, 150, 200]# range(30, 70, length=n) # 30,100
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


    # BFGS(initial_stepnorm=0.0001, linesearch = LineSearches.BackTracking())
    # Flux.Optimise.RADAM(0.001, (0.9, 0.999))
    # Flux.Optimise.ADAM(0.00005, (0.9, 0.999))
    # Flux.Optimise.RMSProp(1e-5)
    # BFGS(alphaguess = InitialQuadratic(α0=3.0), linesearch = LineSearches.BackTracking())
    # BFGS(initial_stepnorm=0.05, linesearch = LineSearches.BackTracking())

    if index < n-3
        global NN_params = NN_params + Float64(1e-6)*randn(eltype(NN_params), size(NN_params))
        local res = DiffEqFlux.sciml_train(tmp_loss, NN_params,  BFGS(initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), cb=callback, maxiters = max_epochs, allow_f_increases=true)
    else
        global NN_params = NN_params + Float64(1e-6)*randn(eltype(NN_params), size(NN_params))
        local res = DiffEqFlux.sciml_train(tmp_loss, NN_params, BFGS(initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), cb=callback, maxiters = max_epochs, allow_f_increases=true)
    end
    global NN_params = res.minimizer
end
println("Finished training")

println("Obtain final results...")
# train
pred_sol_train = solve(remake(prob_nn_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false)
pred_train_wf_real, pred_train_wf_imag = compute_waveform(dt_data_train, pred_sol_train, model_params_train[1], model_params_train)
pred_orbit_train = soln2orbit(pred_sol_train, model_params_test)
train_orbit_nn1, train_orbit_nn2 = one2two(pred_orbit_train, mass1_train, mass2_train)

# test
pred_sol_test= solve(remake(prob_nn_test, p = NN_params, tspan=tspan_test), RK4(), saveat = tsteps_test, dt = dt, adaptive=false)
pred_test_wf_real, pred_test_wf_imag = compute_waveform(dt_data_test, pred_sol_test, model_params_test[1], model_params_test)
pred_orbit_test = soln2orbit(pred_sol_test, model_params_test)
test_orbit_nn1, test_orbit_nn2 = one2two(pred_orbit_test, mass1_test, mass2_test)


## save data

df_solution_train = DataFrame(time = tsteps_train[1:length(pred_sol_train)],
                        χ = pred_sol_train[1,:],
                        ϕ = pred_sol_train[2,:],
                        p = pred_sol_train[3,:],
                        e = pred_sol_train[4,:])

df_trajectories_train = DataFrame(time = tsteps_train,
                         true_orbit_x1 = train_x_1,
                         true_orbit_y1 = train_y_1,
                         true_orbit_x2 = train_x_2,
                         true_orbit_y2 = train_y_2,
                         pred_orbit_x1 = train_orbit_nn1[1,:],
                         pred_orbit_y1 = train_orbit_nn1[2,:],
                         pred_orbit_x2 = train_orbit_nn2[1,:],
                         pred_orbit_y2 = train_orbit_nn2[2,:])

df_waveforms_train = DataFrame(time = tsteps_train,
                        true_waveform_real = exact_train_wf_real,
                        pred_waveform_real = pred_train_wf_real,
                        pred_waveform_imag = pred_train_wf_imag,
                        error_real = exact_train_wf_real .- pred_train_wf_real)


df_trajectories_test = DataFrame(time = tsteps_test,
                        true_orbit_x1 = test_x_1,
                        true_orbit_y1 = test_y_1,
                        true_orbit_x2 = test_x_2,
                        true_orbit_y2 = test_y_2,
                        pred_orbit_x1 = test_orbit_nn1[1,:],
                        pred_orbit_y1 = test_orbit_nn1[2,:],
                        pred_orbit_x2 = test_orbit_nn2[1,:],
                        pred_orbit_y2 = test_orbit_nn2[2,:])

df_waveforms_test = DataFrame(time = tsteps_test,
                       true_waveform_real = exact_test_wf_real,
                       pred_waveform_real = pred_test_wf_real,
                       pred_waveform_imag = pred_test_wf_imag,
                       error_real = exact_test_wf_real .- pred_test_wf_real)

# niter
niter = range(1, length(train_losses))
if test_name == "original"
    test_name = test_name * "____________________"
end
df_losses = DataFrame(niter=niter, test_name=test_name, train_losses = train_losses, test_losses=test_losses, train_metrics=train_metrics, test_metrics=test_metrics)

# compute metrics
metrics_df = compute_metrics_case2(test_name, df_trajectories_train, df_waveforms_train, df_trajectories_test, df_waveforms_test)


if save_data
    println("Saving data")
    # naming dirs
    solutions_dir = output_dir*"solutions/"
    img_dir = output_dir*"train_img_for_gif/"
    metrics_dir = output_directory*"metrics/"

    # checking if directories exist
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    if ! isdir(solutions_dir)
        mkdir(solutions_dir)
    end
    if ! isdir(img_dir)
        mkdir(img_dir)
    end
    if ! isdir(metrics_dir)
        mkdir(metrics_dir)
    end

    # save plots
    if save_plots_gif
        for (ind, img) in enumerate(plot_list)
            savefig(img, img_dir*string(ind)*"_train_img.png")
        end
    end

    # save final plot
    savefig(plt, output_dir*"prediction_plot.png")

    # save metrics
    if ! isfile(metrics_dir*"metrics.csv")
        CSV.write(metrics_dir*"metrics.csv", metrics_df)
    else
        x = DataFrame(CSV.File(metrics_dir*"metrics.csv"))
        append!(x, metrics_df)
        CSV.write(metrics_dir*"metrics.csv", x)
    end

    if ! isfile(metrics_dir*"losses.csv")
        CSV.write(metrics_dir*"losses.csv", df_losses)
    else
        x = DataFrame(CSV.File(metrics_dir*"losses.csv"))
        append!(x, df_losses)
        CSV.write(metrics_dir*"losses.csv", x)
    end

    CSV.write(solutions_dir*"SXS1_solution.csv", df_solution_train)
    CSV.write(solutions_dir*"SXS1_trajectories.csv", df_trajectories_train)
    CSV.write(solutions_dir*"SXS1_waveforms.csv", df_waveforms_train)
    CSV.write(solutions_dir*"SXS1_losses.csv", df_losses)

    # save flux chain models as bson files. To do so, we must save chain model with its parameters
    NN_chiphi_params = NN_params[1:l1]
    NN_pe_params = NN_params[l1+1:end]
    Flux.loadparams!(chain_pe, Flux.params(re_pe(NN_pe_params)))
    Flux.loadparams!(chain_phichi, Flux.params(re_chiphi(NN_chiphi_params)))
    @save solutions_dir*"model_chiphi.bson" chain_phichi
    @save solutions_dir*"model_pe.bson" chain_pe
end

# https://github.com/JuliaIO/BSON.jl 
# https://stackoverflow.com/questions/66395998/saving-and-loading-model-and-the-best-weight-after-training-in-sciml-julia
