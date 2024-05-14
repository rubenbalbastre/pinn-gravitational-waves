"""
Improvement case 1.
"""
    
cd(@__DIR__)
using Pkg; Pkg.activate("../../"); Pkg.instantiate();
using OrdinaryDiffEq;
using Optim;
using LineSearches;
using DiffEqFlux;
using DiffEqSensitivity;
using Plots;
using DataFrames;
using CSV;
using Statistics;
using Flux;
using BSON: @save, @load
gr(); # specify backend for plotting

include("../04_utils/orbital_mechanics_utils.jl");
include("../04_utils/input_preparation.jl");
include("../04_utils/models.jl");
include("../04_utils/metrics.jl");
include("../04_utils/plots.jl")
include("../04_utils/loss_functions.jl")
include("../04_utils/output.jl")
include("../04_utils/nn_models.jl")

# script conditions
show_plots = true
save_plots_gif = false
save_data = true
test_name = "nn1_bi/"
model_name = "nn1_pe/"
output_directory = "../../01_data/02_output/01_case_1/case_n_systems/"
output_dir = output_directory*test_name
solutions_dir = output_dir*"solutions/"
metrics_dir = output_directory*"metrics/"

# Define the experimental parameters
global datasize = 250
global mass_ratio = 0.0
global dt = 100.0
global tspan_train = (0.0f0, 6.0f4)
global tsteps_train = range(tspan_train[1], tspan_train[2], length = datasize)
dt_data_train = tsteps_train[2] - tsteps_train[1]

## Define neural network model
NN, NN_params, chain, re = nn_model_case1_diff_wf(model_name)

@info "Generating exact solution..."
p_array = [100, 120]
e_array = [0.9, 0.7, 0.5, 0.2]
M_array = [1.0]

# data arrays
exact_waveform_train_array = []
tsteps_train_array = []
model_params_train_array = []
u0_train_array = []

for p in p_array
    for e in e_array
        for M in M_array

            # intial conditions
            χ₀ = pi; ϕ₀ = 0.0; p=p; M=M; e=e;
            u0_train = Float64[χ₀, ϕ₀, p, M, e]
            global model_params_train = [p, M, e]

            # solve exact solution
            println(p," ", e, " ", M)
            prob_train = ODEProblem(RelativisticOrbitModel_Ruben, u0_train, tspan_train, model_params_train)
            exact_sol_train = Array(solve(prob_train, RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
            exact_waveform_train = compute_waveform(dt_data_train, exact_sol_train, mass_ratio, model_params_train)[1]

            # push data into arrays
            push!(exact_waveform_train_array, exact_waveform_train)
            push!(model_params_train_array, model_params_train)
            push!(u0_train_array, u0_train)
        end
    end
end

u0_train = u0_train_array[1]
function ODE_model_train(u, NN_params, t)
    du = AbstractNNOrbitModel_Ruben(u, model_params_train, t, NN=NN, NN_params=NN_params)
    return du
end
model_params_test = model_params_train_array[1]
u0_test = u0_train_array[1]
function ODE_model_test(u, NN_params, t)
    du = AbstractNNOrbitModel_Ruben(u, model_params_test, t, NN=NN, NN_params=NN_params)
    return du
end

prob_nn_train = ODEProblem(ODE_model_train, u0_train, tspan_train, NN_params)
prob_nn_test = ODEProblem(ODE_model_test, u0_test, tspan_train, NN_params)

number_of_waveforms = length(u0_train_array)
test_set = range(1, step=3, length=1)
all_set = range(1, number_of_waveforms)
train_set = filter(x-> x ∉ test_set, all_set)

# loss function
coef_weights=1.0
coef_data=1.0
function loss(
    NN_params;
    exact_waveform_train_array,
    prob_nn_train,
    prob_nn_test,
    u0_train_array,
    model_params_train_array,
)
     
    local train_loss = 0
    local test_loss = 0
    # rand_train_set = rand(1:length(train_set), 8)

    # train
    for wave_index in train_set

        global exact_waveform_train = exact_waveform_train_array[wave_index]
        global model_params_train = model_params_train_array[wave_index]
        global u0_train = u0_train_array[wave_index]

        global pred_sol_train = Array(solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
        global train_res_i = loss_function_case1(pred_sol_train, exact_waveform_train, dt_data_train, model_params_train, NN_params, coef_weights=coef_weights, coef_data=coef_data)

        train_loss += abs(train_res_i[1])
    end

    # test
    for wave_index in test_set

        global exact_waveform_test = exact_waveform_train_array[wave_index]
        global model_params_test = model_params_train_array[wave_index]
        global u0_test = u0_train_array[wave_index]

        global pred_sol_test = Array(solve(remake(prob_nn_test, u0=u0_test, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
        global test_res_i = loss_function_case1(pred_sol_test, exact_waveform_test, dt_data_train, model_params_test, NN_params, coef_weights=coef_weights, coef_data=coef_data)

        test_loss += abs(test_res_i[1])
    end

    train_loss = train_loss / length(train_set)
    test_loss = test_loss / length(test_set)
        

    return [train_loss, train_res_i, test_res_i] # we must give the loss value as first argument
end

const train_losses = []
const test_losses = []
const train_metrics = []
const test_metrics = []

callback(θ, train_loss, train_res_i, test_res; show_plots = show_plots, save_plots_gif=save_plots_gif) = begin

    # list to save plots -> make a gif to project presentation
    if length(train_losses) == 0
        global plot_list = []
    end

    # unpackage training results
    train_loss, train_metric, pred_waveform_real_train = train_res_i
    test_loss, test_metric, pred_waveform_real_test = test_res
    N = length(tsteps_train)

    # add losses
    push!(train_losses, train_loss)
    push!(test_losses, test_loss)
    push!(train_metrics, train_metric)
    push!(test_metrics, test_metric)

    # train waveform
    plt1 = plot(tsteps_train, exact_waveform_train, markershape=:none, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5, label="wform data (Re)", legend_position=:topleft, title= "Train progress")
    plot!(plt1, tsteps_train[1:N], pred_waveform_real_train[1:N], markershape=:none, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5, label="wform NN (Re)")
    # test waveform
    plt12 = plot(tsteps_train, exact_waveform_test, markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform data (Re)", legend=:topleft, title= "Test predictions")
    plot!(plt12, tsteps_train[1:end], pred_waveform_real_test[1:end], markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform NN (Re)")
        plot!(plt12, tsteps_train[1:N], pred_waveform_real_test[1:N], markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform NN (Re)")
    # losses plot
    plt3 = plot(train_losses, label="train", title="Loss functions", yaxis=:log)
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

# Train
println("Training...")
loss_f(p) = loss(p, 
    exact_waveform_train_array=exact_waveform_train_array,
    prob_nn_train=prob_nn_train,
    prob_nn_test=prob_nn_test,
    u0_train_array=u0_train_array,
    model_params_train_array=model_params_train_array,
)
# optimisers 
# Flux.Optimise.ADAM(5e-5, (0.9, 0.999)), 
# Flux.Optimise.RADAM(1e-4, (0.9, 0.999)),
# BFGS(initial_stepnorm=0.01, linesearch = LineSearches.BackTracking()),
# NN_params = NN_params + Float64(1e-4)*randn(eltype(NN_params), size(NN_params))
res = DiffEqFlux.sciml_train(
    loss_f, NN_params,
    # Flux.Optimise.ADAM(1e-4, (0.9, 0.999)),
    BFGS(initial_stepnorm=10, linesearch = LineSearches.BackTracking()),
    cb=callback, maxiters = 100)

# save flux chain models as bson files. To do so, we must save chain model with its parameters
if !isdir(output_dir)
    mkdir(output_dir)
end
if ! isdir(solutions_dir)
    mkdir(solutions_dir)
end
if ! isdir(metrics_dir)
    mkdir(metrics_dir)
end
Flux.loadparams!(chain, Flux.params(re(res.minimizer)))
@save solutions_dir*"model_chiphi.bson" chain

# save losses
losses_df = DataFrame(
    epochs = range(1, length(train_losses)),
    test_name=test_name,
    train_loss = train_losses,
    test_loss = test_losses,
    train_metric = train_metrics,
    test_metric = test_metrics,
    # optimiser="BFGS_0.01"
    # funcion="MAE"
    # pen="l2",
    # conf="g1_1__g2_15"
)
if ! isfile(metrics_dir*"losses.csv")
    CSV.write(metrics_dir*"losses.csv", losses_df)
else
    x = DataFrame(CSV.File(metrics_dir*"losses.csv"))
    append!(x, losses_df)
    CSV.write(metrics_dir*"losses.csv", x)
end

savefig(plt, solutions_dir*"plot.png")

# # Learned solutions
# println("Compute final solutions")
# time_spec = (tsteps_train, tspan_train, dt, dt_data_train)
# learned = compute_learned_solutions_case1(time_spec, prob_train, prob_nn, model_params, mass_ratio)
# df_learned_trajectories, df_learned_waveforms = learned

# # Extrapolated solutions
# time_spec = (datasize, tspan, dt, dt_data)
# predictions = compute_extrapolated_solutions_case1(time_spec, prob_train, prob_nn, model_params, mass_ratio)
# df_predicted_trajectories, df_predicted_waveforms = predictions

# # compute metrics
# metrics_df = compute_metrics(
#     test_name,
#     datasize,
#     df_predicted_waveforms, df_predicted_trajectories, df_learned_waveforms, df_learned_trajectories)

# plt = final_plot_case1(
#     df_learned_trajectories, df_learned_waveforms, df_predicted_trajectories, df_predicted_waveforms,
#     train_losses, test_losses, show_plots
# )

# # save results
# if save_data

#     # naming dirs
#     solutions_dir = output_dir*"solutions/"
#     img_dir = output_dir*"train_img_for_gif/"
#     metrics_dir = output_directory*"metrics/"

#     # checking if directories exist
#     if ! isdir(img_dir)
#         mkdir(img_dir)
#     end

#     # save plots
#     for (ind, img) in enumerate(plot_list)
#         savefig(img, img_dir*string(ind)*"_train_img.pdf")
#     end

#     # save final plot
#     savefig(plt, output_dir*"prediction_plot.pdf")

#     # save metrics
#     if ! isfile(metrics_dir*"metrics.csv")
#         CSV.write(metrics_dir*"metrics.csv", metrics_df)
#     else
#         x = DataFrame(CSV.File(metrics_dir*"metrics.csv"))
#         append!(x, metrics_df)
#         CSV.write(metrics_dir*"metrics.csv", x)
#     end



#     # save csv files
#     CSV.write(solutions_dir*"EMRI_learned_trajectories.csv", df_learned_trajectories)
#     CSV.write(solutions_dir*"EMRI_predicted_trajectories.csv", df_predicted_trajectories)
#     CSV.write(solutions_dir*"EMRI_learned_waveforms.csv", df_learned_waveforms)
#     CSV.write(solutions_dir*"EMRI_predicted_waveforms.csv", df_predicted_waveforms)

#     @info "data saved"
# end

# @info "Execution finished"
