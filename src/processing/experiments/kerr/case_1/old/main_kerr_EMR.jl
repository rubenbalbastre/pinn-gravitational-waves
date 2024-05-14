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
test_name = "test_1_cos/"
output_directory = "../../01_data/02_output/01_case_kerr/case_1_system/"
output_dir = output_directory*test_name
solutions_dir = output_dir*"solutions/"
metrics_dir = output_directory*"metrics/"

# Define the experimental parameters
global datasize = 250
global mass_ratio = 0.0
global dt = 100.0

## Define neural network model
NN, NN_params, chain, re = nn_model_case1(test_name)

@info "Generating exact solution..."
# TRAIN
χ₀ = pi; ϕ₀ = 0.0; p=100; M=1.0; e=0.5; a=0.5
u0_train = Float64[χ₀, ϕ₀, p, M, e, a]
global tspan_train = (0.0f0, 6.0f4)
global tsteps_train = range(tspan_train[1], tspan_train[2], length = datasize)
dt_data_train = tsteps_train[2] - tsteps_train[1]
global model_params_train = [p, M, e, a]

function ODE_model_train(u, NN_params, t)
    du = NN_EMR_Kerr(u, model_params_train, NN=NN, NN_params=NN_params)
    return du
end

prob_train = ODEProblem(EMR_Kerr, u0_train, tspan_train, model_params_train)
exact_sol_train = Array(solve(prob_train, RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
exact_waveform_train = compute_waveform(dt_data_train, exact_sol_train, mass_ratio, model_params_train)[1]

# TEST
χ₀ = pi; ϕ₀ = 0.0; p=100; M=1.0; e=0.5; a=0.5
u0_test = Float64[χ₀, ϕ₀, p, M, e, a]
factor = 5
global tspan_test = (tspan_train[2], factor*tspan_train[2])
global tsteps_test = range(tspan_test[1], tspan_test[2], length = datasize*factor)
dt_data_test = tsteps_test[2] - tsteps_test[1]
global model_params_test = [p, M, e, a]

function ODE_model_test(u, NN_params, t)
    du = NN_EMR_Kerr(u, model_params_test, NN=NN, NN_params=NN_params)
    return du
end

prob_test = ODEProblem(RelativisticOrbitModel, u0_test, tspan_test, model_params_test)
exact_sol_test = Array(solve(prob_test, RK4(), saveat = tsteps_test, dt = dt, adaptive=false))
exact_waveform_test = compute_waveform(dt_data_test, exact_sol_test, mass_ratio, model_params_test)[1]

prob_nn_train = ODEProblem(ODE_model_train, u0_train, tspan_train, NN_params)
prob_nn_test = ODEProblem(ODE_model_test, u0_test, tspan_test, NN_params)
NN(u0_train, NN_params) # test it works

# put data into arrays
exact_waveform_train_array = [exact_waveform_train, exact_waveform_test]
tsteps_train_array = [tsteps_train, tsteps_test]
model_params_train_array = [model_params_train, model_params_test]
u0_train_array = [u0_train, u0_test]
dt_data_train_array = [dt_data_train, dt_data_test]


# loss function
coef_data = 1.0
coef_weights = 1.0

function loss(
    NN_params;
    exact_waveform_train_array,
    exact_waveform_test,
    prob_nn_train,
    prob_nn_test,
    u0_train_array,
    u0_test,
    tsteps_train_array,
    tspan_test,
    tsteps_test,
    dt_data_train_array,
    dt_data_test,
    model_params_train_array,
    model_params_test
)
    # train loss 
    local train_loss = 0
    local number_of_waveforms = length(exact_waveform_train)

    for wave_index in [1]# range(2, number_of_waveforms)

        global exact_waveform_train = exact_waveform_train_array[wave_index]
        global tsteps_train = tsteps_train_array[wave_index]
        global tspan_train = (tsteps_train[1], tsteps_train[end])
        global model_params_train = model_params_train_array[wave_index]
        global u0_train = u0_train_array[wave_index]
        global dt_data_train = dt_data_train_array[wave_index]

        global pred_sol_train = Array(solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
        global train_res_i = loss_function_case1(pred_sol_train, exact_waveform_train, dt_data_train, model_params_train, NN_params, coef_data=coef_data, coef_weights=coef_weights)

        train_loss += abs(train_res_i[1])
    end

    # Test loss
    pred_sol_test = Array(solve(remake(prob_nn_test, u0=u0_test, p = NN_params, tspan=tspan_test), RK4(), saveat = tsteps_test, dt = dt, adaptive=false))
    test_res = loss_function_case1(pred_sol_test, exact_waveform_test, dt_data_test, model_params_test, NN_params, coef_data=coef_data, coef_weights=coef_weights)
    train_loss = train_loss / number_of_waveforms

    global custom_act_function_coef = NN_params[1:4]

    return [train_loss, train_res_i, test_res] # we must give the loss value as first argument
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
            linewidth = 2, alpha = 0.5, label="wform data (Re)", legend_position=:topleft, title= "Train progress", titlefontsize = 8, legend_font_pointsize = 6)
    plot!(plt1, tsteps_train[1:N], pred_waveform_real_train[1:N], markershape=:none, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5, label="wform NN (Re)")
    # test waveform
    plt12 = plot(tsteps_test, exact_waveform_test, markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform data (Re)", legend=:topleft, title= "Test predictions", titlefontsize = 8, legend_font_pointsize = 6)
    plot!(plt12, tsteps_test[1:end], pred_waveform_real_test[1:end], markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform NN (Re)")
        plot!(plt12, tsteps_test[1:N], pred_waveform_real_test[1:N], markershape=:none, markeralpha = 0.25,
        linewidth = 2, alpha = 0.5, label="wform NN (Re)")
    # losses plot
    plt3 = plot(train_losses, label="train", title="Loss functions", yaxis=:log)
    plot!(plt3, test_losses, label="test")

    # save plots
    l = @layout [[a; b] a]
    global plt = plot(plt1, plt12, plt3, layout=l)
    if save_plots_gif
        push!(plot_list, plt)
    end

    if show_plots
        display(plt)
    end

    return false
end

# Train
println("Training...")
# NN_params = NN_params + Float64(1e-3)*randn(eltype(NN_params), size(NN_params))
loss_f(p) = loss(p, 
    exact_waveform_train_array=exact_waveform_train_array,
    exact_waveform_test=exact_waveform_test,
    prob_nn_train=prob_nn_train,
    prob_nn_test=prob_nn_test,
    u0_train_array=u0_train_array,
    u0_test=u0_test,
    tsteps_train_array=tsteps_train_array,
    tspan_test=tspan_test,
    tsteps_test=tsteps_test,
    dt_data_train_array=dt_data_train_array,
    dt_data_test=dt_data_test,
    model_params_train_array=model_params_train_array,
    model_params_test=model_params_test
)
# optimisers 
# Flux.Optimise.SGD(0.001)
# Flux.Optimise.ADAM(5e-5, (0.9, 0.999)), 
# Flux.Optimise.RADAM(1e-4, (0.9, 0.999)),
# BFGS(initial_stepnorm=0.01, linesearch = LineSearches.BackTracking()),
res = DiffEqFlux.sciml_train(
    loss_f, NN_params,
    BFGS(initial_stepnorm=0.01, linesearch = LineSearches.BackTracking()), 
    cb=callback, maxiters = 200)

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
    x = DataFrame(CSV.File(metrics_dir*"losses.csv", types=Dict("test_name" => String31)))
    append!(x, losses_df)
    CSV.write(metrics_dir*"losses.csv", x)
end

