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
using Random;
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
include("../04_utils/output_utils.jl")
include("../04_utils/plot_config.jl")


seed = 1234;
Random.seed!(seed)

# script conditions
show_plots = true
save_plots_gif = true
save_data = false

test_name = "test_1_cos/"
model_name = "test_1_cos/"
output_directory = "../../01_data/02_output/01_case_kerr/test_sch_num/"

output_dir, solutions_dir, metrics_dir = create_outputs_directories(test_name, output_directory)

# Define the experimental parameters
const datasize = 500 # 250
const mass_ratio = 0.0
const dt = 100.0

## Define neural network model
NN, NN_params, chain, re = nn_model_case1(model_name)

@info "Generating exact solution..."
# TRAIN
χ₀ = pi + 1e-3; ϕ₀ = 0.0; p=100; M=1.0; e=0.5; a = 0.1
const u0_train = Float64[χ₀, ϕ₀]
const tspan_train = (0.0f0, 6.0f4)
const tsteps_train = range(tspan_train[1], tspan_train[2], length = datasize)
const dt_data_train = tsteps_train[2] - tsteps_train[1]
const model_params_train = [p, M, e, a]

function ODE_model_train(u, NN_params, t; NN=NN)
    du = AbstractNNOrbitModel_kerr(u, model_params_train, t, NN=NN, NN_params=NN_params)
    return du
end

# prob_train = ODEProblem(RelativisticOrbitModel_schwarzschild_numerically, u0_train, tspan_train, model_params_train)
# exact_sol_train_s = Array(solve(prob_train, RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
# exact_waveform_train = compute_waveform(dt_data_train, exact_sol_train_s, mass_ratio, model_params_train)[1]
# plot(exact_waveform_train)
# pred_orbit = soln2orbit(exact_sol_train_s, model_params_train)
# plot(pred_orbit[1,1:end], pred_orbit[2,1:end])

function get_EMR_exact_solution(model, u0, tspan, tsteps, model_params, dt, dt_data)

    mass_ratio = 0

    problem = ODEProblem(model, u0, tspan, model_params)
    exact_solution = Array(
        solve(
            problem,
            RK4(),
            saveat = tsteps,
            dt = dt,
            adaptive=false
        )
    )
    exact_waveform = compute_waveform(dt_data, exact_solution, mass_ratio, model_params)[1]
    exact_orbit = soln2orbit(exact_waveform, model_params)

    return [exact_waveform, exact_orbit]
end

exact_waveform_train, _ =  get_EMR_exact_solution(
    RelativisticOrbitModel_kerr_numerically,
    u0_train,
    tspan_train,
    tsteps_train,
    model_params_train,
    dt,
    dt_data_train
)
# prob_train = ODEProblem(RelativisticOrbitModel_kerr_numerically, u0_train, tspan_train, model_params_train)
# exact_sol_train = Array(solve(prob_train, RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
# exact_waveform_train = compute_waveform(dt_data_train, exact_sol_train, mass_ratio, model_params_train)[1]
# pred_orbit = soln2orbit(exact_sol_train, model_params_train)
# plot(exact_waveform_train)
# plot(pred_orbit[1,1:end], pred_orbit[2,1:end])

# TEST
χ₀ = pi; ϕ₀ = 0.0; p=100; M=1.0; e=0.5; a=0.1
u0_test = Float64[χ₀, ϕ₀]
factor = 5
const tspan_test = (tspan_train[2], factor*tspan_train[2])
const tsteps_test = range(tspan_test[1], tspan_test[2], length = datasize*factor)
const dt_data_test = tsteps_test[2] - tsteps_test[1]
const model_params_test = [p, M, e, a]

function ODE_model_test(u, NN_params, t)
    du = AbstractNNOrbitModel_kerr(u, model_params_test, t, NN=NN, NN_params=NN_params)
    return du
end

# prob_test = ODEProblem(RelativisticOrbitModel, u0_test, tspan_test, model_params_test)
# exact_sol_test = Array(solve(prob_test, RK4(), saveat = tsteps_test, dt = dt, adaptive=false))
# exact_waveform_test = compute_waveform(dt_data_test, exact_sol_test, mass_ratio, model_params_test)[1]

exact_waveform_test, _ =  get_EMR_exact_solution(
    RelativisticOrbitModel_kerr_numerically,
    u0_test,
    tspan_test,
    tsteps_test,
    model_params_test,
    dt,
    dt_data_test
)


prob_nn_train = ODEProblem(ODE_model_train, u0_train, tspan_train, NN_params)
prob_nn_test = ODEProblem(ODE_model_test, u0_test, tspan_test, NN_params)

# put data into arrays
exact_waveform_train_array = [exact_waveform_train, exact_waveform_test]
tsteps_train_array = [tsteps_train, tsteps_test]
model_params_train_array = [model_params_train, model_params_test]
u0_train_array = [u0_train, u0_test]
dt_data_train_array = [dt_data_train, dt_data_test]

# zero training image
NN_params = 0 .* NN_params
pred_sol = Array(solve(remake(prob_nn_train, u0=u0_train, p = NN_params, tspan=tspan_train), RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
pred_waveform_real_train = compute_waveform(dt_data_train, pred_sol, mass_ratio, model_params_train)[1]

N = length(tsteps_train)
plt1 = plot(
    tsteps_train,
    exact_waveform_train,
    label="datos (Re)",
    title= "Predicción de la forma de onda en sistema de entrenamiento", 
    titlefontsize = title_font_size,
    legendfontsize = legend_font_size,
    guidefontsize=title_font_size,
    gridalpha=grid_alpha,
    gridstyle=grid_style,
    tickfontsize=tick_font_size,
    color=:black,
    seriestype=:scatter,
    ms=5,
    markershape=:none,
    size=(1200,350),
    bottom_margin = 25Plots.mm,
    left_margin = 25Plots.mm,
    right_margin = 10Plots.mm,
    top_margin = 10Plots.mm,
    framestyle=:box,
    legend=:outertop,
    legend_column=2,
    xlabel="Tiempo",
    ylabel="Forma de onda"
    )

plot!(plt1, tsteps_train[1:N], pred_waveform_real_train[1:N], label="NN entrenamiento (Re)", linewidth=line_width)
plt = plot(plt1, size=(1600,600))
savefig(plt, img_dir*"0_train_img.pdf")
savefig(plt, img_dir*"0_train_img.png")

# loss function
const coef_data = 1.0
const coef_weights = 1.0

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

    for wave_index in [1] # range(2, number_of_waveforms)

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

    # global custom_act_function_coef = NN_params[1:4]

    return [train_loss, train_res_i, test_res] # we must give the loss value as first argument
end

const train_losses = []
const test_losses = []
const train_metrics = []
const test_metrics = []

callback(θ, train_loss, train_res_i, test_res; show_plots = show_plots, save_plots_gif=save_plots_gif) = begin

    if length(train_losses) == 0
        global plot_list = []
    end

    train_loss, train_metric, _ = train_res_i
    test_loss, test_metric, pred_waveform_real_test = test_res
    N = length(tsteps_train)

    # add losses
    push!(train_losses, train_loss)
    push!(test_losses, test_loss)
    push!(train_metrics, train_metric)
    push!(test_metrics, test_metric)

    # train waveform
    plt1 = plot(
        tsteps_train, exact_waveform_train, label="datos (Re)", title= "Predicción de la forma de onda en sistema de entrenamiento", 
        titlefontsize = title_font_size,
        legendfontsize = legend_font_size,
        guidefontsize=title_font_size,
        gridalpha=grid_alpha,
        gridstyle=grid_style,
        tickfontsize=tick_font_size,
        color=:black,
        seriestype=:scatter,
        ms=5,
        markershape=:none,
        size=(1200,350),
        bottom_margin = 25Plots.mm,
        left_margin = 25Plots.mm,
        right_margin = 10Plots.mm,
        top_margin = 10Plots.mm,
        framestyle=:box,
        legend=:outertop, legend_column=2,
        xlabel="Tiempo",
        ylabel="Forma de onda"
        )

    plot!(plt1, tsteps_train[1:N], pred_waveform_real_train[1:N], label="NN entrenamiento (Re)", linewidth=line_width)
    
    # test waveform
    plt12 = plot(
        tsteps_test, exact_waveform_test,  label="datos (Re)", title= "Predicción de la forma de onda en sistema de test", 
        titlefontsize = title_font_size,
        legendfontsize = legend_font_size,
        gridalpha=grid_alpha,
        gridstyle=grid_style,
        tickfontsize=tick_font_size,
        linewidth=line_width,
        size=(1200,350),
        framestyle=:box,
        legend=false
    )
    
    plot!(plt12, tsteps_test[1:end], pred_waveform_real_test[1:end], label="NN test (Re)")
    plot!(plt12, tsteps_test[1:N], pred_waveform_real_test[1:N], label="NN entrenamiento (Re)")
    
    # losses plot
    plt3 = plot(train_losses, label="entrenamiento", title="Función de coste", yaxis=:log)
    plot!(plt3, test_losses, label="test")

    global plt = plot(plt1, size=(1600,600))
    display(plt)
    if save_plots_gif
        push!(plot_list, plt)
    end

    # Tell sciml_train to not halt the optimization. If return true, then optimization stops.
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

res = DiffEqFlux.sciml_train(
    loss_f, NN_params,
    BFGS(initial_stepnorm=0.01, linesearch = LineSearches.BackTracking()), 
    cb=callback, maxiters = 200
)

# save flux chain models as bson files. To do so, we must save chain model with its parameters
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
)
if ! isfile(metrics_dir*"losses.csv")
    CSV.write(metrics_dir*"losses.csv", losses_df)
else
    x = DataFrame(CSV.File(metrics_dir*"losses.csv", types=Dict("test_name" => String31)))
    append!(x, losses_df)
    CSV.write(metrics_dir*"losses.csv", x)
end


function save_data_images(save_data::bool, img_dir::String)::nothing

    if save_data
        for (ind, img) in enumerate(plot_list)
            savefig(img, img_dir*string(ind)*"_train_img.pdf")
            savefig(img, img_dir*string(ind)*"_train_img.png")
        end
        savefig(plt, output_dir*"prediction_plot.pdf")
    end
    @info "data saved"
end

@info "Execution finished"
