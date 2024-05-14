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
output_directory = "../../01_data/02_output/01_case_1/case_1_system/"
output_dir = output_directory*test_name
solutions_dir = output_dir*"solutions/"

# Define the experimental parameters
factor = 5
global datasize = 250 * factor
global mass_ratio = 0.0
global dt = 100.0

## Define neural network model
_, _, chain, _ = nn_model_case1(test_name)

# load saved models
@load solutions_dir*"model_chiphi.bson" chain

# restructure chains
NN_params, re = Flux.destructure(chain)
NN(u, NN_params) = re(NN_params)(u)

@info "Generating exact solution..."
# TRAIN
χ₀ = pi; ϕ₀ = 0.0; p=100; M=1.0; e=0.5
u0_train = Float64[χ₀, ϕ₀]
global tspan_train = (0.0f0, factor*6.0f4)
global tsteps_train = range(tspan_train[1], tspan_train[2], length = datasize)
dt_data_train = tsteps_train[2] - tsteps_train[1]
global model_params_train = [p, M, e]

function ODE_model_train(u, NN_params, t)
    du = AbstractNNOrbitModel(u, model_params_train, t, NN=NN, NN_params=NN_params)
    return du
end

prob_train = ODEProblem(RelativisticOrbitModel, u0_train, tspan_train, model_params_train)
exact_sol_train = Array(solve(prob_train, RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
exact_waveform_train = compute_waveform(dt_data_train, exact_sol_train, mass_ratio, model_params_train)[1]

# TEST
χ₀ = pi; ϕ₀ = 0.0; p=100; M=1.0; e=0.8
u0_test = Float64[χ₀, ϕ₀]
global tspan_test = (0.0f0, factor*6.0f4)
global tsteps_test = range(tspan_test[1], tspan_test[2], length = datasize)
dt_data_test = tsteps_test[2] - tsteps_test[1]
global model_params_test = [p, M, e]

function ODE_model_test(u, NN_params, t)
    du = AbstractNNOrbitModel(u, model_params_test, t, NN=NN, NN_params=NN_params)
    return du
end

factor = 5
extended_tspan = (tspan_test[1], factor*tspan_test[2])
extended_tsteps = range(tspan_test[1], factor*tspan_test[2], length = factor*datasize)
prob_test = ODEProblem(RelativisticOrbitModel, u0_test, tspan_test, model_params_test)
exact_sol_test = Array(solve(prob_test, RK4(), saveat = tsteps_test, dt = dt, adaptive=false))
exact_waveform_test = compute_waveform(dt_data_test, exact_sol_test, mass_ratio, model_params_test)[1]

prob_nn_train = ODEProblem(ODE_model_train, u0_train, tspan_train, NN_params)
prob_nn_test = ODEProblem(ODE_model_test, u0_test, tspan_test, NN_params)
NN(u0_train, NN_params) # test it works

# put data into arrays
prob_nn_array = [prob_nn_train, prob_nn_test]
sol_exact_array = [exact_sol_train, exact_sol_test]
tsteps_train_array = [tsteps_train, tsteps_test]
model_params_train_array = [model_params_train, model_params_test]
u0_train_array = [u0_train, u0_test]
dt_data_train_array = [dt_data_train, dt_data_test]

# save predictions
exact_waveform_real_train_array = [exact_waveform_train, exact_waveform_test]
pred_waveform_real_train_array = [];
pred_orbits_array = [];
exact_orbits_array = [];

num_waves=2
for index in range(1, num_waves)

    # solve ode problem
    soln_nn = Array(solve(prob_nn_array[index], RK4(), u0 = u0_train_array[index], p = NN_params, saveat = tsteps_train_array[index], dt = dt, adaptive=false))
    pred_waveform_real_train, pred_waveform_imag_train = compute_waveform(dt_data_train_array[index], soln_nn, 0, model_params_train_array[index])
    pred_orbit = soln2orbit(soln_nn, model_params_train_array[index])
    exact_orbit = soln2orbit(sol_exact_array[index], model_params_train_array[index])

    # save info
    push!(pred_waveform_real_train_array, pred_waveform_real_train)
    push!(pred_orbits_array, pred_orbit)
    push!(exact_orbits_array, exact_orbit)
end

# plot waveforms and orbits
plot_list = [];
title_font_size = 24;
legend_font_size = 18;
line_width=3;
tick_font_size = title_font_size;
grid_alpha=0.4;
grid_style=:dot;


for index in range(1, num_waves)

    if index == 1
        title_ = "entrenamiento"
    else
        title_ = "test"
    end

    m = Int64(round(datasize / factor))

    # waveforms
    x1 = plot(
        tsteps_train_array[index], exact_waveform_real_train_array[index],
        label="datos (Re)",
        title= "Predicción de la forma de onda de "*title_,
        xlabel="Tiempo",
        ylabel="Forma de onda",

        framestyle=:box,

        left_margin = 20Plots.mm,
        bottom_margin = 5Plots.mm,
        size=(1200,350),

        color=:black,
        seriestype=:scatter,
        ms=5,
        markershape=:none,
        )
    if index == 1
        plot!(
            x1,
            tsteps_train_array[index], pred_waveform_real_train_array[index],
            label="NN test (Re)",

            framestyle=:box,

            legendfontsize=legend_font_size,
            titlefontsize=title_font_size,
            guidefontsize=title_font_size,
            linewidth = line_width,
            tickfontsize = tick_font_size;
            gridalpha=grid_alpha,
            gridstyle=grid_style,
            )
        plot!(
            x1, tsteps_train_array[index][1:m], pred_waveform_real_train_array[index][1:m],
            label="NN entrenamiento (Re)",

            legend=:outertop,  legend_column=3,
            framestyle=:box,

            legendfontsize=legend_font_size,
            titlefontsize=title_font_size,
            guidefontsize=title_font_size,
            linewidth = line_width,
            tickfontsize = tick_font_size;
            gridalpha=grid_alpha,
            gridstyle=grid_style,
            )
    else
        plot!(
            x1, tsteps_train_array[index], pred_waveform_real_train_array[index],
            label="NN test (Re)",
            
            legend=false, framestyle=:box,

            legendfontsize=legend_font_size,
            titlefontsize=title_font_size,
            guidefontsize=title_font_size,
            linewidth = line_width,
            tickfontsize = tick_font_size;
            gridalpha=grid_alpha,
            gridstyle=grid_style,
            )
    end

    # orbits
    orbits = pred_orbits_array[index]
    N = size(orbits, 2)
    x2 = plot(
        exact_orbits_array[index][1,:][1:N],exact_orbits_array[index][2,:][1:N],
        aspect_ratio=:equal, 
        title="Predicción de órbitas de "*title_,
        label="datos",
        xlabel="x",
        ylabel="y",

        bottom_margin = 5Plots.mm,
        legend=false,

        legendfontsize=legend_font_size,
        titlefontsize=title_font_size,
        guidefontsize=title_font_size,
        linewidth = line_width,
        tickfontsize = tick_font_size;
        gridalpha=grid_alpha,
        gridstyle=grid_style,
        left_margin = 20Plots.mm,
        framestyle=:box,
        color=:black,
        )

    if index == 1
        plot!(
            x2, orbits[1,1:end-1], orbits[2,1:end-1],
            aspect_ratio=:equal,
            label="NN test",framestyle=:box,
            legendfontsize=legend_font_size,
            titlefontsize=title_font_size,
            guidefontsize=title_font_size,
            linewidth = line_width,
            tickfontsize = tick_font_size;
            gridalpha=grid_alpha,
            gridstyle=grid_style,
            linestyle=:dash
            )
        plot!(x2, orbits[1,1:end-1][1:m], orbits[2,1:end-1][1:m],
            aspect_ratio=:equal,
            label="NN entrenamiento", framestyle=:box,
            legendfontsize=legend_font_size,
            titlefontsize=title_font_size,
            guidefontsize=title_font_size,
            linewidth = line_width,
            tickfontsize = tick_font_size;
            gridalpha=grid_alpha,
            gridstyle=grid_style
            )

    else
        plot!(
            x2, orbits[1,1:end-1], orbits[2,1:end-1],
            aspect_ratio=:equal,
            label="NN test",framestyle=:box,
            legendfontsize=legend_font_size,
            titlefontsize=title_font_size,
            guidefontsize=title_font_size,
            linewidth = line_width,
            tickfontsize = tick_font_size;
            gridalpha=grid_alpha,
            gridstyle=grid_style,
            linestyle=:dash
            )
    end

    push!(plot_list, x1)
    push!(plot_list, x2)
end

l = @layout [[a b]; c{0.4h}; d{0.4h}]
plt = plot(plot_list[2], plot_list[4], plot_list[1], plot_list[3], layout=l, size=(1800,2400))
savefig(plt, solutions_dir*"plot.pdf")
