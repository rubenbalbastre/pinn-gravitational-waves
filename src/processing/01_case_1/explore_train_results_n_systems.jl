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
test_name = "nn2/"
output_directory = "../../01_data/02_output/01_case_1/case_n_systems/"
output_dir = output_directory*test_name
solutions_dir = output_dir*"solutions/"

# Define the experimental parameters
global datasize = 250
global mass_ratio = 0.0
global dt = 100.0

## Define neural network model
_, _, chain, _ = nn_model_case1_diff_wf(test_name)

# load saved models
@load solutions_dir*"model_chiphi.bson" chain

# restructure chains
NN_params, re = Flux.destructure(chain)
NN(u, NN_params) = re(NN_params)(u)

@info "Generating exact solution..."
# TRAIN
χ₀ = pi; ϕ₀ = 0.0; p=100; M=1.0; e=0.8
u0_train = Float64[χ₀, ϕ₀, p, M, e]
global tspan_train = (0.0f0, 6.0f4)
global tsteps_train = range(tspan_train[1], tspan_train[2], length = datasize)
dt_data_train = tsteps_train[2] - tsteps_train[1]
global model_params_train = [p, M, e]

function ODE_model_train(u, NN_params, t)
    du = AbstractNNOrbitModel_Ruben(u, model_params_train, t, NN=NN, NN_params=NN_params)
    return du
end

prob_train = ODEProblem(RelativisticOrbitModel_Ruben, u0_train, tspan_train, model_params_train)
exact_sol_train = Array(solve(prob_train, RK4(), saveat = tsteps_train, dt = dt, adaptive=false))
exact_waveform_train = compute_waveform(dt_data_train, exact_sol_train, mass_ratio, model_params_train)[1]

# TEST
χ₀ = pi; ϕ₀ = 0.0; p=100; M=1.0; e=0.3
u0_test = Float64[χ₀, ϕ₀, p, M, e]
global tspan_test = (0.0f0, 6.0f4)
global tsteps_test = range(tspan_test[1], tspan_test[2], length = datasize)
dt_data_test = tsteps_test[2] - tsteps_test[1]
global model_params_test = [p, M, e]

function ODE_model_test(u, NN_params, t)
    du = AbstractNNOrbitModel_Ruben(u, model_params_test, t, NN=NN, NN_params=NN_params)
    return du
end

factor = 5
extended_tspan = (tspan_test[1], factor*tspan_test[2])
extended_tsteps = range(tspan_test[1], factor*tspan_test[2], length = factor*datasize)
prob_test = ODEProblem(RelativisticOrbitModel_Ruben, u0_test, tspan_test, model_params_test)
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

for index in range(1, num_waves)

    if index == 1
        title = "Train predictions" 
    else 
        title = "Test predictions"
    end

    # waveforms
    x1 = plot(tsteps_train_array[index], exact_waveform_real_train_array[index], markershape=:none,
        linewidth = 2, alpha = 0.5, label="wform data (Re)", legend=:topleft, title= title)
    plot!(x1, tsteps_train_array[index], pred_waveform_real_train_array[index], markershape=:none,
        linewidth = 2, alpha = 0.5, label="wform NN (Re)")

    # orbits
    orbits = pred_orbits_array[index]
    N = size(orbits, 2)
    x2 = plot(exact_orbits_array[index][1,:][1:N],exact_orbits_array[index][2,:][1:N],
        linewidth = 2, alpha = 0.5, aspect_ratio=:equal,
        label="orbit data")
    plot!(x2, orbits[1,1:end-1], orbits[2,1:end-1],
        linewidth = 2, alpha = 0.5, aspect_ratio=:equal,
        label="orbit NN")

    # l = @layout [a; b]
    x = plot(x1, x2, layout=grid(1,2))
    push!(plot_list, x)
end

# l = @layout [a; b] # [a; b; c; d]
plt = plot(plot_list..., layout=grid(2,1))
# plot!(size=(2000,900))
display(plt)
savefig(plt, solutions_dir*"orbits_plot.png")
