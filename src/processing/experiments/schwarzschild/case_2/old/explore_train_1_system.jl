cd(@__DIR__)
using Pkg; Pkg.activate("../../");
using OrdinaryDiffEq
using Optim
using LineSearches
using DiffEqFlux
using DiffEqSensitivity
using Plots
using DataFrames
using CSV
using BSON: @load
using Flux
import Logging
include("../04_utils/orbital_mechanics_utils.jl");
include("../04_utils/input_preparation.jl");
include("../04_utils/models.jl");
include("../04_utils/metrics.jl");
include("../04_utils/plots.jl")
include("../04_utils/loss_functions.jl")
include("../04_utils/output.jl")
include("../04_utils/nn_models.jl")

dt = 10

# script conditions
show_plots = true
save_plots_gif = false
save_data = true
model_name = "test/"
test_name = "test_all_n/"
output_directory = "../../01_data/02_output/02_case_2/n_system/"
output_dir = output_directory*test_name
solutions_dir = output_dir*"solutions/"

# data
wave_id = [
    # "SXS:BBH:0169",
    # "SXS:BBH:0168",
    "SXS:BBH:0217",
    # "SXS:BBH:0211",
    "SXS:BBH:0072"
    # "SXS:BBH:0001",
    # "SXS:BBH:2085"
]
train_array, test, train, wave_id_dict = load_data(wave_id, source_path="02_case_2", datasize=1500)
test_x_1, test_x_2, test_y_1, test_y_2, exact_test_wf_real, tsteps_test, tspan_test, model_params_test, u0_test, dt_data_test, mass1_test, mass2_test = test
train_x_1, train_x_2, train_y_1, train_y_2, exact_train_wf_real, tsteps_train, tspan_train, model_params_train, u0_train, dt_data_train, mass1_train, mass2_train = train
train_x_1_array, train_x_2_array, train_y_1_array, train_y_2_array, exact_train_wf_real_array, tsteps_train_array, tspan_train_array,model_params_array, u0_array, dt_data_array, mass1_train_array, mass2_train_array = train_array

## Define neural network model
_, _, _, _, _, chain_chiphi, chain_pe, _, _ = nn_model_case2(model_name, 32, tanh)

# load saved models
@load solutions_dir*"model_chiphi.bson" chain_phichi
@load solutions_dir*"model_pe.bson" chain_pe

# restructure chains
NN_phichi_params, re_chiphi = Flux.destructure(chain_phichi)
NN_phichi(u, NN_phichi_params) = re_chiphi(NN_phichi_params)(u)
NN_pe_params, re_pe = Flux.destructure(chain_pe)
NN_pe(u, NN_pe_params) = re_pe(NN_pe_params)(u)

# define ode model
NN_params = vcat(NN_phichi_params,NN_pe_params)
l1 = length(NN_phichi_params)

function ODE_model_train(u, NN_params, t)
    NN_params1 = NN_params[1:l1]
    NN_params2 = NN_params[l1+1:end]
    du = NR_OrbitModel_Ruben(u, model_params_train, t,
                              NN_chiphi=NN_phichi, NN_chiphi_params=NN_params1,
                              NN_pe=NN_pe, NN_pe_params=NN_params2)
    return du
end

# Getting predictions
num_waves = length(wave_id)
pred_waveform_real_train_array = [];
solutions_array = [];
orbits_1_array = [];
orbits_2_array = [];

for index in range(1, num_waves)

    # solve ode problem
    prob_nn_train = ODEProblem(ODE_model_train, u0_array[index], tspan_train_array[index], NN_params)
    soln_nn = Array(solve(prob_nn_train, RK4(), u0 = u0_array[index], p = NN_params, saveat = tsteps_train_array[index], dt = dt, adaptive=false))
    pred_waveform_real_train, pred_waveform_imag_train = compute_waveform(dt_data_array[index], soln_nn, model_params_array[index][1], model_params_array[index])
    pred_orbit = soln2orbit(soln_nn, model_params_array[index])
    orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1_train_array[index], mass2_train_array[index])

    # save info
    push!(pred_waveform_real_train_array, pred_waveform_real_train)
    push!(solutions_array, soln_nn)
    push!(orbits_1_array, orbit_nn1)
    push!(orbits_2_array, orbit_nn2)
end

# plot waveforms and orbits
plot_list = [];

function set_title(index:: Int64):: String

    if index == 1
        return wave_id_dict[index]
    elseif index == 2
        return wave_id_dict[index]
    end
end

title_font_size = 20;
legend_font_size = 16;
line_width=3;
tick_font_size = title_font_size;
grid_alpha=0.4;
grid_style=:dot;

for index in range(1, num_waves)

    # waveforms
    x1 = plot(tsteps_train_array[index], exact_train_wf_real_array[index],
        label="datos (Re)", title=set_title(index),
        titlefontsize = 24,
        legendfontsize = legend_font_size,
        guidefontsize=title_font_size,
        gridalpha=grid_alpha,
        gridstyle=grid_style,
        tickfontsize=tick_font_size,
        color=:black,
        seriestype=:scatter,
        ms=5,
        markershape=:none,
        size=(1200, 450),
        # bottom_margin = 10Plots.mm,
        left_margin = 25Plots.mm,
        right_margin = 10Plots.mm,
        # top_margin = 10Plots.mm,
        framestyle=:box,
        legend=:top, 
        legend_column=2,
        xlabel="Tiempo",
        ylabel="Forma de onda"
        )
    plot!(x1, tsteps_train_array[index], pred_waveform_real_train_array[index],
        linewidth = line_width, label="NN (Re)")

    # orbits
    orbit_nn1 = orbits_1_array[index]
    orbit_nn2 = orbits_2_array[index]
    N = size(orbit_nn1, 2)
    x2 = plot(
        train_x_1_array[index][1:N], train_y_1_array[index][1:N],
        label="datos", 
        # title=set_title(index),
         aspect_ratio=:equal,
        titlefontsize = title_font_size,
        legendfontsize = legend_font_size,
        guidefontsize=title_font_size,
        gridalpha=grid_alpha,
        gridstyle=grid_style,
        tickfontsize=tick_font_size,
        markershape=:none,
        color=:black,
        # bottom_margin = 10Plots.mm,
        left_margin = 20Plots.mm,
        right_margin = 10Plots.mm,
        # top_margin = 5Plots.mm,
        framestyle=:box,
        # legend=:outertop,
        # legend_column=2,
        linewidth=line_width,
        xlabel="x",
        ylabel="y"
        )
    plot!(x2, orbit_nn1[1,1:end-1], orbit_nn1[2,1:end-1],
        linewidth=line_width,
        label="NN", 
        # title= ""*wave_id_dict[index], 
        linestyle=:dash
        )

    # p, e
    p = solutions_array[index][3,:]
    e = solutions_array[index][4, :]
    x3 = plot(tsteps_train_array[index][1:N], p, label="semi-latus rectum",
        #  title=set_title(index),
        titlefontsize = title_font_size,
        legendfontsize = legend_font_size,
        guidefontsize=title_font_size,
        gridalpha=grid_alpha,
        gridstyle=grid_style,
        tickfontsize=tick_font_size,
        markershape=:none,
        # bottom_margin = 10Plots.mm,
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        # top_margin = 10Plots.mm,
        framestyle=:box,
        legend=:best,
        linewidth=line_width,
        xlabel="Tiempo"
    )
    plot!(twinx(), tsteps_train_array[index][1:N], e, linewidth = line_width, color=2, label="excentricidad",
    titlefontsize = title_font_size,
    legendfontsize = legend_font_size,
    guidefontsize=title_font_size,
    gridalpha=grid_alpha,
    gridstyle=grid_style,
    tickfontsize=tick_font_size,
    framstyle=:box,
    legend=:left
    )

    l = @layout [[a{0.4w} b]; c{0.6h}]
    x = plot(x3, x2, x1, layout=l)
    push!(plot_list, x)
end

l = @layout [grid(2,1)]
plt = plot(plot_list..., layout=l, size=(2000,3000))
savefig(plt, solutions_dir*"plot2.png")
savefig(plt, solutions_dir*"plot2.pdf")
