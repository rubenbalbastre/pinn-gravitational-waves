#=
Auxiliary functions to plot results
=#


function callback_plots(tsteps, wf::Tuple, pred_soln)
    #=
    callback plots
    =#
    waveform_real, pred_waveform_real, waveform_imag, pred_waveform_imag = wf
    N = length(pred_waveform_real)

    # plot waveform -> original and predicted
    plt1 = plot(tsteps, waveform_real,
            markershape=:circle, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5,
            label="wform data (Re)", legend=:topleft)
    plot!(plt1, tsteps[1:N], pred_waveform_real[1:end],
            markershape=:circle, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5,
            label="wform NN (Re)")
    plt2 = plot(tsteps, waveform_imag,
            markershape=:circle, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5,
            label="wform data (Im)", legend=:topleft)
    plot!(plt2, tsteps[1:N], pred_waveform_imag[1:end],
            markershape=:circle, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5,
            label="wform NN (Im)")

    # get solution
    pred_orbit = soln2orbit(pred_soln)
    χ = pred_soln[1,:]
    ϕ = pred_soln[2,:]
    p = pred_soln[3,:]
    e = pred_soln[4,:]
    orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1, mass2)
    N = size(orbit_nn1,2)

    # plot orbit solution
    plt3 = plot(x[1:N],y[1:N],
            linewidth = 2, alpha = 0.5,
            label="orbit data")
    plot!(plt3, orbit_nn1[1,1:end-1], orbit_nn1[2,1:end-1],
            linewidth = 2, alpha = 0.5,
            label="orbit NN")

    # plot e and p
    plt4 = plot(tsteps[1:N], p, linewidth = 2, alpha = 0.5, label="p", legend=:best)
    plt5 = plot(tsteps[1:N], e, linewidth = 2, alpha = 0.5, label="e", legend=:topleft)
    l = @layout [a; b; [c{0.6w} [d;e]]]
    plt = plot(plt1, plt2, plt3, plt4, plt5, layout=l, size=(600,600), legendfontsize=6)

    display(plot(plt))
end


function compute_extrapolated_solutions_case1(
        time_spec, prob, prob_nn, model_params, mass_ratio,
        factor::Int = 5, doplots::Bool = true, saveplots::Bool = true)
        #= 
        Plot extrapolated solutions 
        =#

        # define new time intervals
        datasize, tspan, dt, dt_data = time_spec
        extended_tspan = (tspan[1], factor*tspan[2])
        extended_tsteps = range(tspan[1], factor*tspan[2], length = factor*datasize)

        # obtain solutions to ODE problems
        reference_solution = solve(remake(prob, p = model_params, saveat = extended_tsteps, tspan=extended_tspan),
                                        RK4(), dt = dt, adaptive=false)
        optimized_solution = solve(remake(prob_nn, p = res.minimizer, saveat = extended_tsteps, tspan=extended_tspan),
                                        RK4(), dt = dt, adaptive=false)

        # obtain orbit solutions
        true_orbit = soln2orbit(reference_solution, model_params)
        pred_orbit = soln2orbit(optimized_solution, model_params)

        # compute waveforms
        true_waveform = compute_waveform(dt_data, reference_solution, mass_ratio, model_params)[1]
        pred_waveform = compute_waveform(dt_data, optimized_solution, mass_ratio, model_params)[1]
        
        df_predicted_trajectories = DataFrame(time = extended_tsteps,
                                 true_orbit_x = true_orbit[1,:],
                                 true_orbit_y = true_orbit[2,:],
                                 pred_orbit_x = pred_orbit[1,:],
                                 pred_orbit_y = pred_orbit[2,:],)
        
        df_predicted_waveforms = DataFrame(time = extended_tsteps,
                                        true_waveform = true_waveform,
                                        pred_waveform = pred_waveform)

        return df_predicted_trajectories, df_predicted_waveforms
end


function compute_learned_solutions_case1(time_spec, prob, prob_nn, model_params, mass_ratio)
        #=
        Compute, plot and save learned solutions.
        =#

        # time interval specifications
        tsteps, tspan, dt, dt_data = time_spec

        # solve ODE problems
        reference_solution = solve(remake(prob, p = model_params, saveat = tsteps, tspan=tspan),
        RK4(), dt = dt, adaptive=false)
        optimized_solution = solve(remake(prob_nn, p = res.minimizer, saveat = tsteps, tspan=tspan),
                RK4(), dt = dt, adaptive=false)

        # compute orbits
        true_orbit = soln2orbit(reference_solution, model_params)
        pred_orbit = soln2orbit(optimized_solution, model_params)

        # save in dataframes
        df_learned_trajectories = DataFrame(time = tsteps,
        true_orbit_x = true_orbit[1,:],
        true_orbit_y = true_orbit[2,:],
        pred_orbit_x = pred_orbit[1,:],
        pred_orbit_y = pred_orbit[2,:])

        # compute waveforms
        true_waveform = compute_waveform(dt_data, reference_solution, mass_ratio, model_params)[1]
        pred_waveform = compute_waveform(dt_data, optimized_solution, mass_ratio, model_params)[1]

        df_learned_waveforms = DataFrame(time = tsteps,
                true_waveform = true_waveform,
                pred_waveform = pred_waveform)

        return df_learned_trajectories, df_learned_waveforms
end


function final_plot_case1(
        df_learned_trajectories, df_learned_waveforms, df_predicted_trajectories, df_predicted_waveforms,
        train_losses, test_losses, show_plots)
        #= 
        Paper plot figure 2.
        =#

        # orbits
        plt1 = plot(
                df_predicted_trajectories[!,"true_orbit_x"], df_predicted_trajectories[!,"true_orbit_y"],
                linewidth = 2, label = "truth", linecolor = :blue, legend = :none, framestyle = :none
                )
        plot!(
                plt1, df_predicted_trajectories[!,"pred_orbit_x"], df_predicted_trajectories[!,"pred_orbit_y"],
                linestyle = :dash, linewidth = 2, label = "prediction", linecolor = :red
                )
        plot!(
                plt1,
                df_learned_trajectories[!,"pred_orbit_x"], df_learned_trajectories[!, "pred_orbit_y"],
                linestyle = :dash, linewidth=2, label = "learned", linecolor = :black
                )

        # waveforms
        plt2 = plot(
                df_predicted_waveforms[!, "time"], df_predicted_waveforms[!, "true_waveform"],
                linewidth = 2, label = "truth", linecolor = :blue, legend = :outertop, framestyle = :box
                )
        xlabel!("Time")
        ylabel!("Waveform")
        plot!(
                plt2, df_predicted_waveforms[!, "time"], df_predicted_waveforms[!, "pred_waveform"],
                linestyle = :dash, linewidth = 2, label = "prediction", linecolor = :red
                )
        plot!(
                plt2, 
                df_learned_waveforms[!, "time"], df_learned_waveforms[!, "true_waveform"],
                linestyle = :dash, linewidth = 2, label = "learned", linecolor = :black
                )

        # loss function
        plt3 = plot(range(1, length(train_losses)), train_losses, linewidth=2, linecolor = :black, label="train", yaxis = :log)
        xlabel!(plt3, "Iterations")
        ylabel!(plt3, "Loss function")
        plot!(plt3, range(1, length(test_losses)), test_losses, linewidth=2, linecolor = :red, label="test", yaxis = :log)

        # layout
        # l = @layout [ a b c{1.0w} ] # grid(1,2) 
        plt = plot(plt1, plt3, plt2, size=(600,600), legendfontsize=3)

        if show_plots
                display(plot(plt))
        end

        return plt
end
