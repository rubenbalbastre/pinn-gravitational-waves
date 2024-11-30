title_font_size = 24;
legend_font_size = 18;
line_width=3;
tick_font_size = 24;
grid_alpha=0.4;
grid_style=:dot;


function create_waveform_title(p, M, e, a)
    """
    Create title based on EMR parameters
    """

    title = "a = " * string(a) * ", p = " * string(p) * ", M = " * string(M) * ", e = " * string(e)

    return title
end


function train_plot(tsteps, true_waveform, predicted_waveform; true_label="data", predict_label = "NN prediction", title = nothing, xlabel = "Time", ylabel = "Waveform", size = (1600,600))
    """
    Real waveform vs predicted waveform at zero training step.
    """

    N = length(tsteps)

    plt = plot(
        tsteps, true_waveform, label=true_label, 
        # titlefontsize = title_font_size,
        # legendfontsize = legend_font_size,
        # guidefontsize=title_font_size,
        # gridalpha=grid_alpha,
        # gridstyle=grid_style,
        # tickfontsize=tick_font_size,
        color=:black,
        # seriestype=:scatter,
        ms=4,
        markershape=:o,
        size=size,
        margin=10Plots.mm,
        framestyle=:box,
        legend=:outertop,
        legend_column=2,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )

    plot!(plt, tsteps[1:N], predicted_waveform[1:N], label=predict_label, linewidth=line_width)

    return plt
end


function test_plot(tsteps_train, tsteps, true_waveform, predicted_waveform; true_label="data", predict_label = "NN prediction", title = "", xlabel = "Time", ylabel = "Waveform",size=(1600, 600))
    """
    Real waveform vs predicted waveform at zero training step.
    """

    N = length(tsteps_train)
    plt = plot(tsteps, true_waveform, label="True waveform", color=:blue4, linewidth=3)
    plot!(plt, tsteps, predicted_waveform, label=predict_label, linestyle=:dash, color=:red, linewidth=3)
    plot!(
        plt,
        tsteps_train, true_waveform[1:N],  
        label=true_label, 
        # titlefontsize = title_font_size,
        # legendfontsize = legend_font_size,
        # gridalpha=grid_alpha,
        # gridstyle=grid_style,
        # tickfontsize=tick_font_size,
        color=:black,
        markershape=:o,
        markersize=4,
        # linewidth=line_width,
        plotstyle=:scatter,
        size=size,
        framestyle=:box,
        title=title, 
        xlabel=xlabel,
        ylabel=ylabel,
        # legend=false,
        margin=10Plots.mm,
    )

    return plt
end



function losses_plot(train_losses, test_losses; train_label="entrenamiento", test_label="test", title="Función de coste")
    """
    Losses plot.
    """
    plt = plot(train_losses, label=train_label, title=title, yaxis=:log)
    plot!(plt, test_losses, label=test_label)

    return plt
end