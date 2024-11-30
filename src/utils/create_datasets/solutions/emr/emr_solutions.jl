

function get_true_solution_EMR_schwarzschild(u0::Vector{Float64}, model_params::Vector{Float64}, total_mass::Float64, tspan, tsteps, dt_data::Float64, dt::Float64)
    """
    Computes true solution of a schwarzschild system in Kerr metric
    """
    mass_ratio = 0.0
    exact_problem = ODEProblem(RelativisticOrbitModel_Schwarzschild_EMR, u0, tspan, model_params)
    true_solution = Array(solve(exact_problem, RK4(), saveat = tsteps, dt = dt))
    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return Dict("true_solution" => true_solution, "true_waveform" => true_waveform)
end


function get_true_solution_EMR_kerr(u0::Vector{Float64}, model_params::Vector{Float64}, total_mass::Float64, tspan, tsteps, dt_data::Float64, dt::Float64)
    """
    Computes true solution of a EMR system in Kerr metric
    """
    mass_ratio = 0.0
    exact_problem = ODEProblem(RelativisticOrbitModel_Kerr_EMR, u0, tspan, model_params)
    true_solution = Array(solve(exact_problem, RK4(), saveat = tsteps, dt = dt, adaptive=false))
    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return Dict("true_solution" => true_solution, "true_waveform" => true_waveform)
end
