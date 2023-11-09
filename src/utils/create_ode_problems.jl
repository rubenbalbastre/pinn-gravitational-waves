
function get_problem_information_EMR_schwarzschild(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64,  mass_ratio::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
    """
    Get ODE NN problem and exact solution + waveform
    """

    u0 = Float64[χ₀, ϕ₀]
    tspan = (tspan[1], factor*tspan[2])

    tsteps = range(tspan[1], tspan[2], length = datasize*factor)
    model_params = [p, M, e]
    dt_data = tsteps[2] - tsteps[1]

    function ODE_model(u, NN_params, t)
        du = AbstractNNOrbitModel(u, model_params, t, NN=NN, NN_params=NN_params)
        return du
    end

    prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)
    exact_solution, exact_waveform = get_exact_solution_EMR_schwarzschild(u0, model_params, mass_ratio, tspan, tsteps, dt_data, dt)

    problem = Dict("solution" => exact_solution, "waveform"=> exact_waveform, "nn_problem" => prob_nn, "tsteps"=> tsteps, "model_params"=> model_params, "u0"=> u0, "dt_data"=> dt_data, "tspan" => tspan)

    return problem
end


function get_problem_information_EMR_kerr(χ₀::Float64, ϕ₀::Float64, p::Float64, M::Float64, e::Float64, a::Float64,  mass_ratio::Float64, tspan, datasize::Int64, dt::Float64; factor::Int64 = 1)
    """
    Get ODE NN problem and exact solution + waveform
    """

    u0 = Float64[χ₀, ϕ₀]
    tspan = (tspan[1], factor*tspan[2])

    tsteps = range(tspan[1], tspan[2], length = datasize*factor)
    model_params = [p, M, e, a]
    dt_data = tsteps[2] - tsteps[1]

    function ODE_model(u, NN_params, t)
        du = AbstractNNOrbitModel_kerr(u, model_params, t, NN=NN, NN_params=NN_params)
        return du
    end

    prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)
    exact_solution, exact_waveform = get_exact_solution_EMR_kerr(u0, model_params, mass_ratio, tspan, tsteps, dt_data, dt)

    problem = Dict("solution" => exact_solution, "waveform"=> exact_waveform, "nn_problem" => prob_nn, "tsteps"=> tsteps, "model_params"=> model_params, "u0"=> u0, "dt_data"=> dt_data, "tspan" => tspan)

    return problem
end



function get_exact_solution_EMR_schwarzschild(u0::Vector{Float64}, model_params::Vector{Float64}, mass_ratio::Float64, tspan, tsteps, dt_data::Float32, dt::Float64)
    """
    Get EMR ODE Problem exact solution
    """

    exact_problem = ODEProblem(RelativisticOrbitModel, u0, tspan, model_params)
    exact_solution = Array(solve(exact_problem, RK4(), saveat = tsteps, dt = dt, adaptive=false))
    exact_waveform = compute_waveform(dt_data, exact_solution, mass_ratio, model_params)[1]

    return exact_solution, exact_waveform
end


function get_exact_solution_EMR_kerr(u0::Vector{Float64}, model_params::Vector{Float64}, mass_ratio::Float64, tspan, tsteps, dt_data::Float32, dt::Float64)
    """
    Get EMR ODE Problem exact solution
    """

    exact_problem = ODEProblem(RelativisticOrbitModel_kerr_numerically, u0, tspan, model_params)
    exact_solution = Array(solve(exact_problem, RK4(), saveat = tsteps, dt = dt, adaptive=false))
    exact_waveform = compute_waveform(dt_data, exact_solution, mass_ratio, model_params)[1]

    return exact_solution, exact_waveform
end