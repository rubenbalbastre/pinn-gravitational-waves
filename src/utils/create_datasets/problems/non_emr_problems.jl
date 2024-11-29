"""
This script contains functions to process input data
"""


function string_vector_to_float_vector(string_arr::Vector{String})::Vector{Vector{Float64}}
    """
    Returns a float vector from a string vector
    """

    res = Vector{Vector{Float64}}()

    for i in string_arr
        new_vector = Vector{Float64}()
        arr = replace(i, "[" => "", "]" => "");
        arr_s = split(arr, ",")
        for j in arr_s
            append!(new_vector, parse(Float64, j))
        end
        push!(res, new_vector)
    end

    return res
end


function obtain_q_M(m1::Float64, m2::Float64)::Tuple
    """
    Obtain mass ratio q and total mass M
    Arguments:
        m1: 
        m2:
    Returns:
        q: mass ratio
        M: total mass
    """
    q = m1 / m2
    M = m1 + m2
    return (q, M)
end


function df_to_vectors(info::DataFrame)::Tuple
    """
    Extract dataframe columns as vectors
    Arguments:
        * info: readed info.csv dataframe
    Returns:
        * tuple with the dataframe columns as vectors
    """

    mass_ratio = select(info, "q")[!,1][1];
    IDs = select(info, [:ID])[!,1][1];
    eccentricty =  select( info, [:e] )[!,1][1];
    mA = select( info, [:mA] )[!,1][1];
    mB = select( info, [:mB] )[!,1][1];
    spinA =  select( info, [:spinA] )[!,1][1];
    spinB =  select( info, [:spinB] )[!,1][1];
    initial_sep = select(info, [:initial_separation])[!, 1][1];

    return (mass_ratio, IDs, eccentricty, mA, mB, spinA, spinB, initial_sep)
end


function extract_waveid_info(info::DataFrame, wave_id::String)::Dict
    """
    Returns needed input info from info.csv of a given wave_id
    Arguments:
        * info_df: readed info.csv
        * wave_id: name of one specific waveform
    Returns:
        * mass1
        * mass2
        * q: mass ratio
        * M: total mass
        * model_params: dict containing q and M
        * IDs: vector of waveform IDs
    """

    info = info[ (info[!,"ID"] .== wave_id),:];
    mass_ratio, IDs, eccentricty, mass1, mass2, spinA, spinB, initial_sep = df_to_vectors(info);
    q, M = obtain_q_M(mass1, mass2);
    model_params = Dict("q" => q, "M" => M);

    wave_id_master_data = Dict{String, Any}(
        "mass1" => mass1,
        "mass2" => mass2,
        "q" => q,
        "M" => M,
        "wave_id" => IDs,
        "initial_separation" => initial_sep,
        "eccentricity" => eccentricty,
        "model_params" => model_params
    )

    return wave_id_master_data
end


function gather_waveform_data(tsteps, IDs)::Tuple
    """
    Gather orbits and waveform (real and imaginary parts)
    """
    x, y = file2trajectory(tsteps,"./input/" *IDs[1]*"/trajectoryA_eccentric.txt")
    x2, y2 = file2trajectory(tsteps,"./input/" *IDs[1]*"/trajectoryB_eccentric.txt")
    waveform_real = file2waveform(tsteps,"./input/" *IDs[1]*"/waveform_real_eccentric.txt")
    waveform_imag = file2waveform(tsteps,"./input/" *IDs[1]*"/waveform_imag_eccentric.txt")
    println("Data loaded \n")

    return x, y, x2, y2, waveform_real, waveform_imag
end


function load_sxs_data(wave_list, folder_directory = "../../data/input/case_2/"; datasize=1000)
    """
    Load waveform, orbits and metadata of a set of systems
    """

    dataset = Dict{String, Any}();
        
    for wave_id in wave_list

        # waveform time conditions
        waveform_data = open(folder_directory * wave_id * "/waveform_real.txt", "r")
        waveform_data = readdlm(waveform_data)
        tdata = waveform_data[:, 1]; 
        waveform = waveform_data[:, 2]
        up_limit = argmax(waveform)
        low_limit = Int64(round(size(tdata, 1) * 0.345))
        tspan_train = (tdata[low_limit], tdata[up_limit])
        tsteps_train = range(tspan_train[1], tspan_train[2], length=datasize)
        dt_data = tsteps_train[2] - tsteps_train[1]

        # import data
        println("Importing data: ", wave_id)
        train_x_1, train_y_1 = file2trajectory(tsteps_train, folder_directory * wave_id * "/trajectoryA.txt")
        train_x_2, train_y_2 = file2trajectory(tsteps_train, folder_directory * wave_id * "/trajectoryB.txt")
        true_waveform = file2waveform(tsteps_train, folder_directory * wave_id * "/waveform_real.txt")

        # initial separation
        r₀ = sqrt((train_x_1[1]-train_x_2[1])^2 + (train_y_1[1] - train_y_2[1])^2)

        # ODE parameters
        info_df = DataFrame(CSV.File(folder_directory * "/master_data.csv"))
        wave_id_master_data = extract_waveid_info(info_df, wave_id)

        ϕ₀ = 0.0; 
        χ₀ = pi;
        e₀ = wave_id_master_data["eccentricity"]
        p₀ = r₀ * (1+e₀*cos(χ₀)) / wave_id_master_data["M"]
        u0 = Float32[χ₀, ϕ₀, p₀, e₀]
        
        # add master data information 
        wave_id_master_data["u0"] = u0
        wave_id_master_data["true_waveform"] = true_waveform
        wave_id_master_data["dt_data"] = dt_data
        wave_id_master_data["tspan"] = tspan_train
        wave_id_master_data["tsteps"] = tsteps_train
        wave_id_master_data["true_orbit_x1"] = train_x_1
        wave_id_master_data["true_orbit_y1"] = train_y_1
        wave_id_master_data["true_orbit_x2"] = train_x_2
        wave_id_master_data["true_orbit_y2"] = train_y_2

        # append data to array
        dataset[wave_id] = wave_id_master_data

    end

    return dataset
end



function add_neural_network_problem_to_dataset(dataset, nn_output)
    """
    Update dataset with NN Problem to be solved
    """

    NN_params, NN_chiphi, NN_chiphi_params, NN_pe, NN_pe_params, chain_phichi, chain_pe, re_chiphi, re_pe = nn_output

    for (wave_id, wave_information) in dataset

        function ODE_model(u, NN_params, t)
            """
            Create ODE Model for case non-EMR
            """

            NN_params1 = NN_params[1:l1]
            NN_params2 = NN_params[l1+1:end]

            du = NNOrbitModel_Schwarzchild(
                u, wave_information["model_params"], t,
                NN_chiphi=NN_chiphi, NN_chiphi_params=NN_params1,
                NN_pe=NN_pe, NN_pe_params=NN_params2
            )
            return du
        end

        nn_problem = ODEProblem(
            ODE_model, 
            wave_information["u0"], 
            wave_information["tspan"], 
            NN_params
        )

        wave_information["nn_problem"] = nn_problem
        
    end

    return dataset
end