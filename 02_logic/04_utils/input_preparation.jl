
function string_vector_to_float_vector(string_arr::Vector{String})
    #=
    Returns a float vector from a string vector
    =#

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
end;


function obtain_m1_m2(q::Float64, M::Float64)::Tuple
    #=
    Obtain m1 and m2 from mass ratio q and total mass M
    =#

    m1 = q * M / (1.0 + q)
    m2 = M / (1.0 + q)

    return (m1, m2)
end


function obtain_q_M(m1::Float64, m2::Float64)::Tuple
    #= 
    Obtain mass ratio q and total mass M
    Arguments:
        * m1: 
        * m2:
    Returns:
        q: mass ratio
        M: total mass
    =#
    q = m1 / m2
    M = m1 + m2
    return (q, M)
end


function df_to_vectors(info::DataFrame)::Tuple
    #=
    Extract dataframe columns as vectors
    Arguments:
        * info: readed info.csv dataframe
    Returns:
        * tuple with the dataframe columns as vectors
    =#
    mass_ratio = select(info, "q")[!,1];
    IDs = select(info, [:ID])[!,1];
    eccentricty =  select( info, [:e] )[!,1];
    mA = select( info, [:mA] )[!,1];
    mB = select( info, [:mB] )[!,1];
    spinA =  select( info, [:spinA] )[!,1];
    spinB =  select( info, [:spinB] )[!,1];
    initial_sep = select(info, [:initial_separation])[!, 1];

    return (mass_ratio, IDs, eccentricty, mA, mB, spinA, spinB, initial_sep)
end


function extract_waveid_info(info::DataFrame, wave_id::String)::Tuple
    #=
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
    =#
    # println(start_message)

    # filter 
    info = info[ (info[!,"ID"] .== wave_id),:];
    # extract the theoretical parameters from info.csv file
    mass_ratio, IDs, eccentricty, mA, mB, spinA, spinB, initial_sep = df_to_vectors(info);

    mass1 = mA[1]; mass2 = mB[1];
    q, M = obtain_q_M(mass1, mass2);
    model_params = Dict("q" => q, "M" => M);

    return mass1, mass2, q, M, model_params, IDs, initial_sep[1], eccentricty

end


function gather_waveform_data(tsteps, IDs)::Tuple
    #=
    Gather orbits and waveform (real and imaginary parts)
    =#
    x, y = file2trajectory(tsteps,"./input/" *IDs[1]*"/trajectoryA_eccentric.txt")
    x2, y2 = file2trajectory(tsteps,"./input/" *IDs[1]*"/trajectoryB_eccentric.txt")
    waveform_real = file2waveform(tsteps,"./input/" *IDs[1]*"/waveform_real_eccentric.txt")
    waveform_imag = file2waveform(tsteps,"./input/" *IDs[1]*"/waveform_imag_eccentric.txt")
    @info "Data loaded \n"

    return x, y, x2, y2, waveform_real, waveform_imag
end



function load_data(wave_list; source_path = "02_case_2", datasize=1000)

    train_x_1_array = [];
    train_y_1_array = [];
    train_x_2_array = [];
    train_y_2_array = [];
    exact_train_wf_real_array = [];
    model_params_array = [];
    tspan_train_array = [];
    tsteps_train_array = [];
    dt_data_array = [];
    mass1_train_array = [];
    mass2_train_array = [];
    u0_array = [];
    wave_id_dict = Dict();
        
    for (index, w_id) in enumerate(wave_list)

        # save dict name
        wave_id_dict[index] = w_id

        # path to wave id data
        folder_dir = "../../01_data/01_input/"*source_path*"/"*w_id*"/"

        # waveform time conditions
        f = open(folder_dir*"waveform_real.txt", "r")
        data = readdlm(f)
        tdata = data[:, 1]; wf = data[:, 2]
        up_limit = argmax(wf); low_limit = Int64(round(size(tdata, 1) * 0.345))
        tspan_train = (tdata[low_limit], tdata[up_limit])
        tsteps_train = range(tspan_train[1], tspan_train[2], length=datasize)
        dt_data = tsteps_train[2] - tsteps_train[1]

        # import data
        println("Importing data: ", w_id)
        train_x_1, train_y_1 = file2trajectory(tsteps_train, folder_dir*"trajectoryA.txt")
        exact_train_wf_real = file2waveform(tsteps_train, folder_dir*"waveform_real.txt")
        train_x_2, train_y_2 = file2trajectory(tsteps_train, folder_dir*"trajectoryB.txt")

        # initial separation
        r₀ = sqrt((train_x_1[1]-train_x_2[1])^2 + (train_y_1[1] - train_y_2[1])^2)

        # ODE parameters
        info_df = DataFrame(CSV.File("../../01_data/01_input/"* source_path*"/info.csv"))
        mass1_train, mass2_train, q, M, model_params, IDs, initial_sep, eccentricty = extract_waveid_info(info_df, w_id)
        model_params_train = [q, M]
        ϕ₀ = 0.0; χ₀ = pi; e₀ = eccentricty[1]
        p₀ = r₀ * (1+e₀*cos(χ₀)) / M
        u0_train = Float32[χ₀, ϕ₀, p₀, e₀, q, M]

        # append data to array
        push!(train_x_1_array, train_x_1)
        push!(train_y_1_array, train_y_1)
        push!(train_x_2_array, train_x_2)
        push!(train_y_2_array, train_y_2)
        push!(exact_train_wf_real_array, exact_train_wf_real)
        push!(model_params_array, model_params_train)
        push!(tsteps_train_array, tsteps_train)
        push!(tspan_train_array, tspan_train)
        push!(u0_array, u0_train)
        push!(dt_data_array, dt_data)
        push!(mass1_train_array, mass1_train)
        push!(mass2_train_array, mass2_train)
    end

    # TEST SPLIT (the rest is used for train)

    train_index = 2
    train_x_1 = train_x_1_array[train_index]
    train_y_1 = train_y_1_array[train_index]
    train_x_2 = train_x_2_array[train_index]
    train_y_2 = train_y_2_array[train_index]
    exact_train_wf_real = exact_train_wf_real_array[train_index]
    tsteps_train = tsteps_train_array[train_index]
    tspan_train = tspan_train_array[train_index]
    model_params_train = model_params_array[train_index]
    u0_train = u0_array[train_index]
    dt_data_train = dt_data_array[train_index]
    mass1_train = mass1_train_array[train_index]
    mass2_train = mass2_train_array[train_index]

    test_index = 1
    test_x_1 = train_x_1_array[test_index]
    test_y_1 = train_y_1_array[test_index]
    test_x_2 = train_x_2_array[test_index]
    test_y_2 = train_y_2_array[test_index]
    exact_test_wf_real = exact_train_wf_real_array[test_index]
    tsteps_test = tsteps_train_array[test_index]
    tspan_test = tspan_train_array[test_index]
    model_params_test = model_params_array[test_index]
    u0_test = u0_array[test_index]
    dt_data_test = dt_data_array[test_index]
    mass1_test = mass1_train_array[test_index]
    mass2_test = mass2_train_array[test_index]

    test = [test_x_1, test_x_2, test_y_1, test_y_2, exact_test_wf_real, tsteps_test, tspan_test, model_params_test, u0_test, dt_data_test, mass1_test, mass2_test]
    train = [train_x_1, train_x_2, train_y_1, train_y_2, exact_train_wf_real, tsteps_train, tspan_train, model_params_train, u0_train, dt_data_train, mass1_train, mass2_train]
    train_array = [train_x_1_array, train_x_2_array, train_y_1_array, train_y_2_array, exact_train_wf_real_array, tsteps_train_array, tspan_train_array,model_params_array, u0_array, dt_data_array, mass1_train_array, mass2_train_array]
    wave_id_dict

    return train_array, test, train, wave_id_dict
end
