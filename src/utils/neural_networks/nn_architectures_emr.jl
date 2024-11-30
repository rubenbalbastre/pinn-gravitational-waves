

function process_chain(chain:: Chain; initialize_as_zero::Bool = true)
    """
    Extract parameters from chain. By default, initialize them to zero
    """

    NN_params, re = Flux.destructure(chain)

    if initialize_as_zero
        println("Reducing weights two orders of magnitued")
        NN_params::Vector{Float64} = NN_params / 100
    end

    NN(u, NN_params) = re(NN_params)(u)

    return NN, NN_params, chain, re
end


function nn_model_emr_kerr_from_schwarzschild(number_of_neurons_in_hidden_layer::Int64, activation_function; type::String = "standard")
    """
    Define multiple configurations of architecture of type 1
    # softmodulusQ, leakyrelu, relu, cos, tanh, abs, sigmoid
    """

    if type == "standard"
        chain = Chain(
            # x = [χ, ϕ, a, p, M, e]
            # x -> Float32.([normalize_chi(x[1]), x[3], x[3]^2, x[3]^3, 1/x[3], 1/x[3]^2]),
            # Dense(6, Int(number_of_neurons_in_hidden_layer/2), activation_function),
            x -> Float32.([x[1], x[1]*x[3], x[3],  x[3]^2, x[3]^3]),
            Dense(5, Int(number_of_neurons_in_hidden_layer/2), activation_function),    
            Dense(Int(number_of_neurons_in_hidden_layer/2), number_of_neurons_in_hidden_layer, activation_function),
            Dense(number_of_neurons_in_hidden_layer, number_of_neurons_in_hidden_layer, activation_function),
            # Dense(number_of_neurons_in_hidden_layer, number_of_neurons_in_hidden_layer, activation_function),
            # Dropout(0.8),
            Dense(number_of_neurons_in_hidden_layer, 2)
        )
    end

    NN, NN_params, chain, re = process_chain(chain)

    return NN, NN_params, chain, re
end


function nn_model_emr_kerr_from_newton(number_of_neurons_in_hidden_layer::Int64, activation_function; type::String = "standard")
    """
    Define multiple configurations of architecture of type 1
    # softmodulusQ, leakyrelu, relu, cos, tanh, abs, sigmoid
    """

    if type == "standard"
        chain = Chain(
            # x = [χ, ϕ, a, p, M, e]
            # x -> Float32.([normalize_chi(x[1]), x[3], x[3]^2, x[3]^3, 1/x[3], 1/x[3]^2]),
            # Dense(6, Int(number_of_neurons_in_hidden_layer/2), activation_function),
            x -> Float32.([x[1], x[1]*x[3], x[3],  x[3]^2, x[3]^3]),
            Dense(5, Int(number_of_neurons_in_hidden_layer/2), activation_function),    
            Dense(Int(number_of_neurons_in_hidden_layer/2), number_of_neurons_in_hidden_layer, activation_function),
            Dense(number_of_neurons_in_hidden_layer, number_of_neurons_in_hidden_layer, activation_function),
            # Dense(number_of_neurons_in_hidden_layer, number_of_neurons_in_hidden_layer, activation_function),
            # Dropout(0.8),
            Dense(number_of_neurons_in_hidden_layer, 2)
        )
    end

    NN, NN_params, chain, re = process_chain(chain)

    return NN, NN_params, chain, re
end


function nn_model_case1_arch1(n::Int64, f; type::String = "standard")
    """
    Define multiple configurations of architecture of type 1
    # softmodulusQ, leakyrelu, relu, cos, tanh, abs, sigmoid
    """

    if type == "standard"
        chain = Chain(
            x -> convert.(Float32, x),
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            Dense(1, n, f),
            Dense(n, n, f),
            Dense(n, 2)
        )
    elseif type == "simple"
        chain = Chain(
            x -> convert.(Float32, x),
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            Dense(1, n, f),
            Dense(n, 2)
        )
    end

    NN, NN_params, chain, re = process_chain(chain)

    return NN, NN_params, chain, re
end


function nn_model_case1(test:: String)
    """
    NN architectures for experiment 1
    """
        
    if test == "LSTM_dense/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            LSTM(1, 6),
            Dense(6, 10, cos),
            Dense(10,2)
        )

    elseif test == "LSTM/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            LSTM(1, 2),
            Dense(2,2)
        )

    elseif test == "LSTM_2/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            LSTM(1, 16),
            Dense(16,2)
        )

    elseif test == "LSTM_3/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            LSTM(1, 32),
            Dense(32, 32, cos),
            Dense(32,2)
        )

    elseif test == "GRU_3/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            GRU(1, 32),
            Dense(32, 32, cos),
            Dense(32,2)
        )
    
    elseif test == "GRU_2/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            GRU(1, 16),
            Dense(16,2)
        )

    elseif test == "GRU/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            GRU(1, 2),
            Dense(2,2)
        )

    ########################################################################

    elseif test == "test_1_adaptative_5/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, sigmoid), # [coef*cos(coef*x[1]) + coef*sin(coef*x[1]) for coef in x]
            x -> map(x -> x[1] + sum([coef*cos(coef*x[1]) for coef in x[2:end]]), [map(x -> x, x[1:16]), repeat(x[16:end], 1, 16)]),
            Dense(2,2)
        )

    elseif test == "test_1_custom_af/"
        chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 32, custom_act_function),
        Dense(32, 32, custom_act_function),
        Dense(32, 2)
    )

    ########################################################################

    elseif test == "test_1_no_input_cos/"
    chain = Chain(x -> [x[1]],
        Dense(1,32, cos),
        Dense(32, 32, cos),
        Dense(32, 2)
    )

    elseif test == "test_3_cos/"
        chain = Chain(x -> [x[1]],
            Dense(1,32, cos),
            Dense(32, 64, cos),
            Dense(64, 32, cos),
            Dense(32, 2)
        )

    elseif test == "ori_dropout/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, tanh),
            Dense(32, 32, tanh),
            Dropout(0.5),
            Dense(32, 2)
        )

    ########################################################################

    elseif test == "encoder/"
        chain = Chain(
            x -> convert.(Float32, x),
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            Dense(1, 32, cos),
            Dense(32, 32, cos),
            Dense(32,16, tanh),
            Dense(16, 2, tanh),
            Dense(2, 32, tanh),
            Dense(32, 2)
        )

    else
        @error "Architecture non correctly specified" * test
    end

    NN, NN_params, chain, re = process_chain(chain)

    return NN, NN_params, chain, re
end


function nn_model_case1_diff_wf(test:: String)::Tuple
    """
    NN architectures for experiment 1 of case 1 (EMR)
    """

    # 01_original
    if test == "nn1/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, cos),
            Dense(32, 32, cos),
            Dense(32, 2)
        )
    elseif test == "nn1_pe/"
        chain = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])), 1/(1-x[5]), 1/sqrt(abs(1-x[5])), 1/(1-x[5])^2,sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2],
            Dense(12, 32, cos),
            Dense(32, 32, cos),
            Dense(32, 2)
        )
    elseif test == "nn2/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, cos),
            Dense(32, 2)
        )

    elseif test == "nn2_pe/"
        chain = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])), 1/(1-x[5]), 1/sqrt(abs(1-x[5]^2)), 1/sqrt(abs(1-x[5])), 1/(1-x[5])^2, sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1])],
            x -> convert.(Float32, x),
            Dense(15, 32, cos),
            Dense(32, 2)
        )

    elseif test == "nn3_pe/"
        chain = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])), 1/(1-x[5]), 1/sqrt(abs(1-x[5])), 1/(1-x[5])^2,sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2],
            x -> convert.(Float32, x),
            LSTM(12, 16),
            Dense(16, 32, cos),
            Dense(32, 2)
        )

    elseif test == "nn3/"
        chain = Chain(
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            LSTM(1, 16),
            Dense(16, 32, cos),
            Dense(32, 2)
        )

    elseif test == "nn4/"
        chain = Chain(
            # x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])), 1/(1-x[5]), 1/sqrt(abs(1-x[5])), 1/(1-x[5])^2, sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1])],
            # x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])), 1/(1-x[5]), 1/sqrt(abs(1-x[5])), 1/(1-x[5])^2, sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1])],
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])), 1/(1-x[5]), 1/sqrt(abs(1-x[5])), 1/(1-x[5])^2,sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2],
            x -> convert.(Float32, x),
            LSTM(12, 14),
            Dense(14, 32, cos),
            Dense(32, 2)
        )
        
    else
        @error "Architecture non correctly specified"
    end

    NN, NN_params, chain, re = process_chain(chain)

    return NN, NN_params, chain, re
end

