using Flux;


function softmodulusQ(x)
    if abs(x) <= 1
        res = x^2 * (2 - abs(x))
    elseif abs(x) > 1
        res = abs(x)
    end
    return res
end

global custom_act_function_coef = [0,0,0,0]
function custom_act_function(x; custom_act_function_coef=custom_act_function_coef)

    # println(custom_act_function_coef)
    res = custom_act_function_coef[1]*tanh(x) + custom_act_function_coef[2]*sigmoid(x) + custom_act_function_coef[3]*cos(x) + custom_act_function_coef[4]*softmodulusQ(x)

end

function nn_model_case1(test:: String)
    """
    NN architectures for experiment 1
    """

    ########################################################################

    if test == "test_1_cos/"
        n_ = 28
        chain = Chain(
            x -> convert.(Float32, x),
            x -> [cos(x[1])],
            x -> convert.(Float32, x),
            Dense(1, n_, cos),
            Dense(n_, n_, cos),
            Dense(n_, 2)
        )

    elseif test == "test_1_tanh/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, tanh),
            Dense(32, 32, tanh),
            Dense(32, 2)
        )

    elseif test == "test_1_relu/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, relu),
            Dense(32, 32, relu),
            Dense(32, 2)
        )

    elseif test == "test_1_leaky_relu/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, leakyrelu),
            Dense(32, 32, leakyrelu),
            Dense(32, 2)
        )

    elseif test == "test_1_abs/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, softmodulusQ),
            Dense(32, 32, softmodulusQ),
            Dense(32, 2)
        )

    elseif test == "test_1_sigmoid/"
        chain = Chain(
            x -> [cos(x[1])],
            Dense(1, 32, sigmoid),
            Dense(32, 32, sigmoid),
            Dense(32, 2)
        )

   ########################################################################

    elseif test == "simple_cos/"
    chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 16, cos),
        Dense(16, 2)
    )

    elseif test == "simple_tanh/"
    chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 16, tanh),
        Dense(16, 2)
    )
    
    elseif test == "simple_abs/"
    chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 16, softmodulusQ),
        Dense(16, 2)
    )

    elseif test == "simple_relu/"
    chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 16, relu),
        Dense(16, 2)
    )

    elseif test == "simple_leaky_relu/"
    chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 16, leakyrelu),
        Dense(16, 2)
    )

    elseif test == "simple_sigmoid/"
    chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 16, leakyrelu),
        Dense(16, 2)
    )

    ########################################################################

    elseif test == "LSTM_dense/"
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

    ########################################################################

    elseif test == "original_mp/"
    chain = Chain(
        x -> [x[1], 1/x[1], x[1]^2, cos(x[1]), sin(x[1]), 1/cos(x[1]), 1/sin(x[1]), tan(x[1]), 1/tanh(x[1])],
        Dense(9, 32, cos),
        Dense(32, 32, cos),
        Dense(32, 2)
    )

    ########################################################################

    elseif test == "ori_dropout/"
    chain = Chain(
        x -> [cos(x[1])],
        Dense(1, 32, tanh),
        Dense(32, 32, tanh),
        Dropout(0.5),
        Dense(32, 2)
    )


    else
        @error "Architecture non correctly specified"
    end

    NN_params, re = Flux.destructure(chain)
    NN_params = NN_params .* 0
    NN(u, NN_params) = re(NN_params)(u)

    return NN, NN_params, chain, re
end


function nn_model_case1_diff_wf(test:: String)
    """
    NN architectures for experiment 1
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

    NN_params, re = Flux.destructure(chain)
    NN_params = NN_params .* 0
    NN(u, NN_params) = re(NN_params)(u)

    return NN, NN_params, chain, re
end


function nn_model_case2(test:: String, n_, activation_function)
    """
    NN architectures for experiment 2.
    """

    if test == "test/"

        chain_chiphi = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1])],
            Dense(11, n_, activation_function),
            Dense(n_, 2)
        )
        chain_pe = Chain(
            x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4], sqrt(abs(x[4]))],
            Dense(11, n_, activation_function),
            Dense(n_, 2)
        )

    # original from article
    elseif test == "test_1/"

        chain_chiphi = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2],
            Dense(9, 32, tanh),
            Dense(32, 2)
            )

        chain_pe = Chain(
            x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4]],
            Dense(10, 32, tanh),
            Dense(32, 2)
            )

    elseif test == "test_1_cos/"

    chain_chiphi = Chain(
        x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2],
        x -> convert.(Float32, x),
        Dense(9, 32, cos),
        Dense(32, 2)
        )

    chain_pe = Chain(
        x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4]],
        x -> convert.(Float32, x),
        Dense(10, 32, cos),
        Dense(32, 2)
        )
            
    # elseif test == "test_dropout/"

    #     chain_chiphi = Chain(
    #         x -> [x[5]*cos(x[1]),x[5]*1/abs(x[3]),x[5]*1/sqrt(abs(x[3])),x[5]*sqrt(abs(x[3])),x[5]*x[3],x[5]*sqrt(abs(x[3]))^3,x[5]*x[3]^2,x[5]*x[4],x[5]*x[4]^2, x[5]*sin(x[1]), x[5]*sin(x[1])*cos(x[1]), x[5]*tan(x[1]), x[5], 1/x[6], x[5]*x[6], x[5]^2],
    #         Parallel(
    #             vcat,
    #             Chain(Dense(16, 32, cos), Dense(32, 16, sin)),
    #             Chain(Dense(16, 32, sin), Dense(32, 16, cos))
    #         ),
    #         Dense(32,32, tanh),
    #         Dropout(0.7),
    #         Dense(32, 2)
    #     )
    #     chain_pe = Chain(
    #         x -> [x[5]*1/sqrt(abs(x[3]))^3,x[5]*1/abs(x[3]),x[5]*1/sqrt(abs(x[3])),x[5]*sqrt(abs(x[3])),x[5]*x[3],x[5]*sqrt(abs(x[3]))^3,x[5]*x[3]^2,x[5]*x[4],x[5]*x[4]^2,x[5]*x[3]*x[4], x[5]*sqrt(abs(x[4])), x[5]*tan(x[4]), x[5], 1/x[6], x[5]*x[6], x[5]^2],
    #         Parallel(
    #             vcat,
    #             Chain(Dense(16, 32, cos), Dense(32, 16, sin)),
    #             Chain(Dense(16, 32, sin), Dense(32, 16, cos))
    #         ),
    #         Dense(32,32, tanh),
    #         Dropout(0.7),
    #         Dense(32, 2)
    #     )
    
    elseif test == "test_dropout/"

        # n_ acts as alpha input

        chain_chiphi = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1]), tan(x[1])],
            Dense(12, 32, tanh),
            Dense(32, 32, tanh),
            Dropout(n_),
            Dense(32, 2)
        )
        chain_pe = Chain(
            x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4], sqrt(abs(x[4])), tan(x[4])],
            Dense(12, 32, tanh),
            Dense(32, 32, tanh),
            Dropout(n_),
            Dense(32, 2)
        )
    
    elseif test == "test_layernorm/"

        chain_chiphi = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1]), tan(x[1])],
            Dense(12, 32, tanh),
            Dense(32, 32, tanh),
            LayerNorm(32),
            Dense(32, 2)
        )
        chain_pe = Chain(
            x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4], sqrt(abs(x[4])), tan(x[4])],
            Dense(12, 32, tanh),
            Dense(32, 32, tanh),
            LayerNorm(32),
            Dense(32, 2)
        )


    elseif test == "test_layernorm_2/"

        chain_chiphi = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1]), tan(x[1])],
            Dense(12, 32, tanh),
            LayerNorm(32),
            Dense(32, 32, tanh),
            Dense(32, 2)
        )
        chain_pe = Chain(
            x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4], sqrt(abs(x[4])), tan(x[4])],
            Dense(12, 32, tanh),
            LayerNorm(32),
            Dense(32, 32, tanh),
            Dense(32, 2)
        )


    # elseif test == "rnn/"
    #     chain_chiphi = Chain(
    #         x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1]), tan(x[1])],
    #         x -> convert.(Float32, x),
    #         LSTM(12,6),
    #         Dense(6,2))
    #     chain_pe = Chain(
    #         x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4], sqrt(abs(x[4])), tan(x[4])],
    #         x -> convert.(Float32, x),
    #         LSTM(12,6),
    #         Dense(6,2))

    # RNN 1
    elseif test == "architecture_lstm_1/"

        chain_chiphi = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1])],
            x -> convert.(Float32, x),
            LSTM(11,2)
        )
        chain_pe = Chain(
            x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4], sqrt(abs(x[4]))],
            x -> convert.(Float32, x),
            LSTM(11,2)
        )

    elseif test == "architecture_lstm_2/"

        chain_chiphi = Chain(
            x -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, sin(x[1]), sin(x[1])*cos(x[1])],
            x -> convert.(Float32, x),
            LSTM(11,16),
            Dense(16, 32, tanh),
            Dense(32, 2)
        )
        chain_pe = Chain(
            x -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4], sqrt(abs(x[4]))],
            x -> convert.(Float32, x),
            LSTM(11,16),
            Dense(16, 32, tanh),
            Dense(32, 2)
        )

    else
        @error "Architecture non correctly specified"
    end
    
    NN_chiphi_params, re_chiphi = Flux.destructure(chain_chiphi)
    NN_chiphi_params = NN_chiphi_params .* 0
    NN_chiphi(u, NN_chiphi_params) = re_chiphi(NN_chiphi_params)(u)
    NN_pe_params, re_pe = Flux.destructure(chain_pe)
    NN_pe_params = NN_pe_params .* 0
    NN_pe(u, NN_pe_params) = re_pe(NN_pe_params)(u)

    NN_params = vcat(NN_chiphi_params,NN_pe_params)

    return NN_params, NN_chiphi, NN_chiphi_params, NN_pe, NN_pe_params, chain_chiphi, chain_pe, re_chiphi, re_pe
end