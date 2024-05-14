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
    res = custom_act_function_coef[1]*tanh(x) + custom_act_function_coef[2]*sigmoid(x) + custom_act_function_coef[3]*cos(x) + custom_act_function_coef[4]*softmodulusQ(x)
end


function nn_model_case2(test:: String, n_ = 32, activation_function = tanh)
    """
    NN architectures for experiment 2 (non-EMR)
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
    
    elseif test == "test_dropout/"

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

    # process parameters
    
    NN_chiphi_params, re_chiphi = Flux.destructure(chain_chiphi)
    NN_chiphi_params = NN_chiphi_params .* 0
    NN_chiphi(u, NN_chiphi_params) = re_chiphi(NN_chiphi_params)(u)
    NN_pe_params, re_pe = Flux.destructure(chain_pe)
    NN_pe_params = NN_pe_params .* 0
    NN_pe(u, NN_pe_params) = re_pe(NN_pe_params)(u)

    NN_params = vcat(NN_chiphi_params,NN_pe_params)

    return NN_params, NN_chiphi, NN_chiphi_params, NN_pe, NN_pe_params, chain_chiphi, chain_pe, re_chiphi, re_pe
end