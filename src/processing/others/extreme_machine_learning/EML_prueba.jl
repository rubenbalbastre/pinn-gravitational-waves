using EML;

nhidden = 100
x = [1.0 1.0; 0.0 1.0; 0.0 0.0; 1.0 0.0]
y = [0.0 1.0; 1.0 0.0; 0.0 0.0; 1.0 0.0]
elm = ELM(nhidden, x, y, sigmoid)
fit!(elm,x,y)
y_pred = predict(elm, new_x)

function fit_elm(elm, x, y)
    """
    Update weights
    """

    hidden_matrix = elm.activation.(elm.weight_matrix*x .+ elm.bias_vector)
    elm.output_weights .= hidden_matrix\y

end

nepochs = 100;
for epoch in range(1, nepochs)
    fit_elm(elm, x, y)
end