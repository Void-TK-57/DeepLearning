using Flux

@inline loss(m, x, y) = Flux.Losses.mse(m(x), y)

function train!(model, data, epochs, opt, loss; freeze=false)::Vector{Float64}
    # training history
    history::Vector{Float64} = []

    # loss function
    @inline loss_(x,y) = loss(model(x),y)

    # layers to be optimzed
    ps = Flux.params(model)
    if freeze != false
        ps = ps[freeze]
    end

    # training loop
    for i in 1:epochs
        # train for 1 epoch
        Flux.train!(loss_, ps, [data], opt)
        # calculate epoch loss and push it to the vector
        epoch_loss = loss_(data...)
        push!(history, epoch_loss)
        # log it
        @info "Epoch $i/$epochs, Loss: $epoch_loss"
    end

    return history
end

@inline logistic_regression(input, output) = Flux.Dense(input, output, σ)


function autoencoder(input, layers::Vector)
    previous_output_size = input
    architecture = []

    for (output_size, σ) in layers
        push!(architecture, Dense(previous_output_size, output_size, σ))
        previous_output_size = output_size
    end

    return Flux.Chain(architecture...)
end

function conv_autoencoder()
    return Flux.Chain(
        Flux.Conv((40, 1), 1=>5, stride=2),
        Flux.MeanPool((20, 1), stride=3),
        Flux.Conv((20,1), 5=>15,stride=1),
        Flux.MeanPool((15,1),stride=2),
        Flux.Conv((10,1), 15=>60,stride=1),
        Flux.MeanPool((8,1),stride=4),
        Flux.flatten,
        Flux.Dense(60, 30, tanh),
        Flux.Dense(30, 1, σ)
    )
end

function LeNet5()
    model = Flux.Chain(
        Flux.Conv((5, 5), 1=>6, tanh, stride=1, pad=2),
        Flux.MeanPool((2, 2), stride=2),
        Flux.Conv((5, 5), 6=>16, tanh, stride=1),
        Flux.MeanPool((2, 2), stride=2),
        Flux.Conv((5, 5), 16=>120, tanh, stride=1),
        Flux.flatten,
        Flux.Dense(120, 84, tanh),
        Flux.Dense(84, 10, σ)
    )
    return model
end
