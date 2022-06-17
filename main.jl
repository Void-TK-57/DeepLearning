using GLM
using ROC

include("model.jl")
include("data.jl")

function main()
    data = convert.(Float32, load_signals())
    matrix = Matrix(data)[:,1:end-1]
    matrix = convert(Matrix, transpose(matrix))
    @show size(matrix)

    regression = Flux.Chain(Flux.Dense(size(matrix, 1), 20, tanh), Flux.Dense(20, 10, tanh), Flux.Dense(10, 1, σ))
    history = train!(regression, (matrix, data[:, :target]), 500, Flux.Optimise.ADAM(0.001), Flux.Losses.binarycrossentropy)
    display(plot(1:500, history, title="Loss", lw=1, xlabel="epoch", ylabel="loss"))

    r = roc(regression(matrix), data[:, :target])
    @show AUC(r)
    display(plot(r))

    regression = conv_autoencoder()
    history = train!(regression, (reshape(matrix, 512, 1, 1, 174), data[:, :target]), 500, Flux.Optimise.ADAM(0.001), Flux.Losses.binarycrossentropy)
    display(plot(1:500, history, title="Loss", lw=1, xlabel="epoch", ylabel="loss"))

    r = roc(regression(reshape(matrix, 512, 1, 1, 174)), data[:, :target])
    @show AUC(r)
    display(plot(r))

    # load architecture
    model = autoencoder(512, [256=>tanh, 64=>tanh, 32=>tanh, 64=>tanh, 256=>tanh, 512=>tanh])

    opt = Flux.Optimise.ADAM(0.001)
    epochs = 1000

    history = train!(model, (matrix, matrix), epochs, opt, Flux.Losses.mse)
    display(plot(1:epochs, history, title="Loss", lw=1, xlabel="epoch", ylabel="loss"))

    display_signal(matrix[:, 5])
    display_signal(model(matrix[:, 5]))
    display_signal(model[1:3](matrix[:, 5]))

    display_signal(matrix[:, 150])
    display_signal(model(matrix[:, 150]))
    display_signal(model[1:3](matrix[:, 150]))

    # create decoder and add it on top of the encoder
    total_model = Flux.Chain(model[1:3]..., Flux.Dense(32, 20, tanh), Flux.Dense(20, 10, tanh), Flux.Dense(10, 1, σ))
    history = train!(total_model, (matrix, data[:, :target]), 500, Flux.Optimise.ADAM(0.001), Flux.Losses.binarycrossentropy, freeze=4:6)
    display(plot(1:500, history, title="Loss", lw=1, xlabel="epoch", ylabel="loss"))

    r = roc(total_model(matrix), data[:, :target])
    @show AUC(r)
    display(plot(r))

    return total_model

end
