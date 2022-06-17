using DataFrames
using CSV
using Plots

function display_image(data)
    display(heatmap(1:size(data, 1), 1:size(data, 2), data))
end

function display_signal(data)
    display(plot(1:size(data, 1), data, title="Signal"))
end

function read_file(path::String)
    # open file
    file = open(path)

    # vector of lines
    content = Vector{String}()

    # read conent in the file
    for line in readlines(file)
        push!(content, line)
    end

    # return the vector with the content
    return content
end

function search_signals!(data::DataFrame, path::String, target::Bool=true)
    # check if the path is a .txt file
    if isfile(path) && occursin(r"^(.+\.txt)$", path)
        # load path data
        row = parse.(Float64, read_file(path))
        push!(row, target)
        push!(data, row)
    elseif isdir(path)
        # else, call recursively search txt for each subpath
        for subpath in readdir(path; join=true)
            search_signals!(data, subpath, target)
        end
    end
end

function load_signals(path::String = "/home/void/Data/datasets")
    signals = DataFrame()
    for i in 1:512
        signals[:, "X"*string(i)] = Float64[]
    end
    signals[:,"target"] = Bool[]

    search_signals!(signals, path*"/datanormal/", false)
    search_signals!(signals, path*"/fault/", true)

    return signals
end

function load_mnist(path::String="/home/void/Home/Data/MNIST_digits/")
    train_data = CSV.read(path*"mnist_train.csv", DataFrame)
    train_data = permutedims(Matrix{Float32}( train_data ))
    raw = reshape(train_data[2:end, :], 28, 28, 1, :)
    raw = permutedims(raw, (2, 1, 3, 4))
    #train_X = zeros(Float32, 32, 32, 1, size(raw, 4))
    #train_X[3:30, 3:30, :, :] = raw
    train_X = raw
    train_y = train_data[1, :]

    test_data = CSV.read(path*"mnist_test.csv", DataFrame)
    test_data = permutedims(Matrix{Float32}( test_data ))
    raw = reshape(test_data[2:end, :], 28, 28, 1, :)
    raw = permutedims(raw, (2, 1, 3, 4))
    #test_X = zeros(Float32, 32, 32, 1, size(raw, 4))
    #test_X[3:30, 3:30, :, :] = raw
    test_X = raw
    test_y = test_data[1, :]

    return ((train_X, Flux.onehotbatch(train_y, 0.0:1:9.0)), (test_X, Flux.onehotbatch(test_y, 0.0:1:9.0)))
end
