using Flux, MLDatasets, CUDA, FileIO
using Flux: train!, onehotbatch

# Get data
x_train, y_train = MLDatasets.MNIST.traindata()
x_test, y_test = MLDatasets.MNIST.testdata()
x_train = Float32.(x_train)
y_train = onehotbatch(y_train, 0:9)

EPOCHS = 400
DESC = 1e-4

model = Chain(
    Dense(28*28, 256, relu),
    Dense(256, 10, relu), softmax
)

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
optimizer = ADAM(DESC)
parameters = params(model)

train_data = [(Flux.flatten(x_train), y_train)]
test_data = [(Flux.flatten(x_test), y_test)]

for i in 1:EPOCHS
    Flux.train!(loss, parameters, train_data, optimizer)
end

acc = let t = 0
  (i) -> begin
    @inbounds inc = findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
    return ifelse(inc, t+1, t)
  end
end

for i in 1:length(y_test) acc(i) end

println(accuracy / length(y_test))