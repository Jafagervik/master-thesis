include("ae.jl")

M, N  = 100000, 2000
data = ones(Float32, M, N)

# Training
args = AEArgs(; M, N, epochs=200, device=gpu)
losses = train(args, data)

# Inference
inference_data = ones(Float32, M, N)
anomalies = infer(inference_data)

plot_anomalies(anomalies)
scores = roc_scores(anomalies)