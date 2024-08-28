function ccds!(prev::Vector{Float32}, Δt::Float32, i::Int)
    file_path = get_datapath(i)
    open(file_path, "r+") do f
        n_samples, n_channels = read(f, Int), read(f, Int)
        signals = mmap(f, Matrix{Float32}, (n_samples, n_channels))

        i != 1 && (signals[begin, :] .+= prev)

        for col in 1:n_channels, row in 2:n_samples
            @inbounds signals[row, col] += signals[row - 1, col]
        end
        prev .= signals[end, :]
        signals .*= Δt
        sync!(signals)
    end
end