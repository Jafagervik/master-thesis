serial_resample(data::Matrix{Float32}, rate::Real) = Float32.(resample(data, rate; dims=1))