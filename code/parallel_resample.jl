function das_resample(signal::Matrix{Float32}, rate::Real)
    rows, cols = size(signal)
    new_rows = Int(fld(rows * rate, 1))
    result = SharedMatrix{Float32}(new_rows, cols)

    @sync @distributed for col in 1:cols
        result[:, col] = Float32.(resample(signal[:, col], rate))
    end

    return sdata(result)
end
