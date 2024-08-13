function parallel_resample(signal::Matrix{Float32}, rate::Real)::Matrix{Float32}
    rows, cols = size(signal)

    result = Matrix{Float32}(undef, Int(fld(rows * rate, 1)), cols)
    slices = pmap(col -> Float32.(resample(col, rate)), eachcol(signal))

    @inbounds @simd for i in eachindex(slices)
        result[:, i] = slices[i]
    end

    return result
end
