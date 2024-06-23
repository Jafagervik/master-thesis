function parallel_resample(signal::Matrix{T}, rate::Real)::Matrix{T}
    rows, cols = size(signal)

    new_rows = Int(fld(rows * rate, 1))

    result = Matrix{T}(undef, new_rows, cols)
    # Apply resample_single to each slice of x in parallel
    slices = pmap(x_slices -> T.(resample(x_slices, rate)), eachcol(signal))

    for (i, slice) in enumerate(slices)
        @inbounds result[:, i] = slice
    end

    return result
end
