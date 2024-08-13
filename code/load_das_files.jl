function process_DAS_chunk(
    n_samples::Int,
    ch_index::StepRange{Int,Int},
    filepaths::Vector{String},
    chunk::UnitRange{Int},
    scale::Float32
)
    n_channels = length(ch_index)
    loc_data = zeros(Float32, n_samples, n_channels)

    for filepath in filepaths[chunk]
        idx = findfirst(x -> x == filepath, filepaths)

        loc_data_path = get_datapath(idx)
        loc_data = transpose(h5read(filepath, "data")[ch_index, :]) * scale

        # Write directly to file instead of writing to array first
        open(loc_data_path, "w") do f
            write(f, n_samples)
            write(f, n_channels)
            write(f, loc_data)
        end
    end
end

@sync begin
    for (pid, chunk) in zip(workers(), chunks)
        @async begin
            remotecall_fetch(
                process_DAS_chunk, pid, n_samples, ch_index, filepaths, chunk, scale
            )
        end
    end
end
