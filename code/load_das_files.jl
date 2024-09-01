function process_DAS_chunk(
    n_samples::Int,
    ch_index::StepRange{Int,Int},
    filepaths::Vector{String},
    chunk::UnitRange{Int},
    scale::Float32)
    n_channels = length(ch_index)
    for filepath in filepaths[chunk]
        idx = findfirst(x -> x == filepath, filepaths)
        loc_data_path = get_datapath(idx)

        h5open(filepath, "r") do file
            data = HDF5.readmmap(file["data"])
            processed_data = transpose(data[ch_index, :]) * scale

            open(loc_data_path, "w+") do f
                write(f, n_samples)
                write(f, n_channels)
                mmap_array = mmap(f, Matrix{Float32}, (n_samples, n_channels))
                mmap_array .= processed_data
                Mmap.sync!(mmap_array)
            end
        end
    end
end