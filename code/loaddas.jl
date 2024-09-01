function load_DAS_files(
    filepaths::Vector{String},
    ch_index::StepRange,
    samples::UnitRange{Int};
    sensitivity_select::Integer=0,
    integrate::Bool=true,
    unwrap::Bool=false,
    spike_thr=nothing,
    user_sensitivity=nothing,
)
    isempty(filepaths) && throw(ArgumentError("No files to load"))
    load_start = now(UTC)

    isdir(TEMPDIR()) || mkdir(TEMPDIR())
    filepaths isa String && (filepaths = [filepaths])
    samples === nothing && (samples = Base.Slice(nothing))

    m = Dict()

    # FIRST FILE 
    df = DictFile(filepaths[begin])
    n_channels, n_samples = size(df.file["data"])

    m = load_dict(df; skip_fields=["data"])
    _fix_meta!(m, n_samples, n_channels)

    if length(m["header"]["channels"][begin][begin:(ch_index.step):end]) == 0 
        throw(KeyError("chIndexes $ch_index not in data (nChannels = $n_channels)"))
    end

    channels_max = length(ch_index)

    sensitivity_unit::String = ""
    unit_out::String = ""
    sensitivity_out::String = ""
    sensitivity_units_out = nothing

    if sensitivity_select >= 0
        try
            sensitivity = m["header"]["sensitivities"][sensitivity_select + 1][]
            sensitivity_unit = m["header"]["sensitivityUnits"][sensitivity_select + 1][]
        catch err
            if err isa KeyError
                sensitivity = m["header"]["sensitivities"]
                sensitivity_unit = m["header"]["sensitivityUnits"]
            else
                @warn "sensitivity_select index $sensitivity_select not found in file."
                throw(CanonicalIndexError)
            end
        end
        unit_out = _combine_units([m["header"]["unit"], sensitivity_unit]; operator="/")
        sensitivities_out = ones(Float32, 1, 1)
        sensitivities_units_out = [""]

    elseif sensitivity_select == -1
        unit_out = m["header"]["unit"] # rad / m / s
        sensitivity = 9362208.90109403f0 # Necessary
        sensitivities_out = m["header"]["sensitivities"]
        sensitivities_units_out = m["header"]["sensitivityUnits"]

    elseif sensitivity_select == -2
        unit_out = _combine_units([m["header"]["unit"], "m"])
        sensitivity = @fastmath 1.0 / m["header"]["gaugeLength"]
        sensitivity_out = @fastmath m["header"]["sensitivities"] *
            m["header"]["gaugeLength"]
        sensitivity_units_out = [
            _combine_units(
                [sensitivityUnit, "m"] for
                sensitivityUnit in m["header"]["sensitivitiUnits"]
            )
        ]

    elseif sensitivity_select == -3 && user_sensitivity === nothing
        unit_out = _combine_units(
            [m["header"]["unit"], user_sensitivity["sensitivityUnit"]]; operator="/"
        )
        sensitivity = user_sensitivity["sensitivity"]
        sensitivity_out = ones(Float32, 1, 1)
        sensitivity_units_out = [""]
    else
        error("Undefined sensitivity select. Must be in range [-3, 0]")
    end

    scale::Float32 = m["header"]["dataScale"] / sensitivity

    # Write first file serially
    loc_data_path = get_datapath(1)
    initial_data = transpose(df.file["data"][ch_index, :]) * scale
    write_matrix(initial_data, loc_data_path)

    if length(filepaths) > 1 && nworkers() > 1
        chunks = begin
            tot_size = length(filepaths[2:end])
            chunk_size = tot_size ÷ nworkers()
            carry = tot_size % nworkers()
            chunks = Vector{UnitRange{Int64}}(undef, nworkers())
            current_index = 2
            for i in workers()
                extra = i - 1 <= carry ? 1 : 0
                elements_per_thread = chunk_size + extra
                end_index = current_index + elements_per_thread - 1
                @inbounds chunks[i - 1] = current_index:end_index
                current_index = end_index + 1
            end
            chunks
        end
        @sync begin
            for (i, (pid, chunk)) in enumerate(zip(workers(), chunks))
                @async begin
                    try
                        remotecall_fetch(
                            process_DAS_chunk,
                            pid,
                            n_samples,
                            ch_index,
                            filepaths,
                            chunk,
                            scale
                        )
                        println("Chunk $i completed on worker $pid")
                    catch e
                        println("Error processing chunk $i on worker $pid: $e")
                    end
                end
            end
        end

    elseif length(filepaths) > 1
        loc_data = zeros(Float32, n_samples, channels_max)
        idx = 2
        for filepath in filepaths[2:end]
            df = DictFile(filepath)
            loc_data = transpose(df.file["data"][ch_index, :]) * scale

            write_matrix(loc_data, get_datapath(idx))
            idx += 1
        end
    end
    (unwrap || spike_thr !== nothing || integrate !== nothing) &&
        m["header"]["dataType"] < 3 &&
        error("Options unwrap, spikeThr or integrate can only 
                be used wih time differentiaded phase data.")
                
    if unwrap || (spike_thr !== nothing)
        signal_nd = get_matrix(data_path)
        if m["header"]["dataType"] !== nothing
            signal_nd = unwrap(
                signal_nd, m["header"]["spatialUnwrRange"] * sensitivity; axis=2
            )
        end
        if m["header"]["spatialUnwrRange"] !== nothing
            signal_nd = unwrap(
                signal_nd, m["header"]["spatialUnwrRange"] * sensitivity; axis=2
            )
        end
        if spike_thr !== nothing
            signal_nd[abs(signal_nd) > spike_thr * sensitivity] = zero(eltype(signal_nd))
        end
    end
    if integrate
        unit_new = _combine_units([unit_out, "s"])
        if !any(u -> u == 's', split(replace(unit_new, r"[\w']+" => "")))
            prev = undef_init(channels_max)
            for i in ProgressBar(eachindex(filepaths))
                ccds!(prev, Float32(m["header"]["dt"]), i)
            end
            unit_out = unit_new
        else
            @warn "Data unit $unit_out is not differentiable. Integration skipped."
        end
    end

    tstart = unix2datetime(m["header"]["time"] + samples.start * m["header"]["dt"])
    meta::Dict{String,Any} = Dict(
        key => m["header"][key] for key in [
            "dt",
            "dx",
            "gaugeLength",
            "experiment",
            "dataType",
            "dimensionRanges",
            "dimensionUnits",
            "dimensionNames",
            "name"
        ]
    )
    meta["fileVersion"] = m["fileVersion"]
    meta["time"] = tstart
    meta["unit"] = unit_out
    meta["sensitivity"] = sensitivity_out
    meta["sensitivityUnits"] = sensitivity_units_out
    meta["filepaths"] = filepaths
    meta["roiDec"] = round(
        m["header"]["dx"] * hex_to_int(m["demodSpec"]["roiDec"][1])[1] * ch_index.step;
        digits=4
    )
    signal = DASDataFrame(TEMPDIR(), n_samples, length(filepaths), tstart, ch_index, meta)
    println("Loaded $nfiles in $(timer(load_start))")
    return signal
end