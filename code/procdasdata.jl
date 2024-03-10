function process_DAS_data(
    data::Matrix{Float32};
    tpr_rate::Float32=0.05f0,
    taper::Bool=true,
    bp_order::Int=4,
    lp_order::Int=4,
    hp_order::Int=4,
    cut_freq::Tuple{Int,Int}=(1, 100),
    Fs::Integer=1000,
    filter_opt::Int=BP,
)
    if taper
        nsamples, cols = size(data)
        cosTpr = tukey(nsamples, tpr_rate)

        if cols > 1
            dataTpr = data .* (cosTpr .* ones(T, 1, cols))
        elseif cols == 1
            dataTpr = data .* cosTpr
        else
            @error "You need at least one channel"
        end

        get_cutoff_freq = x -> x / (Fs / 2)
        cf_low, cf_high = get_cutoff_freq.(cut_freq)

        if (filter_opt == BP)
            data_tpr = filtfilt(
                digitalfilter(Bandpass(cf_low, cf_high; fs=Fs), Butterworth(bp_order)),
                dataTpr
            )
        elseif (filter_opt == LP)
            data_tpr = filtfilt(
                digitalfilter(Lowpass(cf_low; fs=Fs), Butterworth(lp_order)), dataTpr
            )
        elseif (filter_opt == HP)
            data_tpr = filtfilt(
                digitalfilter(Highpass(cf_high; fs=Fs), Butterworth(hp_order)), dataTpr
            )
        end

        return data_tpr
    end

    return nothing
end