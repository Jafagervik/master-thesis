function denoise(
    data::Matrix{Float32};
    tpr_rate::Float32=0.05f0,
    taper::Bool=true,
    bp_order::Int=4,
    lp_order::Int=4,
    hp_order::Int=4,
    cut_freq::Tuple{Int,Int}=(1, 100),
    Fs::Integer=1000,
    filter_opt::Filter=BP,
)
    taper || return nothing
    nsamples, cols = size(data)
    cols >= 0 || @error "You need at least one channel"
    
    cosTpr = tukey(nsamples, tpr_rate)

    if cols > 1
        dataTpr = data .* (cosTpr .* ones(Float32, 1, cols))
    else 
        dataTpr = data .* cosTpr
    end

    get_cutoff_freq = x -> x / (Fs / 2)
    cf_low, cf_high = get_cutoff_freq.(cut_freq)

    filter_opt == BP &&
        return filtfilt(
            digitalfilter(Bandpass(cf_low, cf_high; fs=Fs), Butterworth(bp_order)),
            dataTpr
        )
    filter_opt == LP && 
        return filtfilt(
            digitalfilter(Lowpass(cf_low; fs=Fs), Butterworth(lp_order)), dataTpr
        )
    filter_opt == HP &&
        return filtfilt(
            digitalfilter(Highpass(cf_high; fs=Fs), Butterworth(hp_order)), dataTpr
        )
    end
end