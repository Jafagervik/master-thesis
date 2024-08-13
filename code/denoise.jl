function denoise(
    data::Matrix{Float32},
    hz::Integer;
    tpr_rate::Float32=0.05f0,
    taper::Bool=true,
    bp_order::Int=4,
    lp_order::Int=4,
    hp_order::Int=4,
    cut_freq::Tuple{Int,Int}=(10, 800),
    filter_opt::Filter=BP,
)
    taper || return data
    nsamples, cols = size(data)
    cols >= 0 || error("You need at least one channel")
    
    cosTpr = tukey(nsamples, tpr_rate)
    dataTpr = cols > 1 ? data .* (cosTpr .* ones(Float32, 1, cols)) : data .* cosTpr
    
    cf_low, cf_high = cut_freq ./ (hz / 2)
    
    filter_params = Dict(
        BP => (Bandpass(cf_low, cf_high; fs=hz), bp_order),
        LP => (Lowpass(cf_low; fs=hz), lp_order),
        HP => (Highpass(cf_high; fs=hz), hp_order)
    )
    
    filter_type, order = get(filter_params, filter_opt) do
        error("Invalid filter option")
    end
    
    return filtfilt(digitalfilter(filter_type, Butterworth(order)), dataTpr)
end