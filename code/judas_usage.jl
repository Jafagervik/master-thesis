using Judas
using Dates

function load_data(tp::TimePeriod=Minute(1); step::Int=6 * 2 * 2)
    f = "/media/jorgenaf/BaneNOR-DAS/"
    day = DateTime(2021, 8, 31, 10, 0, 5)
    duration = Second(tp)

    filepaths, ch_idx, samples = find_DAS_files(f, day, duration; step=step)

    s = load_DAS_files(filepaths, ch_idx, samples; sensitivity_select=-1)

    @show times(s)

    return nothing
end

function process(data)
    m = Minute(1)
    hz = freq(data)
    data = combine_matrices(m)

    new_hz = 100.0f0
    rate = new_hz / hz
    resampled = resample_das_signal(data, rate)

    filtered = window_and_filter(resampled)
end

function test(; tp=Minute(5))
    step = 6 * 2 * 2 * 2
    dasdata = load_data(tp; step=step)

    @show channel_distance(dasdata)
    @show size(dasdata)
    @show freq(dasdata)
    @show dasdata

    processed = process(dasdata)
    normalized = normalize(processed)

    before = das_heatmap(processed)
    after = das_heatmap(normalized)

    display(before)
    display(after)
    
    @info "Now ready for ai training"

    return processed
end

load_data()