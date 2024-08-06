using Judas
using Dates
using ArgParse
using BenchmarkTools

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--duration", "-d"
        help = "Duration in minutes"
        arg_type = Int
        default = 1
        "--warmup", "-w"
        help = "Perform warmup run"
        action = :store_true
    end
    return parse_args(s)
end

function load_data(tp::TimePeriod=Minute(1); step::Int=6 * 2 * 2)
    f = "/media/jorgenaf/BaneNOR-DAS/"
    day = DateTime(2021, 8, 31, 10, 0, 5)
    duration = Second(tp)

    filepaths, ch_idx, samples = find_DAS_files(f, day, duration; step=step)
    s = load_DAS_files(filepaths, ch_idx, samples; sensitivity_select=-1)
end

function process(data)
    m = Minute(1)
    hz = freq(data)
    data = combine_matrices(m)
    new_hz = 100.0f0
    rate = new_hz / hz
    resampled = thread_resample(data, rate)

    return denoise(resampled)
end

function test(; tp=Minute(5))
    step = 6 * 2 * 2 * 2

    dasdata = load_data(tp; step=step)
    return process(dasdata)
end

function main()
    args = parse_commandline()
    duration = args["duration"]
    is_warmup = args["warmup"]

    if is_warmup
        println("Performing warmup run with duration: $(duration) minutes")
        test(; tp=Minute(duration))
        println("Warmup completed")
    else
        println("Running benchmark with duration: $(duration) minutes")
        result = @benchmark test(; tp=Minute($duration))
        println("Benchmark completed")
        println("Median time: $(median(result.times)) ns")
        println("Memory: $(result.memory) bytes")
        println("Allocations: $(result.allocs)")
    end
end
main()
