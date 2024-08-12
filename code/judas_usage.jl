using Judas
using Dates

f = "/media/jorgenaf/BaneNOR-DAS/"
day = DateTime(2021, 8, 31, 10, 0, 5)
duration = Minute(5)

# We will only be looking at every 48 * 4 channel of the ones in ROI
step = 6 * 2 * 2 * 2

#  Find DAS Data
filepaths, ch_idx, samples = find_DAS_files(f, day, duration; step=step)

#  Load DAS data
das_signal = load_DAS_files(filepaths, ch_idx, samples; sensitivity_select=-1)

#  Process Data
data = combine_matrices(duraration)

new_hz = 100.0f0
rate = new_hz / freq(das_signal)

resampled = parallel_resample(data, rate)

filtered = denoise(resampled)