using Judas
using Dates

f = "/path/to/das/data/"
day = DateTime(2021, 8, 31, 10, 0, 5)
duration = Hour(1)

# Step size between sensors
step = 48

#  Find DAS Data
filepaths, ch_idx, samples = find_DAS_files(f, day, duration; step=step)

#  Load DAS data
das_signal = load_DAS_files(filepaths, ch_idx, samples; sensitivity_select=-1)

#  Process Data
data = combine_matrices(duraration)

new_hz = 100.0f0
rate = new_hz / freq(das_signal)

resampled = parallel_resample(data, rate)

filtered = denoise(resampled, new_hz)