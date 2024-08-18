using Judas, Dates

f = "/path/to/das/data/"
day = DateTime(2021, 8, 31, 10, 0, 5)
duration = Hour(1)

#  Find and Load DAS Data
filepaths, ch_idx, samples = find_DAS_files(f, day, duration; step=48)
das_signal = load_DAS_files(filepaths, ch_idx, samples; sensitivity_select=-1)

# Plot Heatmap of DAS data
das_heatmap(das_signal; name="Experiment", save=True)

#  Process Data
das_data = data(das_signal)
new_hz = 100.0f0

resampled = das_resample(das_data, new_hz/freq(das_signal))
filtered = denoise(resampled, new_hz)
