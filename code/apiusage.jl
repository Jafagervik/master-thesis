using Judas
using Dates

folder = "path/to/folder"
day = DateTime(2021, 8, 31, 0, 0, 5)
duration = Second(3600 * 5)

# Find HDF5 files within time range, and Tim
files, ch_index, samples = find_DAS_files(folder, day, duration)

sensors = load_DAS_files(files, ch_index, samples)

processed_sensors = window_and_filter(sensors)

train!(processed_sensors)

locs = analyze(processed_sensors)

plot(locs)

info(locs)