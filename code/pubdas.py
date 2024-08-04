from scipy import signal
import numpy as np
from tdms_reader import TdmsReader
import h5py
import obspy
import os


def get_time(filename):
    # read timestamp from filenames of raw data
    # example of filename of a raw tdms file: PSUDAS_UTC_20200301_000615.729.tdms
    import obspy
    ftime = ''.join(filename.split('_')[2:])
    ftime1 = ftime.split('.')[0]
    ftime2 = float(ftime.split('.')[1])/1000  # ms
    time = obspy.UTCDateTime(ftime1)+ftime2
    return time

def kill_foresee(idas_array):
    # return a numpy array without bad traces
    import numpy
    bad_traces = list(range(0, 23)) \
                 + list(range(81, 88)) \
                 + list(range(144, 155)) \
                 + list(range(214, 220)) \
                 + list(range(365, 367)) \
                 + list(range(428, 435)) \
                 + list(range(611, 618)) \
                 + list(range(672, 683)) \
                 + list(range(1017, 1032)) \
                 + list(range(1050, 1073)) \
                 + list(range(1094, 1098)) \
                 + list(range(1130, 1134)) \
                 + list(range(1174, 1176)) \
                 + list(range(1219, 1222)) \
                 + list(range(1224, 1227)) \
                 + list(range(1234, 1239)) \
                 + list(range(1244, 1255)) \
                 + list(range(1264, 1272)) \
                 + list(range(1530, 1587)) \
                 + list(range(2084, 2096)) \
                 + list(range(2119, 2126)) \
                 + list(range(2365, idas_array.shape[0]))
    idas_array = np.delete(idas_array, bad_traces, 0)
    return idas_array

# file list of raw tdms files
datadir = '/mnt/DAS/mar20/'
filelist = os.listdir(datadir)
filelist.sort()
# parameter setting
outdir = '../pubdas/202003/'
nSamp_min = 500*60 # samples for 1 min data
win_len = 10 # 10 min 
i=0
# 
while i < len(filelist):
    stime = get_time(filelist[i])
    # determine if data is continous in 10 minutes. If yes, ifile = 10.
    for ifile in np.arange(win_len-1, 0, -1):
        etime = stime + 60*ifile
        last_file = 'PSUDAS_UTC_'+etime.strftime("%Y%m%d_%H%M%S.%f")[:-3]+'.tdms'
        if last_file in filelist:
            break
    # Read data from 10 tdms files and store in one numpy array
    ifile = ifile+1
    data=np.empty([2137,int(nSamp_min*ifile)])
    for j in range(ifile):
        inputFile=datadir+filelist[i+j]
        print(str(i+j)+': '+inputFile)
        tdms = TdmsReader(inputFile)
        props = tdms.get_properties()
        nCh = tdms.fileinfo['n_channels']
        data0 = np.transpose(tdms.get_data(0,nCh,0,tdms.channel_length-1))
        # kill bad trace
        data[:,nSamp_min*j:nSamp_min*(j+1)]=kill_foresee(data0)
    # downsample from 500 Hz to 125 Hz
    data = signal.decimate(data, 4, axis=1, ftype='fir')
    # prepare timestamp for hdf5 files 
    time = []
    for k in range((ifile)*60*125):
        time.append((stime+k/125).timestamp)
    # write to hdf5 files
    out = outdir + 'FORESEE_UTC_' + stime.strftime("%Y%m%d_%H%M%S")+'.hdf5'
    f = h5py.File(out, "w")
    f.create_dataset("raw", data=np.float16(data))
    f.create_dataset("timestamp", data=time)
    f.close()
    i+=ifile