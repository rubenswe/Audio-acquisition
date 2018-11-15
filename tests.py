"""
Tests for finding the delay of several recordings. Find average and standard deviation.
"""

from audio_acquisition import *
from audio_utilities import *
import soundfile as sf 
import timeit
from scipy.signal import chirp

fs = 44100
T = 20
t = np.linspace(0, T, T*fs)
volume = 0.5

w = (chirp(t, T, 20, 20000, method='linear', phi=90)*volume).astype(np.float32)

def playrec_repeated(input_file, N, t):
    out_files = filename_generator("test", 10)
    in_data, fs = sf.read(input_file)

    for i in range(N):
        playrec(input_file, out_files[i])  # Should create 10 output files. use get_delay to find all delays

    delays = []

    for j in range(N):
        out_data, fs = sf.read(out_files[j])
        delays.append(get_delay(in_data, out_data, fs, t))
    
    return delays


def playrec_from_data_repeated(in_data, N, t, fs):  # t is for get_delay()
    #data = []
    delays = []

    for i in range(N):  # for _ in range(N), (not using i)
        #data.append(playrec_from_data(w, 0, 0, 1))  # 1 channel, no extra recording time delta_t
        data = playrec_from_data(w, 0, 0, 1)  # Overwrites data with the next data
        delays.append(get_delay(in_data, data[:,0], fs, t))  # data[:,0] is the first column, i.e. channel 1(=> This is for mono recording)
        
    return delays

# Result of pleyrec_repeated():
#delays = [0.12941043083900228, 0.1293424036281179, 0.12941043083900228, 0.1291609977324263, 0.12968253968253968,
#         0.12931972789115645, 0.12954648526077098, 0.129297052154195, 0.12941043083900228, 0.1294331065759637]

#print(np.std(delays), np.mean(delays), np.median(delays))
# Results:
# 0.00013422810509703875 0.1294013605442177 0.12941043083900228

delays_from_data = [0.12936507936507938, 0.12931972789115645, 0.12931972789115645, 0.1293424036281179, 0.1293877551020408,
                    0.12936507936507938, 0.12931972789115645, 0.12936507936507938, 0.1292063492063492, 0.1295238095238095]

print(np.std(delays_from_data), np.mean(delays_from_data), np.median(delays_from_data))

#start_time = timeit.default_timer()
#print(playrec_repeated("sweep_file_20_20k.wav", 10, 1))
print(playrec_from_data_repeated(w, 10, 1, 44100))
#print(timeit.default_timer() - start_time)

