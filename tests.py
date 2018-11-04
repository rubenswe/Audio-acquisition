"""
Tests for finding the delay of several recordings. Find average and standard deviation.
"""

from audio_acquisition import *
from audio_utilities import *
import soundfile as sf 
import timeit


def filename_generator(filename, N):
    filenames = []
    for i in range(N):
        name = filename + str(i) + ".wav"
        filenames.append(name)
    return filenames


def playrec_repeated(input_file, N, t):
    # Remember that input filename cant be empty => Only change output file to look at delay?
    out_files = filename_generator("test", 10)
    in_data, fs = sf.read(input_file)

    for i in range(N):
        playrec(input_file, out_files[i])  # Should create 10 output files. use get_delay to find all delays
        # Should playrec() return the data? So it will be available
        #get_delay()...Can also do get_delay() in a separate function? After reading the test files generated above.

    delays = []

    for j in range(N):
        out_data, fs = sf.read(out_files[j])
        delays.append(get_delay(in_data, out_data, fs, t))
    return delays

# Result of pleyrec_repeated():
delays = [0.12941043083900228, 0.1293424036281179, 0.12941043083900228, 0.1291609977324263, 0.12968253968253968,
         0.12931972789115645, 0.12954648526077098, 0.129297052154195, 0.12941043083900228, 0.1294331065759637]

print(np.std(delays), np.mean(delays), np.median(delays))
# Results:
# 0.00013422810509703875 0.1294013605442177 0.12941043083900228

#start_time = timeit.default_timer()
#print(playrec_repeated("sweep_file_20_20k.wav", 10, 1))
#print(timeit.default_timer() - start_time)