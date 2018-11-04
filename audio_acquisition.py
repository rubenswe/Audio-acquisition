import soundfile as sf 
import pyaudio as pa 
import numpy as np 
import time

"""
    Functions for playing and recording wave files.
"""

def playrec(in_file, out_file):  # take in numpy arrays of soundfile objects as parameters?
    """
    Function for playing a test sound(mono) and simultaneously recording(mono/stereo) the response(wire_callback.py) 
    """
    # Duration given by in_file, repeats only in test case? --> tests.py
    # Should return data of record file?

    input_file = sf.SoundFile(in_file, "rb")  # file to be played
    output_file = sf.SoundFile(out_file, "wb", input_file.samplerate, input_file.channels, input_file.subtype)  # File with the recorded data

    p = pa.PyAudio()

    # in_data is the recorded data. callback() is called in a separate thread
    def callback(in_data, frame_count, time_info, status):
        in_data = np.fromstring(in_data, dtype=np.float32).reshape(frame_count, input_file.channels)  # Convert the data before writing
        out_data = input_file.read(frame_count, dtype=np.float32)  # Read in the data to be played(and then recorded into in_data)
        output_file.write(in_data)  # Write the recorded data to the output file record_file.wav
        return (out_data, pa.paContinue)  # out_data is the played data

    stream = p.open(format=pa.paFloat32,
                channels=input_file.channels,  # What to do in the case of 1 channel for playing and 2 for recording? Use 2 channels always?(leaving one channel empty in the mono case if possible)
                rate=input_file.samplerate,
                input=True,
                output=True,
                stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.2)

    stream.stop_stream()
    stream.close()

    p.terminate()

    input_file.close()
    output_file.close()