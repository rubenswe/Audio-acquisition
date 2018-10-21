import pyaudio as pa
import soundfile as sf 
import numpy as np 
import time
import timeit

"""
    Play a sound(chirp, sine) and simultaneously record the response.
"""


input_path = "sweep_file.wav"  # use argparser?
output_path = "record_file.wav"  # use argparser?

input_file = sf.SoundFile(input_path, "rb")  # fs, channels, subtype, endian, and format obtained from reading the file(unless its headerless). Can be found/used by using the SoundFile object.
output_file = sf.SoundFile(output_path, "wb", input_file.samplerate, input_file.channels, input_file.subtype)  # Gets its format from the input

p = pa.PyAudio()

start_time = timeit.default_timer()
# in_data is the recorded data. callback() is called in a separate thread
def callback(in_data, frame_count, time_info, status):
    in_data = np.fromstring(in_data, dtype=np.float32).reshape(frame_count, input_file.channels)  # Convert the data before writing
    out_data = input_file.read(frame_count, dtype=np.float32)  # Read in the data to be played(and then recorded into in_data)
    output_file.write(in_data)  # Write the recorded data to the output file record_file.wav
    return (out_data, pa.paContinue)  # out_data is the played data


stream = p.open(format=pa.paFloat32,
                channels=input_file.channels,
                rate=input_file.samplerate,
                input=True,
                output=True,
                stream_callback=callback)

print("Stream started")
stream.start_stream()  # Calls callback() until there is no more data

while stream.is_active():
    time.sleep(0.2)

stream.stop_stream()
stream.close()
print("Stream ended")

p.terminate()
input_file.close()  # Remember to close the files
output_file.close() 

print("Input file info: {}\n"
      "-------------------------------------\n"
      "Output file info: {}\n".format(sf.info(input_path, True), sf.info(output_path, True)))
print(timeit.default_timer() - start_time)