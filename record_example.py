import soundfile as sf 
import pyaudio as pa 
import numpy as np 

CHUNK = 1024
FORMAT = pa.paFloat32
CHANNELS = 1  # 2 channels makes the .wav file twice as long(something to do with how frames are put in the array).
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pa.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(np.fromstring(data, dtype=np.float32))
    

print("* done recording")

numpydata = np.hstack(frames)

sf.write(WAVE_OUTPUT_FILENAME, numpydata, RATE)

stream.stop_stream()
stream.close()
p.terminate()