import soundfile as sf 
import pyaudio as pa 
import numpy as np 
import time
import scipy.signal as sig 
from scipy.signal import chirp
from scipy.fftpack import ifft, fft, rfft, fftshift, fftfreq
from audio_utilities import pad_input_to_output_length, get_delay, generate_sweep, round_up_to_multiple


class PlayRecorder():

    """
    Class for playing and recording mono audio simultaneously. 
    Methods for measuring impulse response and transfer function.
    """

    def __init__(self, data=None, input_file=None, output_file=None, channels=1, samplerate=44100, chunk=1024): # ch, fs, chunk, format?

        self.p = pa.PyAudio()

        self.recorded_data = []
        self.recorded_data_padded = []  # To be used on the last chunk of data
        self.played_data = []
        self.delay = 0

        self.is_data = None
        self.format = pa.paFloat32  # constant
        self.dtype = np.float32  # constant
        self.channels = channels
        self.samplerate = samplerate
        self.chunk = chunk

        if data.size == 0 and input_file > 0 and output_file > 0:
            self.is_data = False
            assert type(input_file) and type(output_file) == str, "Given input files must be strings ending with .wav"
            self.input_file = sf.SoundFile(input_file, "rb")
            self.output_file = sf.SoundFile(output_file, "wb", self.input_file.samplerate, self.input_file.channels, self.input_file.subtype)
            self.played_data, self.samplerate = self.input_file.read()
        else:
            self.is_data = True
            assert type(data) == np.ndarray, "Given data is not an numpy array"
            self.played_data = data


    def playrec_from_file(self):

        assert self.is_data == False, "The PlayRecorder instance is not initialized in the correct mode(is_data == True)"

        # in_data is the recorded data. callback() is called in a separate thread
        def callback(in_data, frame_count, time_info, status):
            in_data = np.fromstring(in_data, dtype=self.dtype).reshape(frame_count, self.input_file.channels)  # Convert the data before writing
            out_data = self.input_file.read(frame_count, dtype=self.dtype)  # Read in the data to be played(and then recorded into in_data)
            self.output_file.write(in_data)  # Write one frame of the recorded data to the output file
            return (out_data, pa.paContinue)  # out_data is the played data

        stream = self.p.open(format=self.format,
                channels=self.input_file.channels,
                rate=self.input_file.samplerate,
                frames_per_buffer=self.chunk,
                input=True,
                output=True,
                stream_callback=callback)

        print("RECORDING")
        stream.start_stream()  # callback() is called here

        while stream.is_active():
                time.sleep(0.2)

        print("DONE. Closing stream.")

        stream.stop_stream()
        stream.close()

        self.p.terminate()

        self.input_file.close()
        self.output_file.close()


    def playrec_from_data(self, delta_t):

        assert self.is_data == True, "The PlayRecorder instance is not initialized in the correct mode(is_data == False)"
        
        i = 0  # Index to traverse the numpy arrays
        recorded_data_length = round_up_to_multiple(self.samplerate*delta_t, self.chunk)

        played_data_padded = np.pad(self.played_data, (0, recorded_data_length), "constant", constant_values=(0, 0))
        self.recorded_data = np.empty((len(self.played_data)+delta_t*self.samplerate, self.channels), dtype=self.dtype)

        chunk_rest = len(self.recorded_data) % self.chunk
        diff = self.chunk - chunk_rest

        def callback(in_data, frame_count, time_info, status):

            nonlocal i
            in_data = np.fromstring(in_data, dtype=self.dtype).reshape(self.chunk, self.channels)  # Convert the data

            # Last chunk of in_data might be shorter than the chunk size, so the left over samples are zero padded:
            if len(self.recorded_data[i:i+self.chunk]) < self.chunk:
                self.recorded_data_padded = np.pad(self.recorded_data, [(0, diff), (0, 0)], "constant", constant_values=(0, 0))
                self.recorded_data_padded[i:i+diff] = in_data[:diff]
                return (None, pa.paComplete)
            else:
                self.recorded_data[i:i+self.chunk] = in_data

            out_data = played_data_padded[i:i+self.chunk]
            i = i + self.chunk
            return (out_data, pa.paContinue)

        stream = self.p.open(format=self.format,
                channels=self.channels,
                rate=self.samplerate,
                frames_per_buffer=self.chunk,
                input=True,
                output=True,
                stream_callback=callback)

        print("RECORDING")
        stream.start_stream()  # callback() is called here

        while stream.is_active():
            time.sleep(0.2)

        print("DONE. Closing stream.")

        stream.stop_stream()
        stream.close()

        self.p.terminate()

        return self.recorded_data_padded


    def impulse_response(self):
        
        assert self.recorded_data.size > 0, "There is no recorded data"

        # Inverse filter:
        played_padded = pad_input_to_output_length(self.played_data, self.recorded_data)
        T = played_padded.shape[0] / self.samplerate
        t = np.arange(0, T*self.samplerate) / self.samplerate
        R = np.log(20000/20)
        k = np.exp(t*R/T).astype(np.float32)
        f = played_padded[::-1] / k
        # Impulse response:
        return sig.fftconvolve(self.recorded_data, f, mode="same")  # same


    def transfer_function(self):
        played_data_padded = pad_input_to_output_length(self.played_data, self.recorded_data)
        played_data_fft = rfft(played_data_padded) / len(self.recorded_data)
        recorded_data_fft = rfft(self.recorded_data) / len(self.played_data)
        return recorded_data_fft / played_data_fft


    def get_delay(self, t):
        # t is how many seconds of the singals should be evaluated. Use a small value for low latency (f.ex. 1 sec)
        corr = np.correlate(self.played_data[:int(t*self.samplerate)], self.recorded_data[:int(t*self.samplerate)], mode="same")  # mode="same" makes it very slow => use small t
        self.delay = (int(len(corr)/2) - np.argmax(corr)) / self.samplerate  # seconds
        return self.delay
