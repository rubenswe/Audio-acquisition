import soundfile as sf 
import pyaudio as pa 
import numpy as np 
import matplotlib.pyplot as plt 


class PlayRecorder():

    """
    Class for playing and recording audio simultaneously.
    """

    def __init__(self, input_file, output_file):  # data?
        # if type(input) == np.ndarray: ... In the case of input is data and not a filename
        # def __init()__(self, files*)  arbitrary number of files
        # def __init()__(self, input_file, output_file=None ... if output_file == None --> Ignore
        self.input_file = sf.SoundFile(input_file, "rb")  # Use with-statement(context manager)? - will automatically close objects and avoid resource leakage
        self.output_file = sf.SoundFile(output_file, "wb", self.input_file.samplerate, self.input_file.channels, self.input_file.subtype)
        self.p = pa.PyAudio()
        # open stream here or in its own method?
        """self.stream = self.p.open(format=pa.paFloat32,
                        channels=channels,  # What to do in the case of 1 channel for playing and 2 for recording? Use 2 channels always?(leaving one channel empty in the mono case if possible)
                        rate=samplerate,
                        frames_per_buffer=chunk, # What should be used?
                        input_device_index=None,  #  Index of Input Device to use. Unspecified (or None) uses default device. Ignored if input is False.
                        output_device_index=None,  # =select_device(...)
                        input=True,
                        output=True,
                        stream_callback=callback)"""

        # context manager using with-statement? -> closes soundfile files automatically

    # In the case of data(array, not file) as input:
    @classmethod
    def from_data(cls, in_data, parameter_list):  # more parameters here  
        # has to return an instance of the class
        raise NotImplementedError

    def get_available_devices(self):
        for i in range (self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            print("{}: {} Input channels, {} Output channels".format(dev["name"], dev["maxInputChannels"], dev["maxOutputChannels"]))

        print("--------------------------------------------------------------------------------\n")
        print("Default input device: {}\n".format(self.p.get_default_input_device_info()["name"]),
              "Default output device: {}".format(self.p.get_default_output_device_info()["name"]))
        # return devs...?

    def select_device(self, parameter_list):
        raise NotImplementedError

    def playrec_from_file(self, parameter_list):
        # audio_acquisition.py
        # pyaudio instance and stream already opened in __init__()
        # Remember to call close() (or do the closing in the function?)
        raise NotImplementedError

    def playrec_from_data(self, parameter_list):
        # audio_acquisition.py
        # pyaudio instance and stream already opened in __init__()
        # Remember to call close() (or do the closing in the function?)
        raise NotImplementedError

    def close(self):
        # Method for closing all open files, pyaudio instances and streams(another one for streams?)
        raise NotImplementedError


pr = PlayRecorder("sweep_file.wav", "out_file.wav")

pr.get_available_devices()
