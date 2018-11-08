import soundfile as sf 
import pyaudio as pa 
import numpy as np 
import matplotlib.pyplot as plt 


class PlayRecorder():
    def __init__(self, input_file, output_file):  # data?
        # if type(input) == np.ndarray: ... In the case of input is data and not a filename
        self.input_file = sf.SoundFile(input_file, "rb")
        self.output_file = sf.SoundFile(output_file, "wb", self.input_file.samplerate, self.input_file.channels, self.input_file.subtype)
        self.p = pa.PyAudio()

    # In the case of data as input:
    @classmethod
    def from_data(cls, in_data, parameter_list):  # more parameters here  
        # has to return an instance of the class?
        raise NotImplementedError

    def get_available_devices(self):
        for i in range (self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            print("{}: {} Input channels, {} Output channels".format(dev["name"], dev["maxInputChannels"], dev["maxOutputChannels"]))

        print("--------------------------------------------------------------------------------\n")
        print("Default input device: {}\n".format(self.p.get_default_input_device_info()["name"]),
              "Default output device: {}".format(self.p.get_default_output_device_info()["name"]))


pr = PlayRecorder("sweep_file.wav", "out_file.wav")

pr.get_available_devices()