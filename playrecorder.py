import soundfile as sf 
import numpy as np 
import matplotlib.pyplot as plt 


class PlayRecorder():
    def __init__(self, input_file, output_file):
        self.input_file = sf.SoundFile(input_file, "rb")
        self.output_file = sf.SoundFile(output_file, "wb")
