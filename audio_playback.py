import sounddevice
from io import BytesIO
import os
import time
import folder_paths
from scipy.io import wavfile
from scipy.io.wavfile import write
import subprocess
import sounddevice
import numpy as np
import torchaudio
from pydub import AudioSegment
from .utils import get_audio
audio_path = os.path.join(folder_paths.get_input_directory(),"audio")



output_dir = os.path.join(folder_paths.get_output_directory(),"n-suite")
YELLOW = '\33[33m'
END = '\33[0m'

class AudioData:
    def __init__(self, audio_file) -> None:
        
        # Extract the sample rate
        sample_rate = audio_file.frame_rate

        # Get the number of audio channels
        num_channels = audio_file.channels

        # Extract the audio data as a NumPy array
        audio_data = np.array(audio_file.get_array_of_samples())
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    def get_channel_audio_data(self, channel: int):
        if channel < 0 or channel >= self.num_channels:
            raise IndexError(f"Channel '{channel}' out of range. total channels is '{self.num_channels}'.")
        return self.audio_data[channel::self.num_channels]
    
    def get_channel_fft(self, channel: int):
        audio_data = self.get_channel_audio_data(channel)
        return fft(audio_data)

os.makedirs(output_dir,exist_ok=True)
os.makedirs(os.path.join(output_dir,"videos"),exist_ok=True)
class SaveAudio:
     
    @classmethod
    def INPUT_TYPES(s):
        
        #print(f"Temporary folder {frames_output_dir} has been emptied.")
        return {"required": 
                    {"audio": ("AUDIO", ),
                     "start_time": ([str(i) for i in range(10000)],),
                     "end_time": ([str(i) for i in range(10000)],),
                     },
                }
                

    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "DeepFuze"

    def save_audio(self, audio,start_time,end_time):
        audio_path = folder_paths.get_input_directory()
        audio_root = os.path.basename(audio_path)
        file_path = os.path.join(audio_path,str(time.time()).replace(".","")+".wav")
        print(audio_path,file_path)
        outfile = os.path.join(audio_path,str(time.time()).replace(".","_")+".wav")
        torchaudio.save(file_path,audio["waveform"][0],audio["sample_rate"])
        audio_name = file_path.split("/")[-1]
        audio = get_audio(file_path)
        print(audio)
        return {"ui": {"audio":[audio_name,audio_root]},"result" : [audio]}


class PlayBackAudio:

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":{
                "audio": ("AUDIO",)
            }
        }
    OUTPUT_NODE = True
    RETURN_NAMES = ()
    RETURN_TYPES = ()
    CATEGORY = "DeepFuze"
    FUNCTION = "play_audio"

    def play_audio(self,audio):
        file = os.path.join(folder_paths.get_input_directory(),str(time.time()).replace(".","")+".wav")
        torchaudio.save(file,audio["waveform"][0],audio["sample_rate"])
        audio_file = AudioSegment.from_file(file, format="wav")
        audio = AudioData(audio_file)
        sounddevice.play(audio.audio_data,audio.sample_rate)
        return ()
