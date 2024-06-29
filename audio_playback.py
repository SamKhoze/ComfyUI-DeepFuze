import sounddevice
from io import BytesIO
import os
import time
import folder_paths
from scipy.io import wavfile
from scipy.io.wavfile import write
import subprocess
import sounddevice
from pydub import AudioSegment

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
    def __init__(self):
        
        self.type = "output"

     
    @classmethod
    def INPUT_TYPES(s):
        
        try:
            shutil.rmtree(frames_output_dir)
            os.mkdir(frames_output_dir)
        except:
            pass
        

        #print(f"Temporary folder {frames_output_dir} has been emptied.")
        return {"required": 
                    {"audio": ("VHS_AUDIO", ),
                     "METADATA": ("STRING",  {"default": ""}  ), 
                     "start_time": ([str(i) for i in range(10000)],),
                     "end_time": ([str(i) for i in range(10000)],),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
                

    RETURN_NAMES = ()
    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "DeepFuze"

    def save_video(self, audio,METADATA,start_time,end_time,prompt=None, extra_pnginfo=None):
        file_path = os.path.join(audio_path,str(time.time()).replace(".","")+".wav")
        outfile = os.path.join(audio_path,str(time.time()).replace(".","_")+".wav")
        open(file_path,"wb").write(audio())
        Fs, data = wavfile.read(file_path)
        n = data.size
        t = n / Fs
        print(t)
        if t < int(end_time):
            end_time = t
        if int(end_time) > 0:
            subprocess.run(['ffmpeg','-i',file_path,'-ss',start_time,'-to',end_time,outfile])
            file_path = outfile
        file_path_ = file_path.replace(".wav",".mp4")
        print(file_path)
        file_path_ = os.path.join(output_dir,"videos",file_path_.split('/')[-1])
        os.system(f"ffmpeg -i {file_path} {file_path_}")
        return {"ui": {"text": [file_path_.split("/")[-1]],}}


class PlayBackAudio:

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":{
                "audio": ("VHS_AUDIO",)
            }
        }
    OUTPUT_NODE = True
    RETURN_NAMES = ()
    RETURN_TYPES = ()
    CATEGORY = "DeepFuze"
    FUNCTION = "play_audio"

    def play_audio(self,audio):
        file = BytesIO(audio())
        audio_file = AudioSegment.from_file(file, format="wav")
        audio = AudioData(audio_file)
        sounddevice.play(audio.audio_data,audio.sample_rate)
        return ()
