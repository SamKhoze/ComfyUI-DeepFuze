import sounddevice

import os
import time
import folder_paths
from scipy.io import wavfile
from scipy.io.wavfile import write
import subprocess
import sounddevice


audio_path = os.path.join(folder_paths.get_input_directory(),"audio")



YELLOW = '\33[33m'
END = '\33[0m'


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
                    {"audio": ("AUDIO", ),
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
        write(file_path,audio.sample_rate,audio.audio_data)
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
        file_path_ = f"/Users/yash/Desktop/ComfyUI/output/n-suite/videos/{file_path_.split('/')[-1]}"
        os.system(f"ffmpeg -i {file_path} {file_path_}")
        return {"ui": {"text": [file_path_.split("/")[-1]],}}


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
        sounddevice.play(audio.audio_data,audio.sample_rate)
        return ()
