import sounddevice

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
