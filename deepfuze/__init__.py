import os
import subprocess

class DeepFuze:
    def __init__(self):
        self.video_path = ""
        self.audio_path = ""
        self.output_path = ""
        self.checkpoint_pth = ""
        self.sync_level = 5
        self.transform_intensity = 3

    def load_checkpoint(self, checkpoint_path):
        if not os.path.isdir(checkpoint_path):
            raise ValueError("chekpoint folder not found")
        self.checkpoint_path = checkpoint_path

    def load_audio(self, audio_path):
        try:
            if audio_path.split(".")[-1].lower() not in ["wav","mp3"]:
                raise ValueError("Audio file must be wav or mp3")
            if not os.path.isfile(audio_path):
                raise ValueError("Audio file not found")
            self.audio_path = audio_path
            print(f"Audio file loaded from {self.audio_path}")
        except Exception as e:
            print(f"Failed to load audio file: {e}")

    def load_video(self, video_path):
        try:
            if video_path.split(".")[-1].lower() not in ["mp4","avi","webm"]:
                raise ValueError("Video file must be mp4 or avi or webm")
            if not os.path.isfile(video_path):
                raise ValueError("Video file not found")
            self.video_path = video_path
            print(f"video file loaded from {self.video_path}")
        except Exception as e:
            print(f"Failed to load audio file: {e}")

            
    def generate(self,output_path,device='cpu'):
        try:
            command = [
                'python',
                './run.py',               # Script to run
                '--frame-processors',
                "lip_syncer",
                "-s",
                self.audio_path,
                '-t',        # Argument: segmentation path
                self.video_path,
                '-o',
                output_path,
                '--headless'
            ]
            if device=="cuda":
                command.extend(['--execution-providers',"cuda"])
            elif device=="mps":
                command.extend(['--execution-providers',"coreml"])
            result = subprocess.run(command,stdout=subprocess.PIPE)
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
            print(f"Lipsynced video saved at {output_path}")
            return output_path
        except Exception as e:
            print(f"Failed to generate lipsynced video: {e}")
            return None



# Usage example:
if __name__ == "__main__":
    deepfuze = DeepFuze()

    # Load video and audio files
    deepfuze.load_video('path/to/video.mp4')
    deepfuze.load_audio('path/to/audio.mp3')
    deepfuze.load_checkpoint('path/to/checkpoint_path')

    # Generate lipsynced video
    output_path = deepfuze.generate(output='path/to/output.mp4')

    print(f"Lipsynced video saved at {output_path}")
