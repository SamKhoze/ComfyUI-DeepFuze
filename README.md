# DeepFuze 

![DeepFuze Lipsync](https://user-images.githubusercontent.com/4397546/222490039-b1f6156b-bf00-405b-9fda-0c9a9156f991.gif)


## Overview

DeepFuze is a state-of-the-art deep learning tool that seamlessly integrates with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to revolutionize facial transformations, lipsyncing, video generation, voice cloning, face swapping, and lipsync translation. Leveraging advanced algorithms, DeepFuze enables users to combine audio and video with unparalleled realism, ensuring perfectly synchronized facial movements. This innovative solution is ideal for content creators, animators, developers, and anyone seeking to elevate their video editing projects with sophisticated AI-driven features.


[![DeepFuze Lipsync](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Lipsync.jpg)](https://www.youtube.com/watch?v=9WbvlOK_BlI "DeepFuze Lipsync")

[![IMAGE ALT TEXT HERE](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Lipsync_02.jpg)](https://www.youtube.com/watch?v=1c5TK3zTKr8)

---

## Installation

### Prerequisites for Voice Cloning and Lipsyncing

Below are the two ComfyUI repositories required to load video and audio. Install them into your `custom_nodes` folder:

1. Clone the repositories:
    ```bash
    cd custom_nodes
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    git clone https://github.com/a1lazydog/ComfyUI-AudioScheduler.git
    ```

### Running the Model and Installation

2. Clone this repository into the `custom_nodes` folder and install requirements:
    ```bash
    git clone https://github.com/SamKhoze/CompfyUI-DeepFuze.git
    cd CompfyUI-DeepFuze
    pip3 install -r requirements.txt
    ```

3. Download models from the links below or download all models at once via [DeepFuze Models](https://drive.google.com/drive/folders/1dyu81WAP7_us8-loHjOXZzBJETNeJYJk?usp=sharing) 
----

### Windows Native

- Make sure you have `ffmpeg` in the `%PATH%`, following [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) tutorial to install `ffmpeg` or using scoop.

----

### For MAC users please set the environment variable before running it
This method has been tested on a M1 and M3 Mac 

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
### macOS needs to install the original dlib.
```
pip install dlib 
```
---

## DeepFuze Lipsync

This node generates lipsyncing video from, video, image, and WAV audio files.

**Input Types:**
- `images`: Extracted frame images as PyTorch tensors.
- `audio`: An instance of loaded audio data.
- `mata_batch`: Load batch numbers via the Meta Batch Manager node.

**Output Types:**
- `IMAGES`: Extracted frame images as PyTorch tensors.
- `frame_count`: Output frame counts int.
- `audio`: Output audio.
- `video_info`: Output video metadata.

**DeepFuze Lipsync Features:**
- `enhancer`: You can add an enhancer to improve the quality of the generated video. Using gfpgan or RestoreFormer to enhance the generated face via face restoration network
- `frame_enhancer`: You can add an enhancing the whole frame of video
- `face_mask_padding_left` : padding to left on the face while lipsycing
- `face_mask_padding_right` : padding to right on the face while lipsycing
- `face_mask_padding_bottom` : padding to bottom on the face while lipsycing
- `face_mask_padding_top` : padding to top on the face while lipsycing
- `device` : [cpu,gpu]
- `trim_frame_start`: remove the number of frames from start
- `trim_frame_end`: remove the number of frames from end
- `save_outpou`: If it is True, it will save the output.

![Lipsyncing Node example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/examples/node.jpeg)

### DeepFuze_TTS

**Languages:**

**DeepFuze_TTS voice cloning supports 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).**

This node is used to clone any voice from typed input. The audio file should be 10-15 seconds long for better results and should not have much noise. 

**Input Types:**
- `audio`: An instance of loaded audio data.
- `text`: Text to generate the cloned voice audio.

**Output Types:**
- `audio`: An instance of loaded audio data.

![TTS Node example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/imgs/DeepFuze_TTS.jpg)


**Basic Integration**

![BasicWorkspace](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/imgs/BasicWorkspace.jpg)

---

## Example of How to Use DeepFuze Programmatically

```python
from deepfuze import DeepFuze

# Initialize the DeepFuze instance
deepfuze = DeepFuze()

# Load video and audio files
deepfuze.load_video('path/to/video.mp4')
deepfuze.load_audio('path/to/audio.mp3')
deepfuze.load_checkpoint('path/to/checkpoint_path')

# Set parameters (optional)
deepfuze.set_parameters(sync_level=5, transform_intensity=3)

# Generate lipsynced video
output_path = deepfuze.generate(output='path/to/output.mp4')

print(f"Lipsynced video saved at {output_path}")
```

# Acknowledgements

This repository could not have been completed without the contributions from [SadTalker](https://github.com/OpenTalker/SadTalker/tree/main), [Facexlib](https://github.com/xinntao/facexlib), [GFPGAN](https://github.com/TencentARC/GFPGAN), [GPEN](https://github.com/yangxy/GPEN), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [TTS](https://github.com/coqui-ai/TTS/tree/dev), [SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/), and [wav2lip](https://github.com/Rudrabha/Wav2Lip), 

1. Please carefully read and comply with the open-source license applicable to this code and models before using it. 
2. Please carefully read and comply with the intellectual property declaration applicable to this code and models before using it.
3. This open-source code runs completely offline and does not collect any personal information or other data. If you use this code to provide services to end-users and collect related data, please take necessary compliance measures according to applicable laws and regulations (such as publishing privacy policies, adopting necessary data security strategies, etc.). If the collected data involves personal information, user consent must be obtained (if applicable). 
4. It is prohibited to use this open-source code for activities that harm the legitimate rights and interests of others (including but not limited to fraud, deception, infringement of others' portrait rights, reputation rights, etc.), or other behaviors that violate applicable laws and regulations or go against social ethics and good customs (including providing incorrect or false information, terrorist, child/minors pornography and violent information, etc.). Otherwise, you may be liable for legal responsibilities.

The DeepFuze code is developed by Dr. Sam Khoze and his team. Feel free to use the DeepFuze code for personal, research, academic, and non-commercial purposes. You can create videos with this tool, but please make sure to follow local laws and use it responsibly. The developers will not be responsible for any misuse of the tool by users. For commercial use, please contact us at info@cogidigm.com.
