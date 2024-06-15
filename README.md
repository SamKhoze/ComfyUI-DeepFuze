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
# MAC Installation
## Vide Tutorial How To Install on MAC [YOUTUBE LINK](https://youtu.be/FWdOlj60fig)

## For MAC users please set the environment variable before running it

### Install [Pytorch](https://pytorch.org/) 

[Here](https://developer.apple.com/metal/pytorch/) how to install and test your PyTorch

This method has been tested on a M1 and M3 Mac, You must run the below code on your terminal window for Mac Metal Performance Shaders (MPS) Apple's specialized solution for high-performance GPU programming on their devices. Integrating closely with the Metal framework, MPS provides a suite of highly optimized shaders for graphics and computing tasks, which is particularly beneficial in machine learning applications.

## Important Steps (if you miss these steps it will not work)
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
### macOS needs to install the original dlib.
```
pip install dlib 
```
### Install Text to Speech for Voice Cloning Node
```
pip install TTS 
```
### Errors 

If you get an error installing TTS, it is most likely because you have different versions of Python, make sure to install the correct version

If you get an error: ImportError: cannot import name 'get_full_repo_name' from 'huggingface_hub'
Run the below codes on your terminal it will solve the issue

```
conda install chardet 
```
```
pip install --upgrade transformers==4.39.2 
```

After preparing the environmental variables navigate into your custom_nodes folder and git clone or manually download the code and extract it into the custom_nodes folder
```
    git clone https://github.com/SamKhoze/CompfyUI-DeepFuze.git
    cd CompfyUI-DeepFuze
    pip3 install -r requirements.txt
```
if you get any error for any packages, open the requirements.txt file with any text editor remove the version from the front of the package name, and reinstall requirments.txt again

# Models

You can download models directly from [GoogleDrive](https://drive.google.com/drive/folders/1dyu81WAP7_us8-loHjOXZzBJETNeJYJk?usp=sharing) and place models into the PATH `./ComfyUI/models/deepfuze/` Ensure to manually download each model one by one and place them, due to the size of the models some of the models won't download if you download the folder preparing the environmental variables navigate into your custom_nodes folder and git clone or manually download the code and extract it into the custom_nodes folder.

---
## Repository Structure

```plaintext
ComfyUI-DeepFuze/
├── __init__.py
├── __pycache__/
│   ├── __init__.cpython-311.pyc
│   ├── audio_playback.cpython-311.pyc
│   ├── llm_node.cpython-311.pyc
│   ├── nodes.cpython-311.pyc
│   └── utils.cpython-311.pyc
├── audio_playback.py
├── deepfuze/
│   ├── __init__.py
│   ├── audio.py
│   ├── choices.py
│   ├── common_helper.py
│   ├── config.py
│   ├── content_analyser.py
│   ├── core.py
│   ├── download.py
│   ├── execution.py
│   ├── face_analyser.py
│   ├── face_helper.py
│   ├── face_masker.py
│   ├── face_store.py
│   ├── ffmpeg.py
│   ├── filesystem.py
│   ├── globals.py
│   ├── installer.py
│   ├── logger.py
│   ├── memory.py
│   ├── metadata.py
│   ├── normalizer.py
│   ├── process_manager.py
├── requirements.txt
├── images/
├── install.py
├── LICENSE.txt
├── llm_node.py
├── mypy.ini
├── nodes.py
├── README.md
├── requirements.txt
├── run.py
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_cli_face_debugger.py
│   ├── test_cli_face_enhancer.py
│   ├── test_cli_face_swapper.py
│   ├── test_cli_frame_colorizer.py
│   ├── test_cli_frame_enhancer.py
│   ├── test_cli_lip_syncer.py
│   ├── test_common_helper.py
│   ├── test_config.py
│   ├── test_download.py
│   ├── test_execution.py
│   ├── test_face_analyser.py
│   ├── test_ffmpeg.py
│   ├── test_filesystem.py
│   ├── test_memory.py
│   ├── test_normalizer.py
│   ├── test_process_manager.py
│   ├── test_vision.py
│   └── test_wording.py
├── tts_generation.py
└── utils.py
```

# Nodes

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
- `face_enhancer`: You can add an enhancer to improve the quality of the generated video. Using gfpgan or RestoreFormer to enhance the generated face via face restoration network
- `frame_enhancer`: You can add an enhance the whole frame of the video
- `face_mask_padding_left` : padding to left on the face while lipsyncing
- `face_mask_padding_right` : padding to the right on the face while lipsyncing
- `face_mask_padding_bottom` : padding to the bottom on the face while lipsyncing
- `face_mask_padding_top` : padding to the top on the face while lipsyncing
- `device` : [cpu,gpu]
- `trim_frame_start`: remove the number of frames from start
- `trim_frame_end`: remove the number of frames from end
- `save_output`: If it is True, it will save the output.

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

This repository could not have been completed without the contributions from [FaceFusion](https://github.com/facefusion/facefusion), [InsightFace](https://github.com/deepinsight/insightface),[SadTalker](https://github.com/OpenTalker/SadTalker/tree/main), [Facexlib](https://github.com/xinntao/facexlib), [GFPGAN](https://github.com/TencentARC/GFPGAN), [GPEN](https://github.com/yangxy/GPEN), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [TTS](https://github.com/coqui-ai/TTS/tree/dev), [SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/), and [wav2lip](https://github.com/Rudrabha/Wav2Lip), 

1. Please carefully read and comply with the open-source license applicable to this code and models before using it. 
2. Please carefully read and comply with the intellectual property declaration applicable to this code and models before using it.
3. This open-source code runs completely offline and does not collect any personal information or other data. If you use this code to provide services to end-users and collect related data, please take necessary compliance measures according to applicable laws and regulations (such as publishing privacy policies, adopting necessary data security strategies, etc.). If the collected data involves personal information, user consent must be obtained (if applicable). 
4. It is prohibited to use this open-source code for activities that harm the legitimate rights and interests of others (including but not limited to fraud, deception, infringement of others' portrait rights, reputation rights, etc.), or other behaviors that violate applicable laws and regulations or go against social ethics and good customs (including providing incorrect or false information, terrorist, child/minors pornography and violent information, etc.). Otherwise, you may be liable for legal responsibilities.

The DeepFuze code is developed by Dr. Sam Khoze and his team. Feel free to use the DeepFuze code for personal, research, academic, and non-commercial purposes. You can create videos with this tool, but please make sure to follow local laws and use it responsibly. The developers will not be responsible for any misuse of the tool by users. For commercial use, please contact us at info@cogidigm.com.
