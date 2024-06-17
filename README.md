# DeepFuze 

## Overview

DeepFuze is a state-of-the-art deep learning tool that seamlessly integrates with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to revolutionize facial transformations, lipsyncing, video generation, voice cloning, face swapping, and lipsync translation. Leveraging advanced algorithms, DeepFuze enables users to combine audio and video with unparalleled realism, ensuring perfectly synchronized facial movements. This innovative solution is ideal for content creators, animators, developers, and anyone seeking to elevate their video editing projects with sophisticated AI-driven features.

### Watch the 4k Quality Video on [Youtube](https://www.youtube.com/watch?v=9WbvlOK_BlI)

[![DeepFuze Lipsync](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/SamLipsyncDemo.gif)
---
### Watch the 4k Quality Video on [Youtube](https://www.youtube.com/watch?v=1c5TK3zTKr8)

[![IMAGE ALT TEXT HERE](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/Lipsync_Demo02.gif)

---

## Installation

### Prerequisites for Voice Cloning and Lipsyncing

Below are the two ComfyUI repositories required to load video and audio. Install them into your `custom_nodes` folder:

Clone the repositories:


    cd custom_nodes
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    git clone https://github.com/a1lazydog/ComfyUI-AudioScheduler.git


# Installation & Models Download 

----

# Windows Installation

## Portable ComfyUI for Windows

### Step 1
You must install [Visual Studio](https://visualstudio.microsoft.com/downloads/), it works with the community version
OR VS [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop Development with C++" under "Workloads -> Desktop & Mobile"
### Step 2
Install from the ComfyUI manager, select Install via git URL, and copy past:

    https://github.com/SamKhoze/CompfyUI-DeepFuze.git
### Step 3

Restart your ComfyuUI

----
# MAC Installation
## Video Tutorial How To Install on MAC [YOUTUBE LINK](https://youtu.be/FWdOlj60fig)

## For MAC users please set the environment variable before running it

### Install [Pytorch](https://pytorch.org/) 

[Here](https://developer.apple.com/metal/pytorch/) how to install and test your PyTorch

This method has been tested on a M1 and M3 Mac, You must run the below code on your terminal window for Mac Metal Performance Shaders (MPS) Apple's specialized solution for high-performance GPU programming on their devices. Integrating closely with the Metal framework, MPS provides a suite of highly optimized shaders for graphics and computing tasks, which is particularly beneficial in machine learning applications.

## ⚠️⚠️⚠️ Important Steps (if you miss these steps it will not work)
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

# Install ComfyUI-DeepFuze

After preparing the environmental variables navigate into your custom_nodes folder and git clone or manually download the code and extract it into the custom_nodes folder
```

cd custom_nodes
git clone https://github.com/SamKhoze/CompfyUI-DeepFuze.git
cd CompfyUI-DeepFuze
pip install -r requirements.txt
```   

### Errors 

If you get an error installing TTS, it is most likely because you have different versions of Python, make sure to install the correct version
----
If you get an error: ImportError: cannot import name 'get_full_repo_name' from 'huggingface_hub'
Run the below codes on your terminal it will solve the issue

```
conda install chardet 
```
```
pip install --upgrade transformers==4.39.2 
```

if you get any error for any packages, open the requirements.txt file with any text editor remove the version from the front of the package name, and reinstall requirments.txt again

# Models🌟🌟🌟

You can download models directly from [GoogleDrive](https://drive.google.com/drive/folders/1dyu81WAP7_us8-loHjOXZzBJETNeJYJk?usp=sharing) and place models into the PATH `./ComfyUI/models/deepfuze/` Ensure to manually download each model one by one and place them, due to the size of the models some of the models won't download if you download the folder preparing the environmental variables navigate into your custom_nodes folder and git clone or manually download the code and extract it into the custom_nodes folder.

## OpenAI API setup for voice cloning (Optional)
---

You need an OpenAI API Key if you wish to use the "Openai LLM" node for generating dialogues for voice cloning
---
To use the "Openai LLM" node for voice cloning dialogues, you need an OpenAI API Key. You can get this key and set it up by following the instructions in the [OpenAI Developer quickstart guide](https://platform.openai.com/docs/quickstart). Please note that the "Openai LLM" node does not save your API key. Every time you close the node, you will need to manually copy and paste your API key. You can also add the API key as an Environment Variable using the following commands: For Windows: `setx OPENAI_API_KEY "your-api-key-here"`, and for Mac: `export OPENAI_API_KEY='your-api-key-here'`. The next time you need to copy and paste your API key into the LLM Node, you can type the following command in your terminal: `echo $OPENAI_API_KEY`, and it will print your API Key, allowing you to copy and paste it into your Openai LLM node.

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

---
## DeepFuze Lipsync

![DeepFuze FaceSwap Node example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Lipsync_SimpleWorkflow.jpg)

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
- `enhancer`: You can add a face enhancer to improve the quality of the generated video via the face restoration network.
- `frame_enhancer`: You can add an enhance the whole frame of the video.
- `face_mask_padding_left` : Padding to left of the face while lipsyncing.
- `face_mask_padding_right` : Padding to the right of the face while lipsyncing.
- `face_mask_padding_bottom` : Padding to the bottom of the face while lipsyncing.
- `face_mask_padding_top` : Padding to the top of the face while lipsyncing.
- `device` : [cpu,gpu]
- `frame_rate`: Set the frame rate.
- `loop_count`: How many additional times the video should repeat.
- `filename_prefix`: Prefix naming for the output video.
- `pingpong`: Causes the input to be played back in reverse to create a clean loop.
- `save_output`: Saving the output on output folder.

---
## DeepFuze FaceSwap

![DeepFuze FaceSwap Node example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_FaceSwap.jpg)

![DeepFuze FaceSwap example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/AnimateDiff_00002-ezgif.com-video-to-gif-converter.gif)

This node generates lipsyncing video from, video, image, and WAV audio files.

**Input Types:**
- `source_images`: Extracted frame image as PyTorch tensors for swapping.
- `target_images`: Extracted frame images as PyTorch tensors to input the source video/image.
- `mata_batch`: Load batch numbers via the Meta Batch Manager node.


**Output Types:**
- `IMAGES`: Extracted frame images as PyTorch tensors.
- `frame_count`: Output frame counts int.
- `audio`: Output audio.
- `video_info`: Output video metadata.

**DeepFuze FaceSwap Features:**
- `enhancer`: You can add a face enhancer to improve the quality of the generated video via face restoration network.
- `faceswap_model`: You can select different models for swapping.
- `frame_enhancer`: You can add an enhance the whole frame of the video.
- `face_detector_model`: You can select different models for face detection.
- `face_mask_padding_left` : Padding to left on the face while lipsyncing.
- `face_mask_padding_right` : Padding to the right on the face while lipsyncing.
- `face_mask_padding_bottom` : Padding to the bottom on the face while lipsyncing.
- `face_mask_padding_top` : Padding to the top on the face while lipsyncing.
- `device` : [cpu,gpu]
- `frame_rate`: Set the frame rate.
- `loop_count`: How many additional times the video should repeat.
- `filename_prefix`: Prefix naming for the output video.
- `pingpong`: Causes the input to be played back in reverse to create a clean loop.
- `save_output`: Saving the output on output folder.

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


**Basic Integrations**

Voice Cloning + Lipsync Generation
![BasicWorkspace](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/VoiceCloning.jpg)

Voice Cloning + Lipsync Generation + FaceSwap
![BasicWorkspace](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/Lipsync_VoiceCloning_FaceSwap.jpg)

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
