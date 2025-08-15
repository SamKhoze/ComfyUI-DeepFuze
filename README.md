# THIS REPOSITORY IS NOT COMPATIBLE WITH THE LATEST VERSION OF COMFYUI, DUE TO NOT BEING UPDATED #
# 👉🏼👉🏼👉🏼Please take note of the following information: This repository is compatible and optimized for use with MAC CPU+MPS and Windows with CPU+CUDA. The installation process is not beginner-friendly for enabling CUDA [Toolkit==11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=11) (CUDA Deep Neural Network). If you are unsure about installing CUDA, you can still use the CPU, and no CUDA installation will be necessary. However, if you are able to install CUDA correctly, the processing speed will increase significantly.
# DeepFuze 
### Watch the 4k Quality Video on [YOUTUBE](https://youtu.be/PTXMNz2xyVY)
![DeepFuze Lipsync](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/LipsyncDemo.gif)

## Overview

DeepFuze is a state-of-the-art deep learning tool that seamlessly integrates with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to revolutionize facial transformations, lipsyncing, video generation, voice cloning, face swapping, and lipsync translation. Leveraging advanced algorithms, DeepFuze enables users to combine audio and video with unparalleled realism, ensuring perfectly synchronized facial movements. This innovative solution is ideal for content creators, animators, developers, and anyone seeking to elevate their video editing projects with sophisticated AI-driven features.

### Watch the 4k Quality Video on [Youtube](https://www.youtube.com/watch?v=9WbvlOK_BlI)

![DeepFuze Lipsync](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/SamLipsyncDemo.gif)
---
### Watch the 4k Quality Video on [Youtube](https://www.youtube.com/watch?v=1c5TK3zTKr8)

![IMAGE ALT TEXT HERE](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/Lipsync_Demo02.gif)

---

# Installation & Models Download 

----

# Windows Installation 🖥️

## Portable ComfyUI for Windows

### Step 1
You must install [Visual Studio](https://visualstudio.microsoft.com/downloads/), it works with the community version
OR VS [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop Development with C++" under "Workloads -> Desktop & Mobile"

### Step 2 (installing from ComfyUI-Manager)
From [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) search for DeepFuze, and install the node. Restart your ComfyUI, and look at your terminal window to ensure there is no error, or Install from the ComfyUI manager, select "Install Via  GIT URL", and copy past:

    https://github.com/SamKhoze/ComfyUI-DeepFuze.git
	
![GitInstall](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/Git%20Install.jpg)


### Step 3

Restart your ComfyUI

## IMPORTANT NOTE: CUDA INSTALLATION IS NOT BEGINNER-FRIENDLY, IF YOU DON'T KNOW WHAT YOU ARE DOING DO NOT TRY, USE THE CPU VERSION. 
# CUDA Installation 
**[YOUTUBE LINK](https://youtu.be/ZKhFAF6inR4) step by step instructions**

----
Install Nvidia CUDA [Toolkit==11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)  and [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=11) (CUDA Deep Neural Network) for Deep Learning, you must download the **cuDNN version 8.9.2.26** from the [NVIDIA DEVELOPER cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive), if you don't have developer account you can directly download it from [GoogleDrive](https://drive.google.com/file/d/1sBGH8s7OfmkiyMwXBU2iL01bXyODfXgU/view?usp=drive_link). Ensure install **Cuda1 1.8**. I found this [YOUTUBE](https://www.youtube.com/watch?v=ctQi9mU7t9o&t=655s) video useful for installation. If you have a different version of CUDA here is a [YOUTUBE](https://www.youtube.com/watch?v=I3awjvMZw9A&t=2s) link that guides you on how to uninstall your CUDA. Make sure to create paths in your Environment variable as described on [YOUTUBE VIDEO](https://www.youtube.com/watch?v=ctQi9mU7t9o&t=655s) Restart your computer after creating paths. 
Confirm your Cuda Installation, paste this code on your terminal window `nvcc --version` you should get a response like this: 

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0`
```
----

# MAC Installation 👨🏻‍💻
**Do not install via ComfyUI-Manager it will not work, you must install it manually and follow the instructions below:**
## Video Tutorial How To Install on MAC [YOUTUBE LINK](https://youtu.be/FWdOlj60fig)

## For MAC users please set the environment variable before running it

Activate your Virtual Environment, Conda or Venv
### Install [Pytorch](https://pytorch.org/) 

[Here](https://developer.apple.com/metal/pytorch/) how to install and test your PyTorch

This method has been tested on a M1 and M3 Mac, You must run the below code on your terminal window for Mac Metal Performance Shaders (MPS) Apple's specialized solution for high-performance GPU programming on their devices. Integrating closely with the Metal framework, MPS provides a suite of highly optimized shaders for graphics and computing tasks, which is particularly beneficial in machine learning applications.

## ⚠️⚠️⚠️ Important Steps (if you miss these steps it will not work)

**Copy and paste the command below to your terminal window.**
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
**Mac users must INSTALL ONNX RUNTIME CPU instead of onnxruntime-gpu**
```
pip install onnxruntime
```
**macOS needs to install the original dlib.**
```
pip install dlib 
```
**Install Text to Speech for Voice Cloning Node**
```
pip install TTS 
```
**Navigate into** `custom_nodes` **folder**
```
cd custom_nodes
git clone https://github.com/SamKhoze/CompfyUI-DeepFuze.git
```
**Navigate into the** `CompfyUI-DeepFuze` **folder and install** `requirements.txt` **file**
```
cd CompfyUI-DeepFuze
pip install -r requirements.txt
```
**Prerequisites for Voice Cloning and Lipsyncing**

Below are the two ComfyUI repositories required to load video and audio. Install them into your `custom_nodes` folder:

Clone the repositories:


    cd custom_nodes
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git


### Errors 😾

CV Error: If you encounter the error "ComfyUI_windows_portable\ComfyUI\output\deepfuze\faceswap_file.mp4 could not be loaded with cv," it means that `onnxruntime` is not installed. To fix this, make sure to install `onnxruntime` for CPU and `onnxruntime-gpu` for Windows. Mac users should upgrade OpenCV using the command `pip install --upgrade opencv-python-headless` in their virtual environment. For Windows users, go to ComfyUI Manager, click on "pip install," paste `--upgrade opencv-python-headless`, click OK, and restart your ComfyUI. 

----

Missing zlibwapi.dll error: Search for  NVIDIA zlibwapi.dll file, download it and copy it in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll 

----

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

You can download models directly from [GoogleDrive](https://drive.google.com/drive/folders/1CNdsf_M1cBLgFIS5HQIqm0lCoW4M4QkL?usp=sharing) and place models into the PATH `./ComfyUI/models/deepfuze/` Ensure to manually download each model one by one and place them, due to the size of the models some of the models won't download if you download the folder preparing the environmental variables navigate into your custom_nodes folder and git clone or manually download the code and extract it into the custom_nodes folder.

## OpenAI API setup for voice cloning (Optional)
---

You need an OpenAI API Key if you wish to use the "DeepFuze Openai LLM" node for generating dialogues for voice cloning
---
To use the "Openai LLM" node for voice cloning dialogues, you need an OpenAI API Key. You can get this key and set it up by following the instructions in the [OpenAI Developer quickstart guide](https://platform.openai.com/docs/quickstart). Please note that the "Openai LLM" node does not save your API key. Every time you close the node, you will need to manually copy and paste your API key. You can also add the API key as an Environment Variable using the following commands: For Windows: `setx OPENAI_API_KEY "your-api-key-here"`, and for Mac: `export OPENAI_API_KEY='your-api-key-here'`. The next time you need to copy and paste your API key into the LLM Node, you can type the following command in your terminal: `echo $OPENAI_API_KEY`, and it will print your API Key, allowing you to copy and paste it into your Openai LLM node.

# Nodes

**DeepFuze Nodes Overview** [YOUTUBE LINK](https://youtu.be/elQzQo__kWI)
---
## DeepFuze Lipsync Node 🫦

![DeepFuze Lipsync](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Lipsync_Node.jpg)
![DeepFuze Lipsync Node example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Lipsync_SimpleWorkflow.jpg)

This node generates lipsyncing video from, video, image, and audio files. For higher quality export the IMAGE output as an image batch instead of a video combined, you can get up to 4k quality image size.
IMPORTANT: You must load audio with the "VHS load audio" node from the [VideoHelperSuit](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git) node. 

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
## DeepFuze FaceSwap Node 🎭

![DeepFuze FaceSwap Node](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_FaceSwap_Node.jpg)
![DeepFuze FaceSwap Node example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_FaceSwap.jpg)

![DeepFuze FaceSwap example](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/AnimateDiff_00002-ezgif.com-video-to-gif-converter.gif)

This node Swaps, Enhances, and Restores faces from, video, and image. or higher quality export the IMAGE output as an image batch instead of a video combined, you can get up to 4k quality image size.

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

**Face Detector Model Summary Table** (RetinaFace provides higher quality by incorporating contextual information around the face, which helps in detecting faces under various conditions, such as occlusions, different scales, and poses.

| Feature                     | YOLOFace                       | RetinaFace                   | SCRFD                           | YuNet                           |
|-----------------------------|--------------------------------|------------------------------|---------------------------------|---------------------------------|
| **Architecture**            | Single-shot YOLO               | Single-stage RetinaNet       | Single-stage Cascade            | Lightweight Custom              |
| **Speed**                   | Very Fast                      | Moderate                     | Fast                            | Very Fast                       |
| **Accuracy**                | Good                           | Very High                    | High                            | Good                            |
| **Robustness**              | Moderate                       | Very High                    | High                            | Moderate                        |
| **Computational Efficiency**| High                           | Moderate                     | High                            | Very High                       |
| **Use Cases**               | Real-time, less complex scenes | High-precision, robust needs | Balanced, mobile/edge devices   | Mobile, embedded, real-time     |
| **Pros**                    | Speed                          | Accuracy, robustness         | Efficiency, accuracy            | Lightweight, efficient          |
| **Cons**                    | Accuracy trade-offs            | Computationally heavy        | Not the fastest                 | Less robust in complex scenes   |

----
### DeepFuze_TTS Node (Voice Cloning) 🎙️

![DeepFuze TTS_Node](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_TTS_Node.jpg)

**Languages:**

**DeepFuze_TTS voice cloning supports 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).**

This node is used to clone any voice from typed input. The audio file should be 10-15 seconds long for better results and should not have much noise. To avoid any sample rate error, load MP3 audio and only work with [AudioScheduler](https://github.com/a1lazydog/ComfyUI-AudioScheduler) node. We are working on developing a converter node to solve this issue. 

**Input Types:**
- `audio`: An instance of loaded audio data.
- `text`: Text to generate the cloned voice audio.

**Output Types:**
- `audio`: An instance of loaded audio data.

----
### DeepFuze Openai LLM Node 🤖

![DeepFuze Openai_Node](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Openai_LLM_Node.jpg)
![DeepFuze Openai_Node](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Openai_LLM.jpg)

The "LLM Integration" node is used to incorporate LLM (Language Model) into the voice cloning process. You can input your dialogue and configure parameters, and the AI-generated texts will be employed for voice cloning. Furthermore, you can utilize this node in place of ChatGPT to produce text from LLM or to ask any questions in the same manner as you would with ChatGPT. You can view the output of the DeepFuze_LLM by connecting the LLM_RESPONSE to the "Display Any" node from [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) this node also can be used for prompt generations and any nodes input texts. 

**Input Types:**
- `user_query`: Type your dialogues.

**Output Types:**
- `LLM_RESPONSE`: Outputs AI Generated texts.

**DeepFuze Openai LLM Features:**
- `model_name`: You can select from the available openai models.
- `api_key`: Add your API Key. (Your API Key will not be saved, each time you use this node you must manually enter it.
- `max_tokens`: is a parameter that limits the number of tokens in a model's response in OpenAI GPT APIs. It's used in requests made through GPT for Sheets and Docs, and in the ChatOpenAI() class. The default value for max_tokens is 4096 tokens, which is roughly equivalent to 3,000 words.
- `temperature`: controls the level of randomness and creativity in its responses. It's a hyper-parameter in Large Language Models (LLMs) that balances creativity and coherence in the generated text. The temperature setting is always a number between 0 and 1, with the default being 0.7:
0: Produces very straightforward, almost deterministic responses
1: Results in wildly varying responses
0.7: The default temperature for ChatGPT.
- `timeout` : set up time if the request takes too long to complete and the server closes the connection.

----
### DeepFuze Padding Node 👺

![DeepFuze Padding_Node](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Padding_Node.jpg)

**Input Types:**
- `image`: Provides a preview of the padding for the face mask.

**DeepFuze Padding Features:**
- `face_mask_padding_left` : Padding to left on the face while lipsyncing.
- `face_mask_padding_right` : Padding to the right on the face while lipsyncing.
- `face_mask_padding_bottom` : Padding to the bottom on the face while lipsyncing.
- `face_mask_padding_top` : Padding to the top on the face while lipsyncing.

----
### DeepFuze Save Audio (Playback) Node 🔉

![DeepFuze Save_Audio_Node](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Save_Audio_Node.jpg)
![DeepFuze Save_Audio](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/DeepFuze_Audio_Save.jpg)
This node is used to save the output of the "Voice Cloning" node. Additionally, you can trim the audio and play it back.

**Input Types:**
- `audio`: An instance of loaded audio data.

**DeepFuze Padding Features:**
- `METADATA` : Sting Metadata.
- `start_time` : Triming the start time.
- `end_time` : Triming the end time.
- `playback window` : Provides playback, save, and playback speed options.

----
### Basic Integrations

Voice Cloning + Lipsync Generation

![BasicWorkspace](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/VoiceCloning.jpg)

Voice Cloning + Lipsync Generation + FaceSwap

![BasicWorkspace](https://github.com/SamKhoze/ComfyUI-DeepFuze/blob/main/images/Lipsync_VoiceCloning_FaceSwap.jpg)

---
### Repository Structure

<sub>

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
</sub>

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

----
# Acknowledgements

This repository could not have been completed without the contributions from [FaceFusion](https://github.com/facefusion/facefusion), [InsightFace](https://github.com/deepinsight/insightface),[SadTalker](https://github.com/OpenTalker/SadTalker/tree/main), [Facexlib](https://github.com/xinntao/facexlib), [GFPGAN](https://github.com/TencentARC/GFPGAN), [GPEN](https://github.com/yangxy/GPEN), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [TTS](https://github.com/coqui-ai/TTS/tree/dev), [SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/), and [wav2lip](https://github.com/Rudrabha/Wav2Lip), 

1. Please carefully read and comply with the open-source license applicable to this code and models before using it. 
2. Please carefully read and comply with the intellectual property declaration applicable to this code and models before using it.
3. This open-source code runs completely offline and does not collect any personal information or other data. If you use this code to provide services to end-users and collect related data, please take necessary compliance measures according to applicable laws and regulations (such as publishing privacy policies, adopting necessary data security strategies, etc.). If the collected data involves personal information, user consent must be obtained (if applicable). 
4. It is prohibited to use this open-source code for activities that harm the legitimate rights and interests of others (including but not limited to fraud, deception, infringement of others' portrait rights, reputation rights, etc.), or other behaviors that violate applicable laws and regulations or go against social ethics and good customs (including providing incorrect or false information, terrorist, child/minors pornography and violent information, etc.). Otherwise, you may be liable for legal responsibilities.

The DeepFuze code is developed by Dr. Sam Khoze and his team. Feel free to use the DeepFuze code for personal, research, academic, and commercial purposes. You can create videos with this tool, but please make sure to follow local laws and use it responsibly. The developers will not be responsible for any misuse of the tool by users. 
