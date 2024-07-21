import os
import sys
import json
import subprocess
import numpy as np
import re
import cv2
import time
import itertools
import numpy as np
import datetime
from typing import List
import torch
import psutil
import torchaudio
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
from pydub import AudioSegment
from .utils import BIGMAX, DIMMAX, calculate_file_hash, get_sorted_dir_files_from_directory, get_audio, lazy_eval, hash_path, validate_path, strip_path
from PIL import Image, ImageOps
from comfy.utils import common_upscale, ProgressBar
from sys import platform
from scipy.io.wavfile import write
import folder_paths
from .utils import ffmpeg_path, get_audio, hash_path, validate_path, requeue_workflow, gifski_path, calculate_file_hash, strip_path
from comfy.utils import ProgressBar
from .llm_node import LLM_node
from .audio_playback import PlayBackAudio
from .audio_playback import SaveAudio

result_dir = os.path.join(folder_paths.get_output_directory(),"deepfuze")
audio_dir = os.path.join(folder_paths.get_input_directory(),"audio")

try:
    os.makedirs(result_dir)
except: pass
try:
    os.makedirs(audio_dir)
except: pass
audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']

path_cwd = "ComfyUI/custom_nodes/ComfyUI-DeepFuze" if os.path.isdir("ComfyUI/custom_nodes/ComfyUI-DeepFuze") else "custom_nodes/ComfyUI-DeepFuze"

video_extensions = ['webm', 'mp4', 'mkv', 'gif']


def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"


def target_size(width, height, force_size, custom_width, custom_height) -> tuple[int, int]:
    if force_size == "Custom":
        return (custom_width, custom_height)
    elif force_size == "Custom Height":
        force_size = "?x"+str(custom_height)
    elif force_size == "Custom Width":
        force_size = str(custom_width)+"x?"

    if force_size != "Disabled":
        force_size = force_size.split("x")
        if force_size[0] == "?":
            width = (width*int(force_size[1]))//height
            #Limit to a multple of 8 for latent conversion
            width = int(width)+4 & ~7
            height = int(force_size[1])
        elif force_size[1] == "?":
            height = (height*int(force_size[0]))//width
            height = int(height)+4 & ~7
            width = int(force_size[0])
        else:
            width = int(force_size[0])
            height = int(force_size[1])
    return (width, height)

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth, meta_batch=None, unique_id=None):
    video_cap = cv2.VideoCapture(strip_path(video))
    if not video_cap.isOpened():
        raise ValueError(f"{video} could not be loaded with cv.")
    pbar = ProgressBar(frame_load_cap) if frame_load_cap > 0 else None

    # extract video metadata
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # set video_cap to look at start_index frame
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    base_frame_time = 1 / fps
    prev_frame = None

    if force_rate == 0:
        target_frame_time = base_frame_time
    else:
        target_frame_time = 1/force_rate

    yield (width, height, fps, duration, total_frames, target_frame_time)

    time_offset=target_frame_time - base_frame_time
    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            # if didn't return frame, video has ended
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        # if not at start_index, skip doing anything with frame
        total_frame_count += 1
        if total_frame_count <= skip_first_frames:
            continue
        else:
            total_frames_evaluated += 1

        # if should not be selected, skip doing anything with frame
        if total_frames_evaluated%select_every_nth != 0:
            continue

        # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
        # follow up: can videos ever have an alpha channel?
        # To my testing: No. opencv has no support for alpha
        unused, frame = video_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert frame to comfyui's expected format
        # TODO: frame contains no exif information. Check if opencv2 has already applied
        frame = np.array(frame, dtype=np.float32)
        torch.from_numpy(frame).div_(255)
        if prev_frame is not None:
            inp  = yield prev_frame
            if inp is not None:
                #ensure the finally block is called
                return
        prev_frame = frame
        frames_added += 1
        if pbar is not None:
            pbar.update_absolute(frames_added, frame_load_cap)
        # if cap exists and we've reached it, stop processing frames
        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

def gen_format_widgets(video_format):
    for k in video_format:
        if k.endswith("_pass"):
            for i in range(len(video_format[k])):
                if isinstance(video_format[k][i], list):
                    item = [video_format[k][i]]
                    yield item
                    video_format[k][i] = item[0]
        else:
            if isinstance(video_format[k], list):
                item = [video_format[k]]
                yield item
                video_format[k] = item[0]

def get_video_formats():
    formats = []
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_name = format_name[:-5]
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
        with open(video_format_path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
        widgets = [w[0] for w in gen_format_widgets(video_format)]
        if (len(widgets) > 0):
            formats.append(["video/" + format_name, widgets])
        else:
            formats.append("video/" + format_name)
    return formats

def get_format_widget_defaults(format_name):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    results = {}
    for w in gen_format_widgets(video_format):
        if len(w[0]) > 2 and 'default' in w[0][2]:
            default = w[0][2]['default']
        else:
            if type(w[0][1]) is list:
                default = w[0][1][0]
            else:
                #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[0][1]]
        results[w[0][0]] = default
    return results


def apply_format_widgets(format_name, kwargs):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    print(video_format_path)
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in gen_format_widgets(video_format):
        print(w[0][0])
        assert(w[0][0] in kwargs)
        if len(w[0]) > 3:
            w[0] = Template(w[0][3]).substitute(val=kwargs[w[0][0]])
        else:
            w[0] = str(kwargs[w[0][0]])
    return video_format

def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def ffmpeg_process(args, video_format, video_metadata, file_path, env):

    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata = json.dumps(video_metadata)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        #metadata from file should  escape = ; # \ and newline
        metadata = metadata.replace("\\","\\\\")
        metadata = metadata.replace(";","\\;")
        metadata = metadata.replace("#","\\#")
        metadata = metadata.replace("=","\\=")
        metadata = metadata.replace("\n","\\\n")
        metadata = "comment=" + metadata
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(metadata)
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    #TODO: skip flush for increased speed
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                #Check if output file exists. If it does, the re-execution
                #will also fail. This obscures the cause of the error
                #and seems to never occur concurrent to the metadata issue
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode("utf-8"))
                #Res was not set
                print(err.decode("utf-8"), end="", file=sys.stderr)
                print("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode("utf-8"))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode("utf-8"), end="", file=sys.stderr)

def gifski_process(args, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n" \
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: " \
                        + resff.decode("utf-8") + '\ngifski: ' + resgs.decode("utf-8"))
    if len(resff) > 0:
        print(resff.decode("utf-8"), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode("utf-8"), end="", file=sys.stderr)
    #should always be empty as the quiet flag is passed
    if len(outgs) > 0:
        print(outgs.decode("utf-8"))

def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp)-2,0,-1):
        yield inp[i]


video_extensions = ['webm', 'mp4', 'mkv', 'gif']


def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"


def target_size(width, height, force_size, custom_width, custom_height) -> tuple[int, int]:
    if force_size == "Custom":
        return (custom_width, custom_height)
    elif force_size == "Custom Height":
        force_size = "?x"+str(custom_height)
    elif force_size == "Custom Width":
        force_size = str(custom_width)+"x?"

    if force_size != "Disabled":
        force_size = force_size.split("x")
        if force_size[0] == "?":
            width = (width*int(force_size[1]))//height
            #Limit to a multple of 8 for latent conversion
            width = int(width)+4 & ~7
            height = int(force_size[1])
        elif force_size[1] == "?":
            height = (height*int(force_size[0]))//width
            height = int(height)+4 & ~7
            width = int(force_size[0])
        else:
            width = int(force_size[0])
            height = int(force_size[1])
    return (width, height)

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth, meta_batch=None, unique_id=None):
    video_cap = cv2.VideoCapture(strip_path(video))
    if not video_cap.isOpened():
        raise ValueError(f"{video} could not be loaded with cv.")
    pbar = ProgressBar(frame_load_cap) if frame_load_cap > 0 else None

    # extract video metadata
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # set video_cap to look at start_index frame
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    base_frame_time = 1 / fps
    prev_frame = None

    if force_rate == 0:
        target_frame_time = base_frame_time
    else:
        target_frame_time = 1/force_rate

    yield (width, height, fps, duration, total_frames, target_frame_time)

    time_offset=target_frame_time - base_frame_time
    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            # if didn't return frame, video has ended
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        # if not at start_index, skip doing anything with frame
        total_frame_count += 1
        if total_frame_count <= skip_first_frames:
            continue
        else:
            total_frames_evaluated += 1

        # if should not be selected, skip doing anything with frame
        if total_frames_evaluated%select_every_nth != 0:
            continue

        # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
        # follow up: can videos ever have an alpha channel?
        # To my testing: No. opencv has no support for alpha
        unused, frame = video_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert frame to comfyui's expected format
        # TODO: frame contains no exif information. Check if opencv2 has already applied
        frame = np.array(frame, dtype=np.float32)
        torch.from_numpy(frame).div_(255)
        if prev_frame is not None:
            inp  = yield prev_frame
            if inp is not None:
                #ensure the finally block is called
                return
        prev_frame = frame
        frames_added += 1
        if pbar is not None:
            pbar.update_absolute(frames_added, frame_load_cap)
        # if cap exists and we've reached it, stop processing frames
        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

def load_video_cv(video: str, force_rate: int, force_size: str,
                  custom_width: int,custom_height: int, frame_load_cap: int,
                  skip_first_frames: int, select_every_nth: int,
                  meta_batch=None, unique_id=None, memory_limit_mb=None):
    print(meta_batch)
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                                 select_every_nth, meta_batch, unique_id)
        (width, height, fps, duration, total_frames, target_frame_time) = next(gen)

        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, fps, duration, total_frames, target_frame_time)

    else:
        (gen, width, height, fps, duration, total_frames, target_frame_time) = meta_batch.inputs[unique_id]

    if memory_limit_mb is not None:
        memory_limit *= 2 ** 20
    else:
        #TODO: verify if garbage collection should be performed here.
        #leaves ~128 MB unreserved for safety
        memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
    #space required to load as f32, exist as latent with wiggle room, decode to f32
    max_loadable_frames = int(memory_limit//(width*height*3*(4+4+1/10)))
    if meta_batch is not None:
        if meta_batch.frames_per_batch > max_loadable_frames:
            raise RuntimeError(f"Meta Batch set to {meta_batch.frames_per_batch} frames but only {max_loadable_frames} can fit in memory")
        gen = itertools.islice(gen, meta_batch.frames_per_batch)
    else:
        original_gen = gen
        gen = itertools.islice(gen, max_loadable_frames)

    #Some minor wizardry to eliminate a copy and reduce max memory by a factor of ~2
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if meta_batch is None:
        try:
            next(original_gen)
            raise RuntimeError(f"Memory limit hit after loading {len(images)} frames. Stopping execution.")
        except StopIteration:
            pass
    if len(images) == 0:
        raise RuntimeError("No frames generated")
    if force_size != "Disabled":
        new_size = target_size(width, height, force_size, custom_width, custom_height)
        if new_size[0] != width or new_size[1] != height:
            s = images.movedim(-1,1)
            s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
            images = s.movedim(1,-1)

    #Setup lambda for lazy audio capture
    audio =  get_audio(video, skip_first_frames * target_frame_time,
                               frame_load_cap*target_frame_time*select_every_nth)
    #Adjust target_frame_time for select_every_nth
    target_frame_time *= select_every_nth
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1/target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": images.shape[2],
        "loaded_height": images.shape[1],
    }
    print("images", type(images))
    return (images, len(images), audio, video_info)




class DeepFuzeFaceSwap:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats = get_video_formats()
        return {
            "required": {
                "source_images": ("IMAGE",),
                "target_images": ("IMAGE",),
                "enhancer": ("None,codeformer,gfpgan_1.2,gfpgan_1.3,gfpgan_1.4,gpen_bfr_256,gpen_bfr_512,gpen_bfr_1024,gpen_bfr_2048,restoreformer_plus_plus".split(","),{"default":'None'}),
                "faceswap_model":("blendswap_256,inswapper_128,inswapper_128_fp16,simswap_256,simswap_512_unofficial,uniface_256".split(","),{"default":"blendswap_256"}),
                "frame_enhancer": ("None,clear_reality_x4,lsdir_x4,nomos8k_sc_x4,real_esrgan_x2,real_esrgan_x2_fp16,real_esrgan_x4,real_esrgan_x4_fp16,real_hatgan_x4,span_kendata_x4,ultra_sharp_x4".split(","),{"default":'None'}),
                "face_detector_model" : ("retinaface,scrfd,yoloface,yunet".split(","),{"default":"yoloface"}),
                "reference_face_index" : ("INT",{"default":0,"min":0,"max":5,"step":1},{"default":0}),
                "face_mask_padding_left": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "face_mask_padding_right": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "face_mask_padding_bottom": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "face_mask_padding_top": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "device" : (["cpu","cuda","mps"],{"default":"cpu"}),
                "frame_rate": (
                    "FLOAT",
                    {"default": 25, "min": 1, "step": 1},
                ),

            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "deepfuze"}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,{"default":"video/h265-mp4"}),
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }


    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO",)
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info",)

    # RETURN_TYPES = ("VHS_FILENAMES",)
    # RETURN_NAMES = ("Filenames",)
    # OUTPUT_NODE = True
    CATEGORY = "DeepFuze"
    FUNCTION = "faceswampgenerate"

    def faceswampgenerate(
        self,
        source_images,
        target_images,
        enhancer,
        faceswap_model,
        frame_enhancer,
        face_detector_model,
        reference_face_index,
        face_mask_padding_left,
        face_mask_padding_right,
        face_mask_padding_bottom,
        face_mask_padding_top,
        device,
        frame_rate: int,
        audio="",
        loop_count: int = 0,
        filename_prefix="deepfuze",
        format="video/h265-mp4",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None
    ):
        images = target_images
        print(len(source_images),len(images))

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ("",)
        pbar = ProgressBar(len(images))

        first_image = images[0]
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = matcher.fullmatch(existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter

            # Increment the counter by 1 to get the next available value
            counter = max_counter + 1
            output_process = None

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, file)
        Image.fromarray(tensor_to_bytes(first_image)).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        print(format_type, format_ext)
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            frames = map(lambda x : Image.fromarray(tensor_to_bytes(x)), images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            #Acquire additional format_widget values
            kwargs = None
            if manual_format_widgets is None:
                if prompt is not None:
                    kwargs = prompt[unique_id]['inputs']
                else:
                    manual_format_widgets = {}
            if kwargs is None:
                kwargs = get_format_widget_defaults(format_ext)
                missing = {}
                for k in kwargs.keys():
                    if k in manual_format_widgets:
                        kwargs[k] = manual_format_widgets[k]
                    else:
                        missing[k] = kwargs[k]
                if len(missing) > 0:
                    print("Extra format values were not provided, the following defaults will be used: " + str(kwargs) + "\nThis is likely due to usage of ComfyUI-to-python. These values can be manually set by supplying a manual_format_widgets argument")
            kwargs["format"] = format
            kwargs['pix_fmt'] = 'yuv420p10le'
            kwargs['crf'] = 22
            kwargs["save_metadata"] = ["save_metadata", "BOOLEAN", {"default": True}]
            print(kwargs)
            video_format = apply_format_widgets(format_ext, kwargs)
            has_alpha = first_image.shape[-1] == 4
            dim_alignment = video_format.get("dim_alignment", 8)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                new_dims = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                            -first_image.shape[0] % dim_alignment + first_image.shape[0])
                dimensions = f"{new_dims[0]}x{new_dims[1]}"
                print("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = f"{first_image.shape[1]}x{first_image.shape[0]}"
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(len(images))]
            else:
                loop_args = []
            if pingpong:
                if meta_batch is not None:
                    print("pingpong is incompatible with batched output")
                images = to_pingpong(images)
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(len(images))]
            else:
                loop_args = []
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    #Potential solutions include keeping just output frames in
                    #memory or using 3 passes with intermediate file, but
                    #very long gifs probably shouldn't be encouraged
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                pre_pass_args = args[:13] + video_format['pre_pass']
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode("utf-8"))
            if "inputs_main_pass" in video_format:
                args = args[:13] + video_format['inputs_main_pass'] + args[13:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    output_process = gifski_process(args, video_format, file_path, env)
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                #TODO: Check if empty output breaks other custom nodes
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)

        i = 255. * source_images[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = None
        
        filename_with_batch_num = filename.replace("%batch_num%", "1")
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), pnginfo=metadata)
            
        faceswap_filename = os.path.join(result_dir,f"faceswap_{str(time.time()).replace('.','')}.mp4")
        command = [
            'python',
            './run.py',               # Script to run
            '--frame-processors',
            "face_swapper",
            "-s",
            os.path.join(full_output_folder, file),
            '-t',        # Argument: segmentation path
            output_files[-1],
            '--face-detector-model',
            face_detector_model,
            '-o',
            faceswap_filename,
            "--face-swapper-model",
            faceswap_model,
            '--reference-face-position',
            str(reference_face_index),
            '--face-mask-padding',
			str(face_mask_padding_top),
			str(face_mask_padding_bottom),
			str(face_mask_padding_left),
			str(face_mask_padding_right),
            '--headless'
        ]
            

        if device=="cuda":
            command.extend(['--execution-providers',"cuda"])
        elif device=="mps":
            command.extend(['--execution-providers',"coreml"])
        print(command)
        if platform == "win32":
            result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
        else:
            result = subprocess.run(command,cwd="custom_nodes/ComfyUI-DeepFuze",stdout=subprocess.PIPE)
        # audio_file = os.path.join(audio_dir,str(time.time()).replace(".","")+".wav")
        # subprocess.run(["ffmpeg","-i",faceswap_filename, audio_file, '-y'])
        # ffmpeg -i sample.avi -q:a 0 -map a sample.mp3
        # print(result.stdout.splitlines()[-1])
        if enhancer!="None":
            command = [
                'python',
                './run.py',               # Script to run
                '--frame-processors',
                "face_enhancer",
                "--face-enhancer-model",
                enhancer,
                "-t",
                faceswap_filename,
                '-o',
                faceswap_filename,
                '--headless'
            ]
            if device=="cuda":
                command.extend(['--execution-providers',"cuda"])
            elif device=="mps":
                command.extend(['--execution-providers',"coreml"])
            print(command)
            if platform == "win32":
                result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
            else:
                result = subprocess.run(command,cwd="custom_nodes/ComfyUI-DeepFuze",stdout=subprocess.PIPE)

        if frame_enhancer!="None":
            command = [
                'python',
                './run.py',               # Script to run
                '--frame-processors',
                "frame_enhancer",
                "--frame-enhancer-model",
                frame_enhancer,
                "-t",
                faceswap_filename,
                '-o',
                faceswap_filename,
                '--headless'
            ]
            print(command)
            if device=="cuda":
                command.extend(['--execution-providers',"cuda"])
            elif device=="mps":
                command.extend(['--execution-providers',"coreml"])
            if platform == "win32":
                result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
            else:
                result = subprocess.run(command,cwd="custom_nodes/ComfyUI-DeepFuze",stdout=subprocess.PIPE)
            # temp_file = "/".join(faceswap_filename.split("/")[:-1]) + "_"+faceswap_filename.split("/")[-1]
            # subprocess.run(["ffmpeg","-i",faceswap_filename,"-i",audio_file,"-c","copy","-map","0:v:0","-map","1:a:0",temp_file,'-y'])
            # faceswap_filename = temp_file

        print(result.stderr)
        if audio:
            audio_file = os.path.join(audio_dir,str(time.time()).replace(".","")+".wav")
            torchaudio.save(audio_file,audio["waveform"][0],audio["sample_rate"])
            subprocess.run(f"ffmpeg -i {faceswap_filename} -i {audio_file} -c copy {faceswap_filename.replace('.mp4','_.mp4')} -y".split())
            return load_video_cv(faceswap_filename.replace('.mp4','_.mp4'),0,'Disabled',512,512,0,0,1)
        return load_video_cv(faceswap_filename,0,'Disabled',512,512,0,0,1)





class DeepFuzeAdavance:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats = get_video_formats()
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "enhancer": ("None,codeformer,gfpgan_1.2,gfpgan_1.3,gfpgan_1.4,gpen_bfr_256,gpen_bfr_512,gpen_bfr_1024,gpen_bfr_2048,restoreformer_plus_plus".split(","),{"default":'None'}),
                "frame_enhancer": ("None,clear_reality_x4,lsdir_x4,nomos8k_sc_x4,real_esrgan_x2,real_esrgan_x2_fp16,real_esrgan_x4,real_esrgan_x4_fp16,real_hatgan_x4,span_kendata_x4,ultra_sharp_x4".split(","),{"default":'None'}),
                "face_mask_padding_left": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "face_mask_padding_right": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "face_mask_padding_bottom": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "face_mask_padding_top": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                "trim_frame_start": ("INT",{"default":0,"max":2000},),
                "trim_frame_end": ("INT",{"default":0,"max":2000},),
                "device" : (["cpu","cuda","mps"],{"default":"cpu"}),
                "frame_rate": (
                    "FLOAT",
                    {"default": 25, "min": 1, "step": 1},
                ),

            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "deepfuze"}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,{"default":"video/h265-mp4"}),
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }


    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO",)
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info",)

    # RETURN_TYPES = ("VHS_FILENAMES",)
    # RETURN_NAMES = ("Filenames",)
    # OUTPUT_NODE = True
    CATEGORY = "DeepFuze"
    FUNCTION = "lipsyncgenerate"

    def lipsyncgenerate(
        self,
        images,
        audio,
        enhancer,
        frame_enhancer,
        face_mask_padding_left,
        face_mask_padding_right,
        face_mask_padding_bottom,
        face_mask_padding_top,
        trim_frame_start,
        trim_frame_end,
        device,
        frame_rate: int,
        loop_count: int,
        filename_prefix="deepfuze",
        format="video/h265-mp4",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None
    ):
        print(enhancer,frame_rate,format)
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ("",)
        pbar = ProgressBar(len(images))
        trim_frame_end = len(images)-trim_frame_end 

        first_image = images[0]
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = matcher.fullmatch(existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter

            # Increment the counter by 1 to get the next available value
            counter = max_counter + 1
            output_process = None

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, file)
        Image.fromarray(tensor_to_bytes(first_image)).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        print(format_type, format_ext)
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            frames = map(lambda x : Image.fromarray(tensor_to_bytes(x)), images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            #Acquire additional format_widget values
            kwargs = None
            if manual_format_widgets is None:
                if prompt is not None:
                    kwargs = prompt[unique_id]['inputs']
                else:
                    manual_format_widgets = {}
            if kwargs is None:
                kwargs = get_format_widget_defaults(format_ext)
                missing = {}
                for k in kwargs.keys():
                    if k in manual_format_widgets:
                        kwargs[k] = manual_format_widgets[k]
                    else:
                        missing[k] = kwargs[k]
                if len(missing) > 0:
                    print("Extra format values were not provided, the following defaults will be used: " + str(kwargs) + "\nThis is likely due to usage of ComfyUI-to-python. These values can be manually set by supplying a manual_format_widgets argument")
            kwargs["format"] = format
            kwargs['pix_fmt'] = 'yuv420p10le'
            kwargs['crf'] = 22
            kwargs["save_metadata"] = ["save_metadata", "BOOLEAN", {"default": True}]
            print(kwargs)
            video_format = apply_format_widgets(format_ext, kwargs)
            has_alpha = first_image.shape[-1] == 4
            dim_alignment = video_format.get("dim_alignment", 8)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                new_dims = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                            -first_image.shape[0] % dim_alignment + first_image.shape[0])
                dimensions = f"{new_dims[0]}x{new_dims[1]}"
                print("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = f"{first_image.shape[1]}x{first_image.shape[0]}"
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(len(images))]
            else:
                loop_args = []
            if pingpong:
                if meta_batch is not None:
                    print("pingpong is incompatible with batched output")
                images = to_pingpong(images)
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(len(images))]
            else:
                loop_args = []
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    #Potential solutions include keeping just output frames in
                    #memory or using 3 passes with intermediate file, but
                    #very long gifs probably shouldn't be encouraged
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                pre_pass_args = args[:13] + video_format['pre_pass']
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode("utf-8"))
            if "inputs_main_pass" in video_format:
                args = args[:13] + video_format['inputs_main_pass'] + args[13:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    output_process = gifski_process(args, video_format, file_path, env)
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                #TODO: Check if empty output breaks other custom nodes
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)

        audio_file = os.path.join(audio_dir,str(time.time()).replace(".","")+".wav")
        torchaudio.save(audio_file,audio["waveform"][0],audio["sample_rate"])
        print(audio_file)
        filename = os.path.join(result_dir,f"{str(time.time()).replace('.','')}.mp4")
        enhanced_filename = os.path.join(result_dir,f"enhanced_{str(time.time()).replace('.','')}.mp4")
        command = [
            'python',
            './run.py',               # Script to run
            '--frame-processors',
            "lip_syncer",
            "-s",
            audio_file,
            '-t',        # Argument: segmentation path
            output_files[-1],
            '-o',
            filename,
            '--trim-frame-start',
            str(trim_frame_start),
            '--trim-frame-end',
            str(trim_frame_end),
            '--face-mask-padding',
			str(face_mask_padding_top),
			str(face_mask_padding_bottom),
			str(face_mask_padding_left),
			str(face_mask_padding_right),
            '--headless'
        ]
        if device=="cuda":
            command.extend(['--execution-providers',"cuda"])
        elif device=="mps":
            command.extend(['--execution-providers',"coreml"])
        print(command)
        if platform == "win32":
            result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
        else:
            result = subprocess.run(command,cwd="custom_nodes/ComfyUI-DeepFuze",stdout=subprocess.PIPE)
        # print(result.stdout.splitlines()[-1])
        if enhancer!="None":
            command = [
                'python',
                './run.py',               # Script to run
                '--frame-processors',
                "face_enhancer",
                "--face-enhancer-model",
                enhancer,
                "-t",
                filename,
                '-o',
                enhanced_filename,
                '--headless'
            ]
            if device=="cuda":
                command.extend(['--execution-providers',"cuda"])
            elif device=="mps":
                command.extend(['--execution-providers',"coreml"])
            print(command)
            if platform == "win32":
                result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
            else:
                result = subprocess.run(command,cwd="custom_nodes/ComfyUI-DeepFuze",stdout=subprocess.PIPE)
            filename = enhanced_filename

        if frame_enhancer!="None":
            command = [
                'python',
                './run.py',               # Script to run
                '--frame-processors',
                "frame_enhancer",
                "--frame-enhancer-model",
                frame_enhancer,
                "-t",
                filename,
                '-o',
                enhanced_filename,
                '--headless'
            ]
            print(command)
            if device=="cuda":
                command.extend(['--execution-providers',"cuda"])
            elif device=="mps":
                command.extend(['--execution-providers',"coreml"])
            if platform == "win32":
                result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
            else:
                result = subprocess.run(command,cwd="custom_nodes/ComfyUI-DeepFuze",stdout=subprocess.PIPE)
            temp_file = enhanced_filename.replace(".mp4","_.mp4") # "/".join(enhanced_filename.split("/")[:-1]) + "_"+enhanced_filename.split("/")[-1]
            subprocess.run(["ffmpeg","-i",enhanced_filename,"-i",audio_file,"-c","copy","-map","0:v:0","-map","1:a:0",temp_file,'-y'])
            filename = temp_file

        print(result.stderr)
        # try:
        #     os.system(f"rm {audio_file}")
        # except: pass
        return load_video_cv(filename,0,'Disabled',512,512,0,0,1)



import folder_paths
import torch
import time
import os
from pydub import AudioSegment

from scipy.io.wavfile import write




import numpy as np
from scipy.fft import fft

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




checkpoint_path_voice = os.path.join(folder_paths.models_dir,"deepfuze")
print(checkpoint_path_voice)

audio_path = os.path.join(folder_paths.get_input_directory(),"audio")
os.makedirs(audio_path,exist_ok=True)

class TTS_generation:


    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "llm_response": ("NEW_STRING",{"default":""}),
                "text": ("STRING",{
                    "multiline": True,
                    "default": ""
                }),
                "device": (["cpu","cuda","mps"],),
                "supported_language": ("English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko), Hindi (hi)".split(","),),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)  # Output type(s) of the node
    FUNCTION = "generate_audio"  # Entry-point method name

    CATEGORY = "DeepFuze"  # Category for the node in the UI

    def generate_audio(self, audio, text,device,supported_language,llm_response=""):
        
        if not llm_response and not text:
            raise ValueError("Please provide LLM_response or enter text")
        if llm_response:
            text = llm_response
        
        language = supported_language.split("(")[1][:-1]
        file_path = os.path.join(audio_path,str(time.time()).replace(".","")+".wav")
        torchaudio.save(file_path,audio["waveform"][0],audio["sample_rate"])
        command = [
            'python', 'tts_generation.py',
            '--model', checkpoint_path_voice,
            '--text', text,
            '--language', language,
            '--speaker_wav', file_path,
            '--output_file', file_path,
            '--device', device
        ]
        if platform == "win32":
            result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
        else:
            result = subprocess.run(command, cwd="custom_nodes/ComfyUI-DeepFuze",capture_output=True, text=True)

        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        audio = get_audio(file_path)
        return (audio,)



class DeepfuzePreview:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(self):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "face_mask_padding_left": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                    "face_mask_padding_right": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                    "face_mask_padding_bottom": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                    "face_mask_padding_top": ("INT",{"default":0,"min":0,"max":30,"step":1}),
                }
            }   
    RETURN_TYPES = ()
    FUNCTION = "test"  # Entry-point method name
    OUTPUT_NODE = True
    CATEGORY = "DeepFuze"  # Category for the node in the UI

    def test(self, images, face_mask_padding_left, face_mask_padding_right,face_mask_padding_bottom,face_mask_padding_top,  filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        print(filename)
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            command = [
                'python',
                'run.py',               # Script to run
                '--frame-processors',
                "face_debugger",
                "-t",
                os.path.join(full_output_folder, file),
                '-o',
                os.path.join(full_output_folder, "_"+file),
                '--face-mask-types',
                'box', 
                '--face-mask-padding',
                f'{str(face_mask_padding_top)}',
                f'{str(face_mask_padding_bottom)}',
                f'{str(face_mask_padding_left)}',
                f'{str(face_mask_padding_right)}',
                '--headless'
            ]
            print(command)
            if platform == "win32":
                result = subprocess.run(command,cwd=path_cwd,stdout=subprocess.PIPE)
            else:
                result = subprocess.run(command,cwd="custom_nodes/ComfyUI-DeepFuze",stdout=subprocess.PIPE)
            print(result.stdout)
            results.append({
                "filename": "_"+file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            return { "ui": { "images": results } }





NODE_CLASS_MAPPINGS = {
    "DeepFuzeAdavance": DeepFuzeAdavance,
    "DeepFuzeFaceSwap": DeepFuzeFaceSwap,
    "TTS_generation":TTS_generation,
    "LLM_node": LLM_node,
    "PlayBackAudio": PlayBackAudio,
	"DeepFuze Save":SaveAudio,
	"DeepfuzePreview":DeepfuzePreview
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepFuzeAdavance": "DeepFuze Lipsync",
	"DeepFuzeFaceSwap": "DeepFuze FaceSwap",
    "TTS_generation":"DeepFuze TTS",
    "LLM_node": "DeepFuze Openai LLM",
    "PlayBackAudio": "Play Audio",
	"DeepFuze Save":"DeepFuze Save Audio",
	"DeepfuzePreview": "DeepFuze Padding Preview"
}
