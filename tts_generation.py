import os
import torch
import argparse
import sys
from sys import platform

if platform == "win32":
	sys.path.append('./')

from TTS.api import TTS

def main():
    parser = argparse.ArgumentParser(description="Run TTS with specified parameters.")
    
    parser.add_argument('--model', type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="The TTS model to use.")
    parser.add_argument('--text', type=str, required=True, help="The text to be converted to speech.")
    parser.add_argument('--speaker_wav', type=str, required=True, help="The path to the speaker's wav file for voice cloning.")
    parser.add_argument('--language', type=str, default="en", help="The language of the text.")
    parser.add_argument('--output_file', type=str, default="output.wav", help="The output file path for the synthesized speech.")
    parser.add_argument('--device', type=str, choices=["cpu", "mps","cuda"], default="cpu" if torch.cuda.is_available() else "cpu", help="The device to run the model on.")
    
    args = parser.parse_args()
	
    if args.device == "cuda":
        device="cuda"
    else:
        device="cpu"

    # Init TTS
    tts = TTS(model_path=args.model,config_path=os.path.join(args.model,"config.json")).to(device)

    # Run TTS and save to file
    tts.tts_to_file(text=args.text, speaker_wav=args.speaker_wav, language=args.language, file_path=args.output_file)

if __name__ == "__main__":
    main()
