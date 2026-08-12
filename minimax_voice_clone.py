from pathlib import Path
import re
import tempfile

import requests


MINIMAX_VOICE_CLONE_BASE_URLS = {
    "global_en": "https://api.minimax.io",
    "cn_zh": "https://api.minimaxi.com",
}
MINIMAX_VOICE_CLONE_MODELS = (
    "speech-2.8-hd",
    "speech-2.6-hd",
    "speech-02-hd",
    "speech-01-hd",
)
MINIMAX_VOICE_CLONE_AUDIO_FORMATS = (".mp3", ".m4a", ".wav")
MINIMAX_VOICE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{6,254}[A-Za-z0-9]$")


class MiniMaxVoiceCloneClient:
    def __init__(self, api_key, region="global_en", timeout=60, session=None):
        if not api_key.strip():
            raise ValueError("MiniMax API key is required")
        if region not in MINIMAX_VOICE_CLONE_BASE_URLS:
            raise ValueError(f"Unsupported MiniMax region: {region}")
        self.api_key = api_key
        self.base_url = MINIMAX_VOICE_CLONE_BASE_URLS[region]
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _response_data(response):
        response.raise_for_status()
        data = response.json()
        base_response = data.get("base_resp", {})
        if base_response.get("status_code") != 0:
            message = base_response.get("status_msg", "Unknown MiniMax API error")
            raise RuntimeError(message)
        return data

    def upload_clone_audio(self, audio_path):
        audio_path = Path(audio_path)
        if audio_path.suffix.lower() not in MINIMAX_VOICE_CLONE_AUDIO_FORMATS:
            formats = ", ".join(MINIMAX_VOICE_CLONE_AUDIO_FORMATS)
            raise ValueError(f"Voice clone audio must use one of: {formats}")

        with audio_path.open("rb") as audio_file:
            response = self.session.post(
                f"{self.base_url}/v1/files/upload",
                headers=self.headers,
                data={"purpose": "voice_clone"},
                files={"file": (audio_path.name, audio_file, "application/octet-stream")},
                timeout=self.timeout,
            )
        data = self._response_data(response)
        file_id = data.get("file", {}).get("file_id")
        if file_id is None:
            raise RuntimeError("MiniMax audio upload did not return a file id")
        return file_id

    def create_voice_clone(
        self,
        file_id,
        voice_id,
        model,
        need_noise_reduction=False,
        need_volume_normalization=False,
    ):
        if model not in MINIMAX_VOICE_CLONE_MODELS:
            raise ValueError(f"Unsupported MiniMax voice clone model: {model}")
        if not MINIMAX_VOICE_ID_PATTERN.fullmatch(voice_id):
            raise ValueError(
                "MiniMax voice id must be 8-256 characters, start with a letter, "
                "end with a letter or number, and contain only letters, numbers, hyphens, or underscores"
            )
        response = self.session.post(
            f"{self.base_url}/v1/voice_clone",
            headers={**self.headers, "Content-Type": "application/json"},
            json={
                "file_id": file_id,
                "voice_id": voice_id,
                "model": model,
                "need_noise_reduction": need_noise_reduction,
                "need_volume_normalization": need_volume_normalization,
            },
            timeout=self.timeout,
        )
        data = self._response_data(response)
        cloned_voice_id = data.get("voice_id")
        if not cloned_voice_id:
            raise RuntimeError("MiniMax voice clone response did not return a voice id")
        return cloned_voice_id

    def clone_audio(
        self,
        audio_path,
        voice_id,
        model,
        need_noise_reduction=False,
        need_volume_normalization=False,
    ):
        file_id = self.upload_clone_audio(audio_path)
        return self.create_voice_clone(
            file_id,
            voice_id,
            model,
            need_noise_reduction,
            need_volume_normalization,
        )


class MiniMaxVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "api_key": ("STRING", {"default": ""}),
                "voice_id": ("STRING", {"default": "deepfuze_voice_01"}),
                "model": (list(MINIMAX_VOICE_CLONE_MODELS),),
            },
            "optional": {
                "region": (list(MINIMAX_VOICE_CLONE_BASE_URLS), {"default": "global_en"}),
                "need_noise_reduction": ("BOOLEAN", {"default": False}),
                "need_volume_normalization": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("VOICE_ID",)
    FUNCTION = "clone_voice"
    CATEGORY = "DeepFuze"

    def clone_voice(
        self,
        audio,
        api_key,
        voice_id,
        model,
        region="global_en",
        need_noise_reduction=False,
        need_volume_normalization=False,
    ):
        import torchaudio

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
            torchaudio.save(
                str(temporary_path),
                audio["waveform"][0],
                audio["sample_rate"],
            )
            cloned_voice_id = MiniMaxVoiceCloneClient(api_key, region=region).clone_audio(
                temporary_path,
                voice_id,
                model,
                need_noise_reduction,
                need_volume_normalization,
            )
            return (cloned_voice_id,)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
