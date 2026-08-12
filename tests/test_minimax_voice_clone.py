import sys
from pathlib import Path
from types import SimpleNamespace

from minimax_voice_clone import MiniMaxVoiceClone, MiniMaxVoiceCloneClient


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(next(self.responses))


def test_clone_audio_uploads_file_and_creates_voice(tmp_path):
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"audio")
    session = FakeSession(
        [
            {
                "file": {"file_id": 123456789},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
            {
                "voice_id": "created-voice",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        ]
    )
    client = MiniMaxVoiceCloneClient("test-key", session=session)

    result = client.clone_audio(
        audio_path,
        "deepfuze-voice",
        "speech-2.8-hd",
        need_noise_reduction=True,
    )

    assert result == "created-voice"
    assert session.calls[0][0] == "https://api.minimax.io/v1/files/upload"
    assert session.calls[0][1]["data"] == {"purpose": "voice_clone"}
    assert session.calls[0][1]["files"]["file"][0] == "reference.wav"
    assert session.calls[1][0] == "https://api.minimax.io/v1/voice_clone"
    assert session.calls[1][1]["json"] == {
        "file_id": 123456789,
        "voice_id": "deepfuze-voice",
        "model": "speech-2.8-hd",
        "need_noise_reduction": True,
        "need_volume_normalization": False,
    }


def test_cn_region_uses_cn_endpoint(tmp_path):
    audio_path = tmp_path / "reference.m4a"
    audio_path.write_bytes(b"audio")
    session = FakeSession(
        [
            {
                "file": {"file_id": 987654321},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
            {
                "voice_id": "cn-voice",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        ]
    )
    client = MiniMaxVoiceCloneClient("test-key", region="cn_zh", session=session)

    client.clone_audio(audio_path, "cn-voice", "speech-2.6-hd")

    assert session.calls[0][0] == "https://api.minimaxi.com/v1/files/upload"
    assert session.calls[1][0] == "https://api.minimaxi.com/v1/voice_clone"


def test_upload_rejects_unsupported_audio_format(tmp_path):
    audio_path = tmp_path / "reference.ogg"
    audio_path.write_bytes(b"audio")
    client = MiniMaxVoiceCloneClient("test-key", session=FakeSession([]))

    try:
        client.upload_clone_audio(audio_path)
    except ValueError as error:
        assert "must use one of" in str(error)
    else:
        raise AssertionError("Unsupported audio format was accepted")


def test_clone_rejects_invalid_voice_id_before_request():
    client = MiniMaxVoiceCloneClient("test-key", session=FakeSession([]))

    try:
        client.create_voice_clone(123, "bad-id-", "speech-2.8-hd")
    except ValueError as error:
        assert "voice id" in str(error)
    else:
        raise AssertionError("Invalid voice id was accepted")


def test_node_saves_audio_clones_and_removes_temporary_file(monkeypatch):
    saved_paths = []
    cloned_paths = []

    def fake_save(path, waveform, sample_rate):
        saved_paths.append(Path(path))
        assert waveform == "waveform"
        assert sample_rate == 24000

    def fake_clone(client, audio_path, voice_id, model, noise_reduction, volume_normalization):
        cloned_paths.append(audio_path)
        assert voice_id == "deepfuze_voice_01"
        assert model == "speech-2.8-hd"
        assert noise_reduction is True
        assert volume_normalization is False
        return voice_id

    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(save=fake_save))
    monkeypatch.setattr(MiniMaxVoiceCloneClient, "clone_audio", fake_clone)

    result = MiniMaxVoiceClone().clone_voice(
        {"waveform": ["waveform"], "sample_rate": 24000},
        "test-key",
        "deepfuze_voice_01",
        "speech-2.8-hd",
        need_noise_reduction=True,
    )

    assert result == ("deepfuze_voice_01",)
    assert cloned_paths == saved_paths
    assert not cloned_paths[0].exists()
