import subprocess
import pytest

from deepfuze.audio import get_audio_frame, read_static_audio
from deepfuze.download import conditional_download


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download('../../models/facefusion/examples',
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/source.mp3'
	])
	subprocess.run([ 'ffmpeg', '-i', '../../models/facefusion/examples/source.mp3', '../../models/facefusion/examples/source.wav' ])


def test_get_audio_frame() -> None:
	assert get_audio_frame('../../models/facefusion/examples/source.mp3', 25) is not None
	assert get_audio_frame('../../models/facefusion/examples/source.wav', 25) is not None
	assert get_audio_frame('invalid', 25) is None


def test_read_static_audio() -> None:
	assert len(read_static_audio('../../models/facefusion/examples/source.mp3', 25)) == 280
	assert len(read_static_audio('../../models/facefusion/examples/source.wav', 25)) == 280
	assert read_static_audio('invalid', 25) is None
