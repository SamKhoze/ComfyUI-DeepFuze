import shutil
import pytest

from deepfuze.common_helper import is_windows
from deepfuze.download import conditional_download
from deepfuze.filesystem import get_file_size, is_file, is_directory, is_audio, has_audio, is_image, has_image, is_video, filter_audio_paths, filter_image_paths, list_directory, sanitize_path_for_windows


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download('../../models/facefusion/examples',
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/source.jpg',
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/source.mp3',
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/target-240p.mp4'
	])
	shutil.copyfile('../../models/facefusion/examples/source.jpg', '../../models/facefusion/examples/söurce.jpg')


def test_get_file_size() -> None:
	assert get_file_size('../../models/facefusion/examples/source.jpg') > 0
	assert get_file_size('invalid') == 0


def test_is_file() -> None:
	assert is_file('../../models/facefusion/examples/source.jpg') is True
	assert is_file('../../models/facefusion/examples') is False
	assert is_file('invalid') is False


def test_is_directory() -> None:
	assert is_directory('../../models/facefusion/examples') is True
	assert is_directory('../../models/facefusion/examples/source.jpg') is False
	assert is_directory('invalid') is False


def test_is_audio() -> None:
	assert is_audio('../../models/facefusion/examples/source.mp3') is True
	assert is_audio('../../models/facefusion/examples/source.jpg') is False
	assert is_audio('invalid') is False


def test_has_audio() -> None:
	assert has_audio([ '../../models/facefusion/examples/source.mp3' ]) is True
	assert has_audio([ '../../models/facefusion/examples/source.mp3', '../../models/facefusion/examples/source.jpg' ]) is True
	assert has_audio([ '../../models/facefusion/examples/source.jpg', '../../models/facefusion/examples/source.jpg' ]) is False
	assert has_audio([ 'invalid' ]) is False


def test_is_image() -> None:
	assert is_image('../../models/facefusion/examples/source.jpg') is True
	assert is_image('../../models/facefusion/examples/target-240p.mp4') is False
	assert is_image('invalid') is False


def test_has_image() -> None:
	assert has_image([ '../../models/facefusion/examples/source.jpg' ]) is True
	assert has_image([ '../../models/facefusion/examples/source.jpg', '../../models/facefusion/examples/source.mp3' ]) is True
	assert has_image([ '../../models/facefusion/examples/source.mp3', '../../models/facefusion/examples/source.mp3' ]) is False
	assert has_image([ 'invalid' ]) is False


def test_is_video() -> None:
	assert is_video('../../models/facefusion/examples/target-240p.mp4') is True
	assert is_video('../../models/facefusion/examples/source.jpg') is False
	assert is_video('invalid') is False


def test_filter_audio_paths() -> None:
	assert filter_audio_paths([ '../../models/facefusion/examples/source.jpg', '../../models/facefusion/examples/source.mp3' ]) == [ '../../models/facefusion/examples/source.mp3' ]
	assert filter_audio_paths([ '../../models/facefusion/examples/source.jpg', '../../models/facefusion/examples/source.jpg' ]) == []
	assert filter_audio_paths([ 'invalid' ]) == []


def test_filter_image_paths() -> None:
	assert filter_image_paths([ '../../models/facefusion/examples/source.jpg', '../../models/facefusion/examples/source.mp3' ]) == [ '../../models/facefusion/examples/source.jpg' ]
	assert filter_image_paths([ '../../models/facefusion/examples/source.mp3', '../../models/facefusion/examples/source.mp3' ]) == []
	assert filter_audio_paths([ 'invalid' ]) == []


def test_list_directory() -> None:
	assert list_directory('../../models/facefusion/examples')
	assert list_directory('../../models/facefusion/examples/source.jpg') is None
	assert list_directory('invalid') is None


def test_sanitize_path_for_windows() -> None:
	if is_windows():
		assert sanitize_path_for_windows('../../models/facefusion/examples/söurce.jpg') == 'ASSETS~1/examples/SURCE~1.JPG'
		assert sanitize_path_for_windows('invalid') is None
