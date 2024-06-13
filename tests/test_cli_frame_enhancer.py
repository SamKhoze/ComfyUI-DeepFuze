import subprocess
import sys
import pytest

from deepfuze.download import conditional_download


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download('../../models/facefusion/examples',
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/source.jpg',
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/target-240p.mp4'
	])
	subprocess.run([ 'ffmpeg', '-i', '../../models/facefusion/examples/target-240p.mp4', '-vframes', '1', '../../models/facefusion/examples/target-240p.jpg' ])


def test_enhance_frame_to_image() -> None:
	commands = [ sys.executable, 'run.py', '--frame-processors', 'frame_enhancer', '-t', '../../models/facefusion/examples/target-240p.jpg', '-o', '../../models/facefusion/examples/test_enhance_frame_to_image.jpg', '--headless' ]
	run = subprocess.run(commands, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)

	assert run.returncode == 0
	assert 'image succeed' in run.stdout.decode()


def test_enhance_frame_to_video() -> None:
	commands = [ sys.executable, 'run.py', '--frame-processors', 'frame_enhancer', '-t', '../../models/facefusion/examples/target-240p.mp4', '-o', '../../models/facefusion/examples/test_enhance_frame_to_video.mp4', '--trim-frame-end', '10', '--headless' ]
	run = subprocess.run(commands, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)

	assert run.returncode == 0
	assert 'video succeed' in run.stdout.decode()
