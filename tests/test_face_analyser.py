import subprocess
import pytest

import deepfuze.globals
from deepfuze.download import conditional_download
from deepfuze.face_analyser import pre_check, clear_face_analyser, get_one_face
from deepfuze.typing import Face
from deepfuze.vision import read_static_image


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download('../../models/facefusion/examples',
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/source.jpg'
	])
	subprocess.run([ 'ffmpeg', '-i', '../../models/facefusion/examples/source.jpg', '-vf', 'crop=iw*0.8:ih*0.8', '../../models/facefusion/examples/source-80crop.jpg' ])
	subprocess.run([ 'ffmpeg', '-i', '../../models/facefusion/examples/source.jpg', '-vf', 'crop=iw*0.7:ih*0.7', '../../models/facefusion/examples/source-70crop.jpg' ])
	subprocess.run([ 'ffmpeg', '-i', '../../models/facefusion/examples/source.jpg', '-vf', 'crop=iw*0.6:ih*0.6', '../../models/facefusion/examples/source-60crop.jpg' ])


@pytest.fixture(autouse = True)
def before_each() -> None:
	deepfuze.globals.face_detector_score = 0.5
	deepfuze.globals.face_landmarker_score = 0.5
	deepfuze.globals.face_recognizer_model = 'arcface_inswapper'
	clear_face_analyser()


def test_get_one_face_with_retinaface() -> None:
	deepfuze.globals.face_detector_model = 'retinaface'
	deepfuze.globals.face_detector_size = '320x320'

	pre_check()
	source_paths =\
	[
		'../../models/facefusion/examples/source.jpg',
		'../../models/facefusion/examples/source-80crop.jpg',
		'../../models/facefusion/examples/source-70crop.jpg',
		'../../models/facefusion/examples/source-60crop.jpg'
	]
	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		face = get_one_face(source_frame)

		assert isinstance(face, Face)


def test_get_one_face_with_scrfd() -> None:
	deepfuze.globals.face_detector_model = 'scrfd'
	deepfuze.globals.face_detector_size = '640x640'

	pre_check()
	source_paths =\
	[
		'../../models/facefusion/examples/source.jpg',
		'../../models/facefusion/examples/source-80crop.jpg',
		'../../models/facefusion/examples/source-70crop.jpg',
		'../../models/facefusion/examples/source-60crop.jpg'
	]
	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		face = get_one_face(source_frame)

		assert isinstance(face, Face)


def test_get_one_face_with_yoloface() -> None:
	deepfuze.globals.face_detector_model = 'yoloface'
	deepfuze.globals.face_detector_size = '640x640'

	pre_check()
	source_paths =\
	[
		'../../models/facefusion/examples/source.jpg',
		'../../models/facefusion/examples/source-80crop.jpg',
		'../../models/facefusion/examples/source-70crop.jpg',
		'../../models/facefusion/examples/source-60crop.jpg'
	]
	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		face = get_one_face(source_frame)

		assert isinstance(face, Face)


def test_get_one_face_with_yunet() -> None:
	deepfuze.globals.face_detector_model = 'yunet'
	deepfuze.globals.face_detector_size = '640x640'

	pre_check()
	source_paths =\
	[
		'../../models/facefusion/examples/source.jpg',
		'../../models/facefusion/examples/source-80crop.jpg',
		'../../models/facefusion/examples/source-70crop.jpg',
		'../../models/facefusion/examples/source-60crop.jpg'
	]
	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		face = get_one_face(source_frame)

		assert isinstance(face, Face)
