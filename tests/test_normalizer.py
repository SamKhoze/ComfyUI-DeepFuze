from deepfuze.common_helper import is_linux, is_macos
from deepfuze.normalizer import normalize_output_path, normalize_padding, normalize_fps


def test_normalize_output_path() -> None:
	if is_linux() or is_macos():
		assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', '../../models/facefusion/examples/target-240p.mp4') == '../../models/facefusion/examples/target-240p.mp4'
		assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', '../../models/facefusion/examples').startswith('../../models/facefusion/examples/target-240p')
		assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', '../../models/facefusion/examples').endswith('.mp4')
		assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', '../../models/facefusion/examples/output.mp4') == '../../models/facefusion/examples/output.mp4'
	assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', '../../models/facefusion/examples/invalid') is None
	assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', '../../models/facefusion/invalid/output.mp4') is None
	assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', 'invalid') is None
	assert normalize_output_path('../../models/facefusion/examples/target-240p.mp4', None) is None
	assert normalize_output_path(None, '../../models/facefusion/examples/output.mp4') is None


def test_normalize_padding() -> None:
	assert normalize_padding([ 0, 0, 0, 0 ]) == (0, 0, 0, 0)
	assert normalize_padding([ 1 ]) == (1, 1, 1, 1)
	assert normalize_padding([ 1, 2 ]) == (1, 2, 1, 2)
	assert normalize_padding([ 1, 2, 3 ]) == (1, 2, 3, 2)
	assert normalize_padding(None) is None


def test_normalize_fps() -> None:
	assert normalize_fps(0.0) == 1.0
	assert normalize_fps(25.0) == 25.0
	assert normalize_fps(61.0) == 60.0
	assert normalize_fps(None) is None
