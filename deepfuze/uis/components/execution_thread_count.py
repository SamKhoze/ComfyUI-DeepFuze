from typing import Optional
import gradio

import deepfuze.globals
import deepfuze.choices
from deepfuze import wording

EXECUTION_THREAD_COUNT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global EXECUTION_THREAD_COUNT_SLIDER

	EXECUTION_THREAD_COUNT_SLIDER = gradio.Slider(
		label = wording.get('uis.execution_thread_count_slider'),
		value = deepfuze.globals.execution_thread_count,
		step = deepfuze.choices.execution_thread_count_range[1] - deepfuze.choices.execution_thread_count_range[0],
		minimum = deepfuze.choices.execution_thread_count_range[0],
		maximum = deepfuze.choices.execution_thread_count_range[-1]
	)


def listen() -> None:
	EXECUTION_THREAD_COUNT_SLIDER.release(update_execution_thread_count, inputs = EXECUTION_THREAD_COUNT_SLIDER)


def update_execution_thread_count(execution_thread_count : int = 1) -> None:
	deepfuze.globals.execution_thread_count = execution_thread_count

