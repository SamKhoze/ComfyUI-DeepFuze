from typing import Optional
import gradio

import deepfuze.globals
import deepfuze.choices
from deepfuze import wording

EXECUTION_QUEUE_COUNT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global EXECUTION_QUEUE_COUNT_SLIDER

	EXECUTION_QUEUE_COUNT_SLIDER = gradio.Slider(
		label = wording.get('uis.execution_queue_count_slider'),
		value = deepfuze.globals.execution_queue_count,
		step = deepfuze.choices.execution_queue_count_range[1] - deepfuze.choices.execution_queue_count_range[0],
		minimum = deepfuze.choices.execution_queue_count_range[0],
		maximum = deepfuze.choices.execution_queue_count_range[-1]
	)


def listen() -> None:
	EXECUTION_QUEUE_COUNT_SLIDER.release(update_execution_queue_count, inputs = EXECUTION_QUEUE_COUNT_SLIDER)


def update_execution_queue_count(execution_queue_count : int = 1) -> None:
	deepfuze.globals.execution_queue_count = execution_queue_count
