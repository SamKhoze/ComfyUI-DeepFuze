#!/usr/bin/env python3
import sys
from sys import platform

if platform == "win32":
	sys.path.append('./')
from deepfuze import core

if __name__ == '__main__':
	core.cli()
