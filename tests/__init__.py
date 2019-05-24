# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import sys

# Add package to path.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '..')))

# The below imports are there to not have CI fail.
# noinspection PyUnresolvedReferences
import torch
# noinspection PyUnresolvedReferences
import tensorflow
