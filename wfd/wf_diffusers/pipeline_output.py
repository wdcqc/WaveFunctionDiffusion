# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import torch
import numpy as np

from diffusers.utils import deprecate, logging
import PIL
from PIL import Image
from diffusers.utils import (
    BaseOutput,
)

@dataclass
class WaveFunctionDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Wave Function Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        waves (`np.ndarray`)
            List of denoised tile arrays ('waves') that contains probabilities for each possible tile as a numpy
            array of shape `(batch_size, height, width, num_tiles)`.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    waves: Optional[np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
