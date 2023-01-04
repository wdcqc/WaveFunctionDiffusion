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

import warnings

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

import numpy as np


class TilesetData(ConfigMixin, SchedulerMixin):
    """
    Do not use this (the generated JSON takes 1GB space)
    This is left here to warn anyone who thinks this is a good idea
    """

    config_name = "tileset_data.json"

    @register_to_config
    def __init__(
        self,
        tileset: str = "platform",
        w: int = 32,
        h: int = 32,
        img_size: tuple[int] = (512, 512),
        sc_to_gen: np.ndarray = None,
        gen_to_sc: np.ndarray = None,
        tile_count: int = 0,
        freqs: np.ndarray = None,
        conn_probs: np.ndarray = None,
        shrink_range: bool = False,
        gen_to_shrink: np.ndarray = None,
        shrink_to_gen: np.ndarray = None,
        shrink_count: int = 0,
    ):
        self.w = w
        self.h = h
        self.img_size = img_size
        self.tileset = tileset
        self.sc_to_gen = sc_to_gen
        self.gen_to_sc = gen_to_sc
        self.tile_count = tile_count
        self.freqs = freqs
        self.conn_probs = conn_probs
        self.shrink_range = shrink_range
        self.gen_to_shrink = gen_to_shrink
        self.shrink_to_gen = shrink_to_gen
        self.shrink_count = shrink_count

    def load_npz_config(self, npz_file):
        data = np.load(npz_file)
        self.w = data["w"]
        self.h = data["h"]
        self.img_size = data["img_size"]
        self.tileset = data["tileset"].tolist()
        self.sc_to_gen = data["sc_to_gen"]
        self.gen_to_sc = data["gen_to_sc"]
        self.tile_count = data["tile_count"]
        self.freqs = data["freqs"]
        self.conn_probs = data["conn_probs"]
        self.shrink_range = data["shrink_range"]
        self.gen_to_shrink = data["gen_to_shrink"]
        self.shrink_to_gen = data["shrink_to_gen"]
        self.shrink_count = data["shrink_count"]