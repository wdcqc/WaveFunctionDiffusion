# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

from ..scmap.data_processing import (
    get_shrink_mapping,
    get_generator_mapping,
    get_connections_and_freqs,
)
from ..scmap.map_data import get_map_data, get_tile_data
from ..scmap.map_display import get_map_image
import cv2
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import einops as eo
import os
import re
from tqdm.autonotebook import tqdm

class SCInputMapsDataset(torch.utils.data.Dataset):
    def __init__(self, input_maps, w = 32, h = 32, img_size = (512, 512), shrink_range = True):
        self.tileset = None
        self.w = w
        self.h = h
        self.img_size = img_size
        
        self.process_input_maps(input_maps, merge_subtiles = True)
        if shrink_range:
            self.shrink_tile_range()
        else:
            self.shrink_range = False

    def process_input_maps(self, input_maps, merge_subtiles = True):
        tile_pool = []
        size_pool = []
        name_pool = []
        
        all_conn_probs = None
        all_freqs = None
        for map_file in tqdm(input_maps):
            map_data = get_map_data(map_file)
            try:
                tileset, map_size, tile_data = get_tile_data(map_data)
            except AssertionError as e:
                print(map_file, "assert error")
                continue

            if self.tileset is None:
                self.tileset = tileset
                self.sc_to_gen, self.gen_to_sc, self.tile_count = get_generator_mapping(
                    self.tileset, merge_subtiles = merge_subtiles
                )
            elif self.tileset != tileset:
                raise ValueError("Not the same tileset: {}".format(map_file))
                
            tile_data_gen = self.sc_to_gen[tile_data]
            map_conn_probs, map_freqs = get_connections_and_freqs(tile_data_gen, self.tile_count)

            if all_conn_probs is None:
                all_conn_probs = map_conn_probs
                all_freqs = map_freqs
            else:
                all_conn_probs += map_conn_probs
                all_freqs += map_freqs
                
            name_pool.append(self.normalize_map_name(map_file))
            tile_pool.append(tile_data_gen)
            size_pool.append((map_size[0] - self.w + 1) * (map_size[1] - self.h + 1))
                
        all_freqs /= np.sum(all_freqs)
        all_conn_probs /= np.maximum(1, all_conn_probs.sum(axis = 2, keepdims = True))
        
        self.freqs = all_freqs
        self.conn_probs = all_conn_probs
        
        self.tiles = tile_pool
        self.sizes = size_pool
        self.map_names = name_pool
        self.sizes_cumsum = np.cumsum(self.sizes)
        
    def shrink_tile_range(self):
        self.shrink_range = True
        self.shrink_mappings = get_shrink_mapping(self.freqs, is_freq_table = True)
        self.gen_to_shrink, self.shrink_to_gen, self.shrink_count = self.shrink_mappings
        self.freqs_shrink = self.freqs[self.shrink_to_gen]
        
        self.tiles = [self.gen_to_shrink[t] for t in self.tiles]
        
    def normalize_map_name(self, name):
        name = re.sub(r"\.sc(m|x)", "", name)
        name = re.sub(r"\(\([0-9]+\)\)", "", name)
        name = re.sub(r"\([0-9onrpbgNOPRBG_]+\)", "", name)
        name = re.sub(r"（4）", "", name)
        name = re.sub(r"\([0-9]+\)", "", name)
        name = re.sub(r"\[[0-9]+\]", "", name.strip())
        name = re.sub(r"_", " ", name)
        name = re.sub(r"&#x27;", "'", name)
        name = re.sub(r"&amp;", "&", name)
        name = re.sub(r"\(Version [0-9]+\)", "", name)
        name = re.sub(r"(iCCup|iccup|ICCup|_WCG|WCG |WGTour8-obs-|WGTour8-|\[WGTour-obs\]|xPGT-| Obs| obs| Ob| ob)", "", name.strip())
        name = re.sub(r"[0-9]+\.[0-9]+$", "", name.strip())
        name = re.sub(r"^[0-9]+ ", "", name.strip())
        return name.strip()

    def save_wfc(self, data_name):
        if self.shrink_range:
            conn_probs_shrink = self.conn_probs[self.shrink_to_gen, :, :][:, :, self.shrink_to_gen]
            wfc_mats = [conn_probs_shrink[:, 2] > 0, conn_probs_shrink[:, 3] > 0]
        else:
            wfc_mats = [self.conn_probs[:, 2] > 0, self.conn_probs[:, 3] > 0]
        np.savez_compressed(
            data_name,
            w = self.w,
            h = self.h,
            img_size = self.img_size,
            tileset = self.tileset,
            sc_to_gen = self.sc_to_gen,
            gen_to_sc = self.gen_to_sc,
            tile_count = self.tile_count,
            freqs = self.freqs,
            conn_probs = self.conn_probs,
            shrink_range = self.shrink_range,
            gen_to_shrink = self.gen_to_shrink,
            shrink_to_gen = self.shrink_to_gen,
            shrink_count = self.shrink_count,
            wfc_mats = wfc_mats
        )

    def get_map_name(self, i):
        map_id = np.searchsorted(self.sizes_cumsum, i + 1)
        return self.map_names[map_id]

    def __len__(self):
        return np.sum(self.sizes)

    def __getitem__(self, i):
        map_id = np.searchsorted(self.sizes_cumsum, i + 1)
        map_tiles = self.tiles[map_id]
        if map_id > 0:
            i -= self.sizes_cumsum[map_id - 1]
        
        x, y = np.divmod(i, (map_tiles.shape[0] - self.h + 1))
        fragment_tiles = map_tiles[y : y + self.h, x : x + self.w]
        
        if self.shrink_range:
            image_tiles = get_map_image(
                self.gen_to_sc[self.shrink_to_gen[fragment_tiles]],
                tileset = self.tileset,
                pil = False
            )
        else:
            image_tiles = get_map_image(
                self.gen_to_sc[fragment_tiles],
                tileset = self.tileset,
                pil = False
            )
            
        image_tiles = cv2.resize(image_tiles, self.img_size)
        
        out_tiles = fragment_tiles
        out_image = image_tiles.astype(np.float32) / 127.5 - 1
        out_image = eo.rearrange(out_image, "h w c -> c h w")
        return out_tiles, out_image

class SCRandomMapsDataset(torch.utils.data.Dataset):
    def __init__(self, tileset, shrink_mappings=None, freqs=None, w = 32, h = 32, img_size = (512, 512), num_samples=10000):
        self.tileset = tileset
        self.num_samples = num_samples
        self.w = w
        self.h = h
        self.img_size = img_size
        
        self.sc_to_gen, self.gen_to_sc, self.tile_count = get_generator_mapping(
            self.tileset, merge_subtiles = True
        )
        
        if shrink_mappings is None:
            self.shrink_range = False
        else:
            self.shrink_range = True
            self.gen_to_shrink, self.shrink_to_gen, self.shrink_count = shrink_mappings
            
        self.freqs = freqs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        if not self.shrink_range:
            if self.freqs is not None:
                fragment_tiles = np.random.choice(np.arange(self.tile_count), size=(self.h, self.w), p=self.freqs)
            else:
                fragment_tiles = np.random.randint(0, self.tile_count, size=(self.h, self.w))
            image_tiles = get_map_image(
                self.gen_to_sc[fragment_tiles],
                tileset = self.tileset,
                pil = False
            )
        else:
            if self.freqs is not None:
                fragment_tiles = np.random.choice(np.arange(self.shrink_count), size=(self.h, self.w), p=self.freqs)
            else:
                fragment_tiles = np.random.randint(0, self.shrink_count, size=(self.h, self.w))
            image_tiles = get_map_image(
                self.gen_to_sc[self.shrink_to_gen[fragment_tiles]],
                tileset = self.tileset,
                pil = False
            )
            
        image_tiles = cv2.resize(image_tiles, self.img_size)
        
        out_image = image_tiles.astype(np.float32) / 127.5 - 1
        out_image = eo.rearrange(out_image, "h w c -> c h w")
        return fragment_tiles, out_image

class SCInputMapsDreamBoothDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.

    Parameters:
        append_map_name ([`bool`]):
            Adds the map name (ex: Andromeda, Lost Temple) at the end of each prompt with a clause.
            "isometric scspace terrain" -> "isometric scspace terrain, Lost Temple"
        brightness_fix ([`float`]):
            Offsets the brightness of all training data. The first time I trained with original images
            and the resulting generations ended up darker than normal. This is a trick to increase
            the brightness back to normal levels.
        (For other parameters please refer to the original DreamBooth documentation)
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        append_map_name=False,
        brightness_fix=0.10
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.append_map_name = append_map_name
        self.brightness_fix = brightness_fix

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")

        mapdir = instance_data_root
        input_maps = [os.path.join(mapdir, fn) for fn in os.listdir(mapdir) if fn.endswith(".scx") or fn.endswith(".scm")]
        self.sc_dataset = SCInputMapsDataset(
            input_maps, 
            w = 32,
            h = 32,
            img_size = (size, size),
            shrink_range = True
        )
        self.map_indices = np.arange(len(self.sc_dataset))
        np.random.shuffle(self.map_indices)

        self.num_instance_images = len(self.sc_dataset)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        map_index = self.map_indices[index]
        tiles, img = self.sc_dataset[map_index]
        instance_image = torch.as_tensor(img, dtype=torch.float) + self.brightness_fix
        example["instance_images"] = instance_image
        if self.append_map_name:
            instance_prompt = self.instance_prompt + ", " + self.sc_dataset.get_map_name(map_index)
        else:
            instance_prompt = self.instance_prompt
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example