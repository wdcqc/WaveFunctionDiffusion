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
import hashlib

class SCInputMapsDataset(torch.utils.data.Dataset):
    def __init__(self, input_maps, w = 32, h = 32, img_size = (512, 512), shrink_range = True, cache_dir = None, min_size = 1):
        self.tileset = None
        self.w = w
        self.h = h
        self.img_size = img_size
        self.shrink_range = shrink_range
        self.cache_dir = cache_dir
        self.input_hash = self.get_input_hash(input_maps)
        self.min_size = min_size
        
        if cache_dir is None or not self.load_cache(cache_dir):
            self.process_input_maps(input_maps, merge_subtiles = True)
            if shrink_range:
                self.shrink_tile_range()
            if cache_dir is not None:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                self.save_cache(cache_dir)

    def get_input_hash(self, input_maps):
        file_str = "\n".join(sorted(input_maps)).replace("\\", "/")
        md5_digest = hashlib.md5(file_str.encode("utf-8")).hexdigest()
        return md5_digest[:12]

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

            if (map_size[0] - self.w + 1) * (map_size[1] - self.h + 1) < self.min_size:
                print(map_file, "size too small")
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

    def get_cache_path(self, cache_dir):
        ms = "_{}".format(self.min_size) if self.min_size > 1 else ""
        sr = "_sr" if self.shrink_range else ""
        return os.path.join(cache_dir, "{}_{}x{}{}{}.npz".format(self.input_hash, self.w, self.h, ms, sr))

    def save_cache(self, cache_dir):
        cache_path = self.get_cache_path(cache_dir)
        if self.shrink_range:
            shrink_dict = {
                "gen_to_shrink" : self.gen_to_shrink,
                "shrink_to_gen" : self.shrink_to_gen,
                "shrink_count" : self.shrink_count,
                "freqs_shrink" : self.freqs_shrink
            }
        else:
            shrink_dict = {}
            
        flattened_tiles = []
        map_sizes = []
        for tile in self.tiles:
            tile_size = (tile.shape[1], tile.shape[0])
            flattened_tile = tile.flatten()

            map_sizes.append(tile_size)
            flattened_tiles.append(flattened_tile)

        flattened_tiles = np.concatenate(flattened_tiles)
        map_sizes = np.array(map_sizes, dtype=int)
        
        np.savez_compressed(
            cache_path,
            tileset = self.tileset,
            sc_to_gen = self.sc_to_gen,
            gen_to_sc = self.gen_to_sc,
            tile_count = self.tile_count,
            freqs = self.freqs,
            conn_probs = self.conn_probs,
            map_names = self.map_names,
            map_sizes = map_sizes,
            sizes = self.sizes,
            flattened_tiles = flattened_tiles,
            **shrink_dict
        )

    def load_cache(self, cache_dir):
        cache_path = self.get_cache_path(cache_dir)
        if not os.path.isfile(cache_path):
            return False
        else:
            cache_data = np.load(cache_path)
            self.tileset = cache_data["tileset"]
            self.sc_to_gen = cache_data["sc_to_gen"]
            self.gen_to_sc = cache_data["gen_to_sc"]
            self.tile_count = int(cache_data["tile_count"])
            self.freqs = cache_data["freqs"],
            self.conn_probs = cache_data["conn_probs"]
            self.map_names = cache_data["map_names"].tolist()
            self.sizes = cache_data["sizes"]

            if self.shrink_range:
                self.gen_to_shrink = cache_data["gen_to_shrink"]
                self.shrink_to_gen = cache_data["shrink_to_gen"]
                self.shrink_count = int(cache_data["shrink_count"])
                self.freqs_shrink = cache_data["freqs_shrink"]
                self.shrink_mappings = self.gen_to_shrink, self.shrink_to_gen, self.shrink_count

            map_sizes = cache_data["map_sizes"]
            flattened_tiles = cache_data["flattened_tiles"]
            tiles = []
            k = 0
            for i, map_size in enumerate(map_sizes):
                map_x, map_y = map_size
                map_tile = flattened_tiles[k : k + map_x * map_y].reshape((map_y, map_x))
                k += map_x * map_y
                tiles.append(map_tile)
                assert self.sizes[i] == (map_x - self.w + 1) * (map_y - self.h + 1)

            self.tiles = tiles
            self.sizes_cumsum = np.cumsum(self.sizes)
            return True

    def __len__(self):
        return np.sum(self.sizes)

    def randomly_get_item(self, return_map_id=False):
        map_id = np.random.randint(0, len(self.tiles))
        point = np.random.randint(0, self.sizes[map_id])
        map_tiles = self.tiles[map_id]

        x, y = np.divmod(point, (map_tiles.shape[0] - self.h + 1))
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

        if return_map_id:
            return out_tiles, out_image, map_id
        else:
            return out_tiles, out_image

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
        brightness_fix ([`float` or `List[float]`]):
            Offsets the brightness of all training data. The first time I trained with original images
            and the resulting generations ended up darker than normal. This is a trick to increase
            the brightness back to normal levels.
        map_size ([`List[int]`]):
            The size of map cut out to make the image, measured in tiles, in (width, height) format.
        tileset_mix ([`List[str]` or None])
            If None, only one tileset is used.
            If list of str, it uses multiple tilesets from subfolders and replaces <tileset> in the
            instance prompt to each of tileset_keywords. Also requires brightness_fix to be a list
            of corresponding values.
        tileset_keywords ([`List[str]` or None])
            Only read if tileset_mix is set to a list of str. Replaces <tileset> in the instance
            prompt to each of the keywords.
        dataset_cache_dir ([`str` or None])
            Folder to cache the dataset.
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
        brightness_fix=0.10,
        map_size=(32, 32),
        tileset_mix=None,
        tileset_keywords=None,
        min_map_variations=1,
        dataset_cache_dir=None
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.append_map_name = append_map_name
        self.brightness_fix = brightness_fix
        self.tileset_mix = tileset_mix
        self.tileset_keywords = tileset_keywords
        self.min_map_variations = min_map_variations
        self.dataset_cache_dir = dataset_cache_dir

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")

        mapw, maph = map_size
        if tileset_mix is None:
            self.mixed = False
            mapdir = instance_data_root
            input_maps = [os.path.join(mapdir, fn) for fn in os.listdir(mapdir) if fn.endswith(".scx") or fn.endswith(".scm")]
            self.sc_dataset = SCInputMapsDataset(
                input_maps, 
                w = mapw,
                h = maph,
                img_size = (size, size),
                shrink_range = True,
                min_size = min_map_variations,
                cache_dir = dataset_cache_dir
            )
            self.num_instance_images = len(self.sc_dataset)
        else:
            self.mixed = True
            self.sc_datasets = []
            for tileset in tileset_mix:
                mapdir = os.path.join(instance_data_root, tileset)
                input_maps = [os.path.join(mapdir, fn) for fn in os.listdir(mapdir) if fn.endswith(".scx") or fn.endswith(".scm")]
                sc_dataset = SCInputMapsDataset(
                    input_maps,
                    w = mapw,
                    h = maph,
                    img_size = (size, size),
                    shrink_range = True,
                    min_size = min_map_variations,
                    cache_dir = dataset_cache_dir
                )
                self.sc_datasets.append(sc_dataset)
            self.num_instance_images = sum(len(ds) for ds in self.sc_datasets)

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
        if self.mixed:
            tileset_id = np.random.randint(0, len(self.sc_datasets))
            sc_dataset = self.sc_datasets[tileset_id]
            tiles, img, map_index = sc_dataset.randomly_get_item(return_map_id = True)
            instance_image = torch.as_tensor(img, dtype=torch.float) + self.brightness_fix[tileset_id]
            example["instance_images"] = instance_image
            if self.append_map_name:
                instance_prompt = self.instance_prompt.replace("<tileset>", self.tileset_keywords[tileset_id]) + ", " + self.sc_dataset.get_map_name(map_index)
            else:
                instance_prompt = self.instance_prompt.replace("<tileset>", self.tileset_keywords[tileset_id])
        else:
            tiles, img, map_index = self.sc_dataset.randomly_get_item(return_map_id = True)
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