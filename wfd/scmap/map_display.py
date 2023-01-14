# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import cv2, numpy as np, os, struct
from .data_processing import get_cv5_data, randomize_subtiles
from .tile_data import TILE_DATA_PATH

def process_tile_image(tileset, data_folder = TILE_DATA_PATH):
    tile_png = cv2.imread(os.path.join(data_folder, "{}.png".format(tileset)))
    width = tile_png.shape[1]
    tile_png = tile_png.reshape((-1, 32, width // 32, 32, 3)).transpose(0, 2, 1, 3, 4).reshape((-1, 32, 32, 3))
    return tile_png

def process_tile_mapping(tileset, data_folder = TILE_DATA_PATH):
    with open(os.path.join(data_folder, "{}.cv5.bin".format(tileset)), "rb") as fp:
        bin_data = fp.read()

    tile_to_png_index = np.array(struct.unpack("<" + "H" * (len(bin_data) // 2), bin_data))
    return tile_to_png_index

def get_map_image(tiles, tileset = None, tile_image = None, tile_mapping = None, pil = False, bgr = False):
    if tile_image is None:
        tile_image = process_tile_image(tileset)
    if tile_mapping is None:
        tile_mapping = process_tile_mapping(tileset)
    
    y, x = tiles.shape
    o = tile_mapping[tiles]
    i = tile_image[o]
    n = i.transpose((0, 2, 1, 3, 4))
    k = n.reshape(32 * y, 32 * x, 3)

    if not bgr:
        k = k[..., ::-1]
    
    if pil:
        from PIL import Image
        return Image.fromarray(k)
    return k
    
def show_map(tiles, tileset = None, tile_image = None, tile_mapping = None, display_handler = None):
    if display_handler is None:
        return
    m = get_map_image(tiles, tileset = tileset, tile_image = tile_image, tile_mapping = tile_mapping, bgr = True)
    display_handler(m)
    
def get_jpg(data):
    _, encoded_img = cv2.imencode('.jpg', data) 
    return encoded_img

def demo_map_image(tile_result, wfc_data_path, random_subtiles=True, rare_prob = 0.05, image_size = None):
    consts = np.load(wfc_data_path)
    tileset = consts["tileset"].tolist()
    tiles_sc = consts["gen_to_sc"][consts["shrink_to_gen"][tile_result]]
    if random_subtiles:
        cv5_data = get_cv5_data(tileset)
        randomize_subtiles(tiles_sc, tileset, cv5_data = cv5_data, rare_prob = rare_prob)
    image = get_map_image(
        tiles_sc,
        tileset = tileset,
        pil = True
    )
    if image_size is None:
        return image
    else:
        return image.resize(image_size)