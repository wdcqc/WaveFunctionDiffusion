# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import numpy as np, json, struct, os
from .map_data import get_map_data, get_tile_data, replace_tile_data, get_default_output_map_data
from tqdm.auto import tqdm
from .tile_data import TILE_DATA_PATH

CV5_PATHF = os.path.join(TILE_DATA_PATH, "{}_v2.npz")
MAPPING_PATHF = os.path.join(TILE_DATA_PATH, "{}_mapping.npz")

doodad_start_each_tileset = {
    "badlands" : 1024,
    "platform" : 933,
    "install" : 1024,
    "ashworld" : 1024,
    "jungle" : 1024,
    "desert" : 770,
    "ice" : 1024,
    "twilight" : 797,
}

def l1_normalize(vec):
    s = np.sum(vec)
    if s == 0 or s == 1:
        return vec
    vec /= s
    return vec

def get_cv5_data(tileset, pathf = CV5_PATHF):
    data = np.load(pathf.format(tileset))["data"].tolist()
    cv5_data = json.loads(data)
    return cv5_data

def get_null_mapping(tileset):
    cv5_data = get_cv5_data(tileset)
    nulls = [d["null"] for d in cv5_data]
    non_nulls = np.logical_not(nulls)
    
    sc_to_gen = np.full((len(cv5_data),), -1, dtype = int)
    sc_to_gen[nulls] = 0
    sc_to_gen[non_nulls] = np.arange(1, 1 + np.sum(non_nulls))
    
    gen_to_sc = np.concatenate([[1], np.where(non_nulls)[0]])
    return sc_to_gen, gen_to_sc, len(gen_to_sc)

def get_generator_mapping(tileset, pathf = MAPPING_PATHF, merge_subtiles = True):
    if not merge_subtiles:
        return get_null_mapping(tileset)
    tileset_data = np.load(pathf.format(tileset))
    sc_to_gen = tileset_data["sc_to_gen"]
    gen_to_sc = tileset_data["gen_to_sc"]
    tile_count = int(tileset_data["total_count"])
    return sc_to_gen, gen_to_sc, tile_count

def get_shrink_mapping(data, is_freq_table = False):
    if is_freq_table:
        unique_vals = np.where(data > 0)[0]
        old_range = np.zeros(len(data), dtype=int)
    else:
        unique_vals = np.unique(data)
        old_range = np.zeros((data.max() + 1,), dtype=int)
    if 0 not in unique_vals:
        unique_vals = np.concatenate([[0], unique_vals])
    for i, v in enumerate(unique_vals):
        old_range[v] = i
    new_range = np.arange(len(unique_vals))
    return old_range, unique_vals, len(unique_vals)

def get_connections_and_freqs(data, tile_count):
    max_x = data.shape[1]
    max_y = data.shape[0]
    connections = np.zeros((tile_count, 4, tile_count), dtype = np.float32)
    freqs = np.zeros((tile_count,), dtype = np.float32)
    for y, line in enumerate(data):
        for x, v in enumerate(line):
            if v == 0:
                continue
                
            freqs[v] += 1
            if x > 0:
                next_grid = y, x-1
                nv = data[next_grid]
                if nv > 0:
                    connections[v, 0, nv] += 1

            if y > 0:
                next_grid = y-1, x
                nv = data[next_grid]
                if nv > 0:
                    connections[v, 1, nv] += 1

            if x < max_x - 1:
                next_grid = y, x+1
                nv = data[next_grid]
                if nv > 0:
                    connections[v, 2, nv] += 1

            if y < max_y - 1:
                next_grid = y+1, x
                nv = data[next_grid]
                if nv > 0:
                    connections[v, 3, nv] += 1
                    
    freqs /= np.sum(freqs)
    connections /= np.maximum(1, connections.sum(axis = 2, keepdims = True))
    return connections, freqs

def process_input_maps(maps, merge_subtiles = True):
    current_tileset = None
    all_conn_probs = None
    all_freqs = None
    for map_file in tqdm(maps):
        map_data = get_map_data(map_file)
        tileset, map_size, tile_data = get_tile_data(map_data)
        
        if current_tileset is None:
            current_tileset = tileset
            sc_to_gen, gen_to_sc, tile_count = get_generator_mapping(tileset, merge_subtiles = merge_subtiles)
            
        elif current_tileset != tileset:
            raise ValueError("must be maps of same tileset")
            
        tile_data_gen = sc_to_gen[tile_data]
        map_conn_probs, map_freqs = get_connections_and_freqs(tile_data_gen, tile_count)
        
        if all_conn_probs is None:
            all_conn_probs = map_conn_probs
            all_freqs = map_freqs
        else:
            all_conn_probs += map_conn_probs
            all_freqs += map_freqs
            
    all_freqs /= np.sum(all_freqs)
    all_conn_probs /= np.maximum(1, all_conn_probs.sum(axis = 2, keepdims = True))
    return current_tileset, sc_to_gen, gen_to_sc, tile_count, all_conn_probs, all_freqs

def get_subtile_probs(indices, rare_prob = 0.125):
    probs = np.ones_like(indices, dtype = np.float32)
    diffs = np.diff(indices)
    if (diffs > 1).any():
        first_index = np.where(diffs > 1)[0][0]
        probs[first_index + 1:] = rare_prob
    return probs / np.sum(probs)

def randomize_subtiles(tile_results, tileset, cv5_data = None, rare_prob = 0.125):
    if cv5_data is None:
        cv5_data = get_cv5_data(tileset)
    
    doodad_start = doodad_start_each_tileset[tileset]
    nonnull_tiles = np.array([not d['null'] for d in cv5_data])
    subtile_avail = nonnull_tiles[:doodad_start * 16].reshape(doodad_start, 16)
    subtile_range = np.arange(16)
    
    # randomly choose subtiles
    subtile_map = np.full(tile_results.shape, -1)
    subtile_map[tile_results == 1] = -2
    subtile_map[tile_results >= doodad_start * 16] = -2

    for y, row in enumerate(tile_results):
        for x, tile in enumerate(row):
            if subtile_map[y, x] == -1:
                tile_group = tile // 16
                choices = subtile_range[subtile_avail[tile_group]]
                if len(choices) == 0:
                    subtile_map[y, x] = 0
                else:
                    probs = get_subtile_probs(choices, rare_prob = rare_prob)
                    subtile = np.random.choice(choices, p = probs)
                    subtile_map[y, x] = subtile

            if subtile_map[y, x] != -2:
                if x < subtile_map.shape[1] - 1 and subtile_map[y, x + 1] == -1 and tile_group % 2 == 0:
                    next_tile_group = tile_results[y, x + 1] // 16
                    if next_tile_group == tile_group + 1:
                        subtile_map[y, x + 1] = subtile_map[y, x]
                if y < subtile_map.shape[0] - 1 and subtile_map[y + 1, x] == -1 and cv5_data[tile]["has_below"] > 0:
                    next_tile = tile_results[y + 1, x]
                    next_tile_group = next_tile // 16
                    if cv5_data[next_tile]["has_above"] > 0 and not cv5_data[next_tile + subtile_map[y, x]]["null"]:
                        subtile_map[y + 1, x] = subtile_map[y, x]

    subtile_map[tile_results == 1] = 0
    subtile_map[tile_results >= doodad_start * 16] = 0
    
    assert (subtile_map >= 0).all()
    tile_results += subtile_map