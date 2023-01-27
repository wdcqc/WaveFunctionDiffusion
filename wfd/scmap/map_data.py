# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import os, numpy as np, struct
from ..mpqapi import get_chk_from_mpq, pack_to_mpq
from .default import DEFAULT_CHK_PATH

def get_map_data(map_path):
    if map_path.endswith(".chk"):
        with open(map_path, "rb") as fp:
            data = fp.read()
        return data
    else:
        try:
            data = get_chk_from_mpq(map_path)
        except IOError:
            raise FileNotFoundError("MPQ fails to extract / not exists")
        if not data or len(data) == 0:
            raise FileNotFoundError("MPQ fails to extract / not exists")
        return data
    
def era_to_tileset(era):
    return ["badlands", "platform", "install", "ashworld", "jungle", "desert", "ice", "twilight"][era % 8]

def tileset_to_era(tileset):
    return {
        "badlands" : 0,
        "platform" : 1,
        "install" : 2,
        "ashworld" : 3,
        "jungle" : 4,
        "desert" : 5,
        "ice" : 6,
        "twilight" : 7,
    }[tileset]

def get_tile_data(data):
    ofs = data.find(b'DIM ')
    assert ofs == data.rfind(b'DIM '), 'multiple DIM sections, probably protected map'
    assert data[ofs + 4 : ofs + 8] == b'\x04\x00\x00\x00', 'wrong DIM length'
    
    dim = struct.unpack("<HH", data[ofs + 8 : ofs + 12])
    
    ofs = data.find(b'ERA ')
    assert ofs == data.rfind(b'ERA '), 'multiple ERA sections, probably protected map'
    assert data[ofs + 4 : ofs + 8] == b'\x02\x00\x00\x00', 'wrong ERA length'
    
    era = struct.unpack("<H", data[ofs + 8 : ofs + 10])
    tileset = era_to_tileset(era[0])
    
    ofs = data.find(b'MTXM')
    assert ofs == data.rfind(b'MTXM'), 'multiple MTXM sections, probably protected map'
    
    # unpack 1 value (TIL it actually works)
    data_len, = struct.unpack("<I", data[ofs + 4 : ofs + 8])
    assert data_len == dim[0] * dim[1] * 2, 'wrong MTXM length'
    
    section_data = struct.unpack("<" + "H" * (data_len // 2), data[ofs + 8 : ofs + 8 + data_len])
    
    return tileset, dim, np.array(section_data).reshape((dim[1], dim[0]))

def get_default_output_map_data(map_file = DEFAULT_CHK_PATH):
    return get_map_data(map_file)

def replace_tile_data(data, new_tileset, new_dim, new_mtxm):
    ofs = data.find(b'DIM ')
    assert ofs == data.rfind(b'DIM '), 'multiple DIM sections, probably protected map'
    assert data[ofs + 4 : ofs + 8] == b'\x04\x00\x00\x00', 'wrong DIM length'
    
    new_dim_bytes = struct.pack("<HH", *new_dim)
    data = data[: ofs + 8] + new_dim_bytes + data[ofs + 12 :]
    
    ofs = data.find(b'ERA ')
    assert ofs == data.rfind(b'ERA '), 'multiple ERA sections, probably protected map'
    assert data[ofs + 4 : ofs + 8] == b'\x02\x00\x00\x00', 'wrong ERA length'
    
    new_era = tileset_to_era(new_tileset)
    new_era_bytes = struct.pack("<H", new_era)
    data = data[: ofs + 8] + new_era_bytes + data[ofs + 10 :]

    for sig in (b'MTXM', b'TILE'):
        ofs = data.find(sig)
        assert ofs == data.rfind(sig), 'multiple MTXM sections, probably protected map'

        # unpack 1 value (TIL it actually works)
        data_len, = struct.unpack("<I", data[ofs + 4 : ofs + 8])
        
        new_mtxm_bytes = struct.pack("<" + "H" * len(new_mtxm), *new_mtxm)
        new_len_bytes = struct.pack("<I", len(new_mtxm_bytes))
        
        data = data[: ofs + 4] + new_len_bytes + new_mtxm_bytes + data[ofs + 8 + data_len :]
    
    # something breaks when changing size, so we have to fix them as well
    ofs = data.find(b'ISOM')
    assert ofs == data.rfind(b'ISOM'), 'multiple ISOM sections??? not even a protector would do that'
    data_len, = struct.unpack("<I", data[ofs + 4 : ofs + 8])
    
    w, h = new_dim
    isom_byte_count = (w // 2 + 1) * (h + 1) * 4 * 2
    new_len = struct.pack("<I", isom_byte_count)
    new_isom = struct.pack("<" + "B" * isom_byte_count, *([0] * isom_byte_count))
    
    data = data[: ofs + 4] + new_len + new_isom + data[ofs + 8 + data_len :]
    
    ofs = data.find(b'MASK')
    assert ofs == data.rfind(b'MASK'), 'multiple MASK sections??? not even a protector would do that'
    data_len, = struct.unpack("<I", data[ofs + 4 : ofs + 8])
    
    w, h = new_dim
    mask_byte_count = w * h
    new_len = struct.pack("<I", mask_byte_count)
    new_mask = struct.pack("<" + "B" * mask_byte_count, *([255] * mask_byte_count))
    
    data = data[: ofs + 4] + new_len + new_mask + data[ofs + 8 + data_len :]
    
    return new_dim, data