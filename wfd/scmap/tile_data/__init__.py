import os

TILE_DATA_PATH = os.path.dirname(__file__)

def find_tile_data(f):
    return os.path.join(TILE_DATA_PATH, f)

def get_tileset_keyword(tileset):
    return "installation" if tileset == "install" else tileset