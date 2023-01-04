import numpy as np, os, json, struct

doodad_start_each_tileset = {
    "badlands" : 1024,
    "platform" : 933, # !!!!!!! it is not always 1024 !!!!!!!
    "install" : 1024,
    "ashworld" : 1024,
    "jungle" : 1024,
    "desert" : 770,
    "ice" : 1024,
    "twilight" : 797,
}

def get_height(flag):
    return 2 if flag & 4 != 0 else (1 if flag & 2 != 0 else 0)

def parse_section_vf4(d):
    data = struct.unpack("<" + "H" * 16, d)
    arr = np.array(data)
    
    height = [get_height(c) for c in data]
    walkable = [(1 if c & 1 != 0 else 0) for c in data]
    blocks_view = [(1 if c & 8 != 0 else 0) for c in data]
    ramp = [(1 if c & 16 != 0 else 0) for c in data]
    return {
        'all_walkable' : bool(np.array(walkable).all()),
        'all_unwalkable' : bool(np.logical_not(walkable).all()),
        'all_blocks_view' : bool(np.array(blocks_view).all()),
        'all_ramp' : bool(np.array(ramp).all()),
        'main_height' : get_height(data[5]),
        'height' : height,
        'walkable' : walkable,
        'blocks_view' : blocks_view,
        'ramp' : ramp,
    }

def parse_section_cv5(d):
    data = struct.unpack("<" + ("HBBHHHHHHHH" + "H" * 16), d)
    return {
        'index' : data[0],
        'raw_buildability' : data[1],
        'flags' : {
            "edge" : data[1] & 1 != 0,
            "cliff" : data[1] & 4 != 0,
        },
        'build' : {
            "creep" : data[1] & 64 != 0,
            "unbuildable" : data[1] & 128 != 0,
        },
        'build2' : {
            "buildable" : data[2] & 128 != 0,
        },
        'doodad_name' : data[5],
        'raw_groundHeight' : data[2],
        'megaTileIndices' : data[11:27]
    }

def analyze_tile_data(tilegroup_data, minitile_data):
    tiles = [] # megatiles
    for tile_group in tilegroup_data:
        for i in range(16):
            minitile_index = tile_group["megaTileIndices"][i]
            if(minitile_index == 0):
                tiles.append({
                    "tilegroup_index" : tile_group["index"],
                    "edge" : tile_group["flags"]["edge"],
                    "cliff" : tile_group["flags"]["cliff"],
                    "creep" : tile_group["build"]["creep"],
                    "buildable" : bool(not tile_group["build"]["unbuildable"]),
                    "buildable2" : tile_group["build2"]["buildable"],
                    "raw_groundHeight" : tile_group["raw_groundHeight"],

                    "null" : True,

                    # for easy fallback
                    "all_walkable" : False,
                    "all_unwalkable" : True,
                    "all_blocks_view" : False,
                    "all_ramp" : False,
                })
            else:
                minitiles = minitile_data[minitile_index]
                tiles.append({
                    "tilegroup_index" : tile_group["index"],
                    "edge" : tile_group["flags"]["edge"],
                    "cliff" : tile_group["flags"]["cliff"],
                    "creep" : tile_group["build"]["creep"],
                    "buildable" : bool(not tile_group["build"]["unbuildable"]),
                    "buildable2" : tile_group["build2"]["buildable"],
                    "raw_groundHeight" : tile_group["raw_groundHeight"],

                    "null" : False,

                    "all_walkable" : minitiles["all_walkable"],
                    "all_unwalkable" : minitiles["all_unwalkable"],
                    "mini_walkable" : minitiles["walkable"],

                    "all_blocks_view" : minitiles["all_blocks_view"],
                    "all_ramp" : minitiles["all_ramp"],

                    "mini_blocks_view" : minitiles["blocks_view"],
                    "mini_ramp" : minitiles["ramp"],

                    "mini_height" : minitiles["height"],
                    "main_height" : minitiles["main_height"],
                })
    return tiles

def analyze_tiles(p, k):
    with open(os.path.join(p, k + ".cv5"), "rb") as fp:
        cv5_data = fp.read()
    with open(os.path.join(p, k + ".vf4"), "rb") as fp:
        vf4_data = fp.read()

    minitile_data = [parse_section_vf4(vf4_data[c * 32 : (c+1) * 32]) for c in range(len(vf4_data) // 32)]
    tilegroup_data = [parse_section_cv5(cv5_data[c * 52 : (c+1) * 52]) for c in range(len(cv5_data) // 52)]

    tiles = analyze_tile_data(tilegroup_data, minitile_data)
    return tiles

def save_json(p, n, d):
    with open(os.path.join(p, n + ".json"), "w", encoding = "utf8") as fp:
        json.dump(d, fp)

def work_tileset(tileset,
                 base_path = "../tile_data/",
                 cv5_root = "../sc_files/TileSet",
                 ):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    data = analyze_tiles(cv5_root, tileset)
    save_json(base_path, tileset, data)
    print("{} converted to json format, length {}".format(tileset, len(data)))

def additional_info_tileset(tileset,
                            base_path = "../tile_data/",
                            cv5_root = "../sc_files/TileSet",
                            ):
    cv5_data_path = os.path.join(base_path, "{}.json".format(tileset))
    with open(cv5_data_path, encoding = "utf8") as fp:
        cv5_data = json.load(fp)

    cv5_file_path = os.path.join(cv5_root, "{}.cv5".format(tileset))
    with open(cv5_file_path, "rb") as fp:
        cv5_raw = fp.read()

    cv5_tilegroups = [struct.unpack("<HBB" + "H" * 24, cv5_raw[i * 52 : (i+1) * 52]) for i in range(len(cv5_raw) // 52)]

    assert len(cv5_data) // 16 == len(cv5_tilegroups)
    
    for i, tile in enumerate(cv5_data):
        tilegroup = i // 16
        if tilegroup < doodad_start_each_tileset[tileset]:
            tile["doodad"] = False
            tile["edges"] = [
                cv5_tilegroups[tilegroup][3],
                cv5_tilegroups[tilegroup][4],
                cv5_tilegroups[tilegroup][5],
                cv5_tilegroups[tilegroup][6],
            ]
            tile["has_above"] = cv5_tilegroups[tilegroup][8]
            tile["has_below"] = cv5_tilegroups[tilegroup][10]
        else:
            tile["doodad"] = True
            tile["dd_flags"] = cv5_tilegroups[tilegroup][2]
            tile["dd_overlay_id"] = cv5_tilegroups[tilegroup][3]
            tile["dd_name_id"] = cv5_tilegroups[tilegroup][5]
            tile["dd_bin_id"] = cv5_tilegroups[tilegroup][7]
            tile["dd_width"] = cv5_tilegroups[tilegroup][8]
            tile["dd_height"] = cv5_tilegroups[tilegroup][9]

    cv5_data_path_out = os.path.join(base_path, "{}_v2.npz".format(tileset))
    np.savez_compressed(cv5_data_path_out, data=json.dumps(cv5_data))

    print("additional information added to {}_v2.npz, length {}".format(tileset, len(cv5_data)))

def get_tile_index_mapping(tileset, base_path = "../tile_data/"):
    cv5_data_path = os.path.join(base_path, "{}.json".format(tileset))
    with open(cv5_data_path, encoding = "utf8") as fp:
        cv5_data = json.load(fp)
        
    total_megatiles = len(cv5_data)
    total_tilegroups = total_megatiles // 16
    doodad_start = doodad_start_each_tileset[tileset]
    
    sc_to_gen = np.full((total_megatiles,), -1)
    nulls = np.array([d['null'] for d in cv5_data])
    non_doodads_check = np.logical_not(nulls[0 : doodad_start * 16 : 16])
    non_doodad_count = np.sum(non_doodads_check)
    print("tileset: {} total tilegroups: {} non_doodads: {}".format(tileset, total_tilegroups, non_doodad_count))

    doodad_range = np.zeros((doodad_start,))
    doodad_range[non_doodads_check] = np.arange(1, non_doodad_count + 1)
    before_doodads = np.repeat(doodad_range, (16,))

    sc_to_gen[:len(before_doodads)] = before_doodads
    sc_to_gen[nulls] = 0
    doodad_indices = (sc_to_gen == -1)
    doodad_count = np.sum(doodad_indices)
    sc_to_gen[sc_to_gen == -1] = np.arange(non_doodad_count + 1, non_doodad_count + 1 + doodad_count)

    gen_len = non_doodad_count + 1 + doodad_count

    gen_to_sc = np.full((gen_len,), -1)
    gen_to_sc[0] = 1 # null tile. use null[1] to avoid potential desync problem.
    gen_to_sc[1 : non_doodad_count + 1] = np.where(non_doodads_check)[0] * 16
    gen_to_sc[non_doodad_count + 1 : non_doodad_count + 1 + doodad_count] = np.where(doodad_indices)[0]

    for i, k in enumerate(gen_to_sc):
        if i < non_doodad_count + 1:
            if sc_to_gen[k] != i:
                print(i, k, sc_to_gen[k])
        else:
            if sc_to_gen[k] != i:
                print(i, k, sc_to_gen[k])

    assert (sc_to_gen >= 0).all()
    assert (gen_to_sc >= 0).all()
    
    print("{} analysis done, doodads: {}, total: {}".format(tileset, doodad_count, 1 + non_doodad_count + doodad_count))
        
    return {
        "non_doodad_count" : non_doodad_count,
        "doodad_count" : doodad_count,
        "total_count" : 1 + non_doodad_count + doodad_count,
        "gen_to_sc" : gen_to_sc,
        "sc_to_gen" : sc_to_gen,
    }

if __name__ == "__main__":
    base_path = "../tile_data/"

    work_tileset("ashworld")
    work_tileset("badlands")
    work_tileset("desert")
    work_tileset("ice")
    work_tileset("install")
    work_tileset("jungle")
    work_tileset("platform")
    work_tileset("twilight")

    additional_info_tileset("badlands")
    additional_info_tileset("platform")
    additional_info_tileset("install")
    additional_info_tileset("ashworld")
    additional_info_tileset("jungle")
    additional_info_tileset("ice")
    additional_info_tileset("desert")
    additional_info_tileset("twilight")
    
    np.savez_compressed(os.path.join(base_path, "ice_mapping.npz"), **get_tile_index_mapping("ice"))
    np.savez_compressed(os.path.join(base_path, "desert_mapping.npz"), **get_tile_index_mapping("desert"))
    np.savez_compressed(os.path.join(base_path, "platform_mapping.npz"), **get_tile_index_mapping("platform"))
    np.savez_compressed(os.path.join(base_path, "install_mapping.npz"), **get_tile_index_mapping("install"))
    np.savez_compressed(os.path.join(base_path, "ashworld_mapping.npz"), **get_tile_index_mapping("ashworld"))
    np.savez_compressed(os.path.join(base_path, "jungle_mapping.npz"), **get_tile_index_mapping("jungle"))
    np.savez_compressed(os.path.join(base_path, "badlands_mapping.npz"), **get_tile_index_mapping("badlands"))
    np.savez_compressed(os.path.join(base_path, "twilight_mapping.npz"), **get_tile_index_mapping("twilight"))