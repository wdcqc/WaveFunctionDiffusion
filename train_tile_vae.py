# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import torch, torch.nn as nn
import numpy as np, os, re, json, sys, time, shutil, argparse
from PIL import Image

# Import VAEs
from diffusers import AutoencoderKL
from wfd.wf_diffusers import AutoencoderTile
from torchinfo import summary

# SCMap utilities
from wfd.scmap import randomize_subtiles, get_cv5_data, process_input_maps, get_shrink_mapping
from wfd.scmap import get_default_output_map_data, replace_tile_data, get_map_data, get_tile_data
from wfd import SCInputMapsDataset, SCRandomMapsDataset

# Train loop
from wfd import train_loop
from wfd import default_loss_weights

def round_to_multiples_of_n(x, n):
    return x + (n - x%n)%n

def decode_image(tensor):
    x = (tensor.transpose((1, 2, 0)) + 1) * 127.5
    x = x.astype(np.uint8)
    return Image.fromarray(x)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to TileVAE config json file.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Path to training config json file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved TileVAE model (`save_pretrained`).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to TileVAE model to output.",
    )
    parser.add_argument(
        "--save_dataset_wfc",
        type=str,
        default=None,
        help="Path to save dataset WFC parameters. (npz format)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of maps.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="The size for generated images from map tiles.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        default=(32, 32),
        help="The size of tile matrix sliced from input maps.",
    )
    parser.add_argument(
        "--autosave",
        type=int,
        default=100,
        help="Save training results every N steps.",
    )
    parser.add_argument(
        "--image_vae",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="The path to image VAE model (`AutoencoderKL`). Can either be a huggingface hub or a local path.",
    )
    parser.add_argument(
        "--random_dataset",
        action="store_true",
        help="Whether or not to use a randomized tiles dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The thing that makes a dev cool.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.output is None and args.model_path is not None:
        args.output = args.model_path

    if args.model_config is None and args.model_path is None:
        raise ValueError("Either model_config or model_path needs to be specified.")

    return args

def main(args):
    print("[ENV] TORCH_VERSION =", torch.__version__)

    # Load the dataset
    data_dir = args.data_dir
    maps = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    w, h = args.tile_size
    dataset = SCInputMapsDataset(maps, w = w, h = h, img_size = args.img_size)
    if args.random_dataset:
        dataset_r = SCRandomMapsDataset(dataset.tileset, w = w, h = h, img_size = args.img_size, shrink_mappings=dataset.shrink_mappings, freqs=dataset.freqs_shrink)

    print("[DATASET] TILE_COUNT =", dataset.shrink_count)

    # Save WFC
    if args.save_dataset_wfc is not None:
        dataset.save_wfc(args.save_dataset_wfc)

    # Create or load tilenet VAE
    if args.model_config is not None:
        with open(args.model_config, encoding = "utf-8") as fp:
            config = json.load(fp)
        tile_count = round_to_multiples_of_n(dataset.shrink_count, config["conv_init_groups"])
        config["in_channels"] = tile_count
        config["out_channels"] = tile_count
        tilenet = AutoencoderTile(**config)
    elif args.model_path is not None:
        tilenet = AutoencoderTile.from_pretrained(args.model_path)
        tile_count = tilenet.in_channels
    else:
        raise NotImplementedError

    tilenet.to(args.device)

    # Print a summary
    summary(tilenet, input_size = (1, tile_count, h, w))

    # Create or load image VAE
    vae = AutoencoderKL.from_pretrained(args.image_vae)
    vae.to(args.device)

    # Load training config
    if args.train_config is not None:
        with open(args.train_config, encoding = "utf-8") as fp:
            train_config = json.load(fp)
    else:
        train_config = {
            "epochs" : 5,
            "learning_rate" : 0.001,
            "loss_weights" : default_loss_weights(),
            "recon_temperature" : 0.01,
            "batch_size" : 4,
            "nudge" : 5,
            "nudge_loss_type" : "ce",
            "clamp_grad" : 10,
            "grad_accum" : 1,
            "log_interval" : 100,
        }

    try:
        os.makedirs(os.path.pathname(args.output))
    except:
        pass

    # Start training
    try:
        print("[INFO] Starting Tilenet VAE training...")
        train_loop(
            tilenet,
            vae,
            dataset_r if args.random_dataset else dataset,
            tile_count = tile_count,
            epochs = train_config["epochs"],
            lr = train_config["learning_rate"],
            batch_size = train_config["batch_size"],
            clamp = train_config["clamp_grad"],
            loss_weights = train_config["loss_weights"],
            recon_temperature = train_config["recon_temperature"],
            max_nudge = train_config["nudge"],
            nudge_loss_type = train_config["nudge_loss_type"],
            grad_accum = train_config["grad_accum"],
            log_interval = train_config["log_interval"],
            device = args.device,
            autosave = args.autosave,
            autosave_path = args.output,
        )
    except KeyboardInterrupt as ke:
        # note that this won't work when the script is run from command line
        tilenet.save_pretrained(args.output)
        print("[INFO] Received KeyboardInterrupt, saved training result to {}".format(args.output))
        raise ke
    else:
        tilenet.save_pretrained(args.output)
        print("[INFO] Training finished, saved training result to {}".format(args.output))

if __name__ == "__main__":
    args = parse_args()
    main(args)