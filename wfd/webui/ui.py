# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import gradio as gr
import torch

from ..wf_diffusers import AutoencoderTile
from ..wf_diffusers import WaveFunctionDiffusionPipeline
from ..wf_diffusers import WaveFunctionDiffusionImg2ImgPipeline

from ..scmap import tiles_to_scx, demo_map_image, get_tile_data
from .doki import theme, theme_settings
import random, time, json

from PIL import Image
import numpy as np
import os

DEFAULT_TILESET = "jungle"
SUPPORTED_TILESETS = ["ashworld", "badlands", "desert", "ice", "jungle", "platform", "twilight", "installation", "platform_32x32"]

should_log_prompt = False

current_tileset = ""
current_mode = ""
current_tilenet = None
current_pipeline = None

def set_theme(theme):
    # use this before running start_demo
    settings_file = os.path.join(os.path.dirname(__file__), "doki_settings.json")
    try:
        with open(settings_file, encoding = "utf-8") as fp:
            doki = json.load(fp)
    
        doki["theme"] = theme
        
        with open(settings_file, "w", encoding = "utf-8") as fp:
            json.dump(doki, fp)
    except Exception as e:
        print(e)

def get_pretrained_path(tileset):
    if tileset == "platform_32x32":
        return "wdcqc/starcraft-platform-terrain-32x32"
    elif tileset == "ashworld":
        return "wdcqc/starcraft-terrain-64x64"
    elif tileset == "badlands":
        return "wdcqc/starcraft-terrain-64x64"
    elif tileset == "desert":
        return "wdcqc/starcraft-terrain-64x64"
    elif tileset == "ice":
        return "wdcqc/starcraft-terrain-64x64"
    elif tileset == "jungle":
        return "wdcqc/starcraft-terrain-64x64"
    elif tileset == "platform":
        return "wdcqc/starcraft-terrain-64x64"
    elif tileset == "twilight":
        return "wdcqc/starcraft-terrain-64x64"
    elif tileset == "installation":
        return "wdcqc/starcraft-terrain-64x64"
    raise NotImplementedError

def get_vae_path(tileset):
    if tileset == "platform_32x32":
        return "wdcqc/starcraft-platform-terrain-32x32", "tile_vae"
    elif tileset == "ashworld":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_ashworld"
    elif tileset == "badlands":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_badlands"
    elif tileset == "desert":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_desert"
    elif tileset == "ice":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_ice"
    elif tileset == "jungle":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_jungle"
    elif tileset == "platform":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_platform"
    elif tileset == "twilight":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_twilight"
    elif tileset == "installation":
        return "wdcqc/starcraft-terrain-64x64", "tile_vae_install"
    raise NotImplementedError

def get_wfc_data_path(tileset):
    if tileset == "platform_32x32":
        return get_tile_data("wfc/platform_32x32.npz")
    elif tileset == "ashworld":
        return get_tile_data("wfc/ashworld_64x64.npz")
    elif tileset == "badlands":
        return get_tile_data("wfc/badlands_64x64.npz")
    elif tileset == "desert":
        return get_tile_data("wfc/desert_64x64.npz")
    elif tileset == "ice":
        return get_tile_data("wfc/ice_64x64.npz")
    elif tileset == "jungle":
        return get_tile_data("wfc/jungle_64x64.npz")
    elif tileset == "platform":
        return get_tile_data("wfc/platform_64x64.npz")
    elif tileset == "twilight":
        return get_tile_data("wfc/twilight_64x64.npz")
    elif tileset == "installation":
        return get_tile_data("wfc/install_64x64.npz")
    raise NotImplementedError

def get_dreambooth_prompt(tileset):
    if tileset == "platform_32x32":
        return ", isometric scspace terrain"
    elif tileset == "ashworld":
        return ", isometric starcraft ashworld terrain"
    elif tileset == "badlands":
        return ", isometric starcraft badlands terrain"
    elif tileset == "desert":
        return ", isometric starcraft desert terrain"
    elif tileset == "ice":
        return ", isometric starcraft ice terrain"
    elif tileset == "jungle":
        return ", isometric starcraft jungle terrain"
    elif tileset == "platform":
        return ", isometric starcraft platform terrain"
    elif tileset == "twilight":
        return ", isometric starcraft twilight terrain"
    elif tileset == "installation":
        return ", isometric starcraft installation terrain"
    raise NotImplementedError

def run_demo(
    prompt,
    auto_add_dreambooth,
    neg_prompt,
    tileset,
    steps,
    cfg_scale,
    wfc_guidance_start_step,
    wfc_guidance_strength,
    wfc_guidance_final_step,
    wfc_guidance_final_strength,
    map_width,
    map_height,
    show_raw_image
):
    global current_tileset, current_mode, current_tilenet, current_pipeline

    if should_log_prompt:
        time_start = time.time()

    # 1. Define pipeline
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("Warning: CUDA is not available. Using CPU, which will be very slow for this demo.")
        device = "cpu"

    wfc_data_path = get_wfc_data_path(tileset)

    if current_tileset == tileset and current_mode == "txt2img":
        tilenet = current_tilenet
        pipeline = current_pipeline
    else:
        vae_path, vae_subfolder = get_vae_path(tileset)
        tilenet = AutoencoderTile.from_pretrained(
            vae_path,
            subfolder=vae_subfolder
        ).to(device)
        pipeline = WaveFunctionDiffusionPipeline.from_pretrained(
            get_pretrained_path(tileset),
            tile_vae = tilenet,
            wfc_data_path = wfc_data_path
        )
        pipeline.to(device)
        pipeline.set_precision("half")

        current_mode = "txt2img"
        current_tileset = tileset
        current_tilenet = tilenet
        current_pipeline = pipeline
        

    if auto_add_dreambooth:
        prompt = prompt + get_dreambooth_prompt(tileset)

    if map_width * map_height > 64 * 64:
        pipeline.enable_attention_slicing()
    else:
        pipeline.disable_attention_slicing()

    # 2. Pipe
    pipeline_output = pipeline(
        prompt,
        width = 8 * map_width,
        height = 8 * map_height,
        negative_prompt = neg_prompt,
        num_inference_steps = steps,
        guidance_scale = cfg_scale,
        wfc_guidance_start_step = wfc_guidance_start_step,
        wfc_guidance_strength = wfc_guidance_strength,
        wfc_guidance_final_steps = wfc_guidance_final_step,
        wfc_guidance_final_strength = wfc_guidance_final_strength,
    )

    # 3. Get result
    if show_raw_image:
        image = pipeline_output.images[0]
    wave = pipeline_output.waves[0]
    tile_result = wave.argmax(axis=2)

    # 4. Get map image
    if not show_raw_image:
        image = demo_map_image(tile_result, wfc_data_path = wfc_data_path)

    # 5. Generate scx file
    gen_map = "outputs/{}_{:04d}.scx".format(time.strftime("%Y%m%d_%H%M%S"), random.randint(0, 1e4))
    tiles_to_scx(
        tile_result,
        gen_map,
        wfc_data_path = wfc_data_path
    )

    if should_log_prompt:
        print("Generated image and map for prompt [{}] at {} steps in {:.2f} seconds.".format(
            prompt,
            steps,
            time.time() - time_start
        ))

    return image, gen_map

def run_demo_img2img(
    prompt,
    auto_add_dreambooth,
    neg_prompt,
    image_pil,
    tileset,
    brightness,
    steps,
    cfg_scale,
    wfc_guidance_start_step,
    wfc_guidance_strength,
    wfc_guidance_final_step,
    wfc_guidance_final_strength,
    map_width,
    map_height,
    show_raw_image
):
    global current_tileset, current_mode, current_tilenet, current_pipeline

    if should_log_prompt:
        time_start = time.time()

    # 1. Define pipeline
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("Warning: CUDA is not available. Using CPU, which will be very slow for this demo.")
        device = "cpu"

    wfc_data_path = get_wfc_data_path(tileset)

    if current_tileset == tileset and current_mode == "img2img":
        tilenet = current_tilenet
        pipeline = current_pipeline
    else:
        vae_path, vae_subfolder = get_vae_path(tileset)
        tilenet = AutoencoderTile.from_pretrained(
            vae_path,
            subfolder=vae_subfolder
        ).to(device)
        pipeline = WaveFunctionDiffusionImg2ImgPipeline.from_pretrained(
            get_pretrained_path(tileset),
            tile_vae = tilenet,
            wfc_data_path = wfc_data_path
        )
        pipeline.to(device)
        pipeline.set_precision("half")

        current_mode = "img2img"
        current_tileset = tileset
        current_tilenet = tilenet
        current_pipeline = pipeline

    if auto_add_dreambooth:
        prompt = get_dreambooth_prompt(tileset) + prompt

    image = image_pil.resize((8 * map_width, 8 * map_height))

    if brightness != 0:
        img_np = np.array(image, dtype = np.float32)
        img_np += brightness * 127.5
        img_np = np.minimum(img_np, 255)
        img_np = np.maximum(img_np, 0)
        img_np = img_np.astype(np.uint8)
        image = Image.fromarray(img_np)

    if map_width * map_height > 64 * 64:
        pipeline.enable_attention_slicing()
    else:
        pipeline.disable_attention_slicing()

    # 2. Pipe
    pipeline_output = pipeline(
        prompt,
        negative_prompt = neg_prompt,
        image = image,
        num_inference_steps = steps,
        guidance_scale = cfg_scale,
        wfc_guidance_start_step = wfc_guidance_start_step,
        wfc_guidance_strength = wfc_guidance_strength,
        wfc_guidance_final_steps = wfc_guidance_final_step,
        wfc_guidance_final_strength = wfc_guidance_final_strength,
    )

    # 3. Get result
    if show_raw_image:
        image = pipeline_output.images[0]
    wave = pipeline_output.waves[0]
    tile_result = wave.argmax(axis=2)

    # 4. Get map image
    if not show_raw_image:
        image = demo_map_image(tile_result, wfc_data_path = wfc_data_path)

    # 5. Generate scx file
    gen_map = "outputs/{}_{:04d}.scx".format(time.strftime("%Y%m%d_%H%M%S"), random.randint(0, 1e4))
    tiles_to_scx(
        tile_result,
        gen_map,
        wfc_data_path = wfc_data_path
    )

    if should_log_prompt:
        print("Generated image and map for prompt [{}] at {} steps in {:.2f} seconds.".format(
            prompt,
            steps,
            time.time() - time_start
        ))

    return image, gen_map

def preinstall():
    global current_tileset, current_mode, current_tilenet, current_pipeline

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tileset = DEFAULT_TILESET
    vae_path, vae_subfolder = get_vae_path(tileset)
    tilenet = AutoencoderTile.from_pretrained(
        vae_path,
        subfolder=vae_subfolder
    ).to(device)
    wfc_data_path = get_wfc_data_path(tileset)
    pipeline = WaveFunctionDiffusionPipeline.from_pretrained(
        get_pretrained_path(tileset),
        tile_vae = tilenet,
        wfc_data_path = wfc_data_path
    )
    pipeline.to(device)
    pipeline.set_precision("half")

    current_mode = "txt2img"
    current_tileset = tileset
    current_tilenet = tilenet
    current_pipeline = pipeline

def start_demo(args):
    global should_log_prompt
    if hasattr(args, "log_prompt") and args.log_prompt:
        should_log_prompt = True

    # Max map size should be set to a value that avoids CUDA Out of Memory error
    if hasattr(args, "max_size") and args.max_size is not None:
        max_map_size = args.max_size
    else:
        max_map_size = 256

    if args.colab:
        preinstall()
    with gr.Blocks(analytics_enabled=False, title="WaveFunctionDiffusion demo page") as demo:
        if args.link_to_colab:
            gr.Markdown("Run this demo in [Google Colab](https://colab.research.google.com/github/wdcqc/WaveFunctionDiffusion/blob/remaster/colab/WaveFunctionDiffusion_Demo.ipynb)")
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="Lost Temple")
                    auto_add_dreambooth = gr.Checkbox(value=True, label="Automatically add the dreambooth prompt (e.g. 'isometric starcraft jungle terrain')")
                    neg_prompt = gr.Textbox(label="Negative Prompt")
                    tileset = gr.Dropdown(SUPPORTED_TILESETS, value=DEFAULT_TILESET, label="Tileset")
                    steps = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Scheduler steps")
                    cfg_scale = gr.Slider(minimum=0, maximum=15, step=0.1, value=3.5, label="CFG Guidance Scale")
                    wfc_guidance_start_step = gr.Slider(minimum=0, maximum=100, value=30, step=1, label="WFC guidance starting step")
                    wfc_guidance_strength = gr.Slider(minimum=0, maximum=100, value=5, step=0.1, label="WFC guidance strength")
                    wfc_guidance_final_step = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="WFC guidance final steps")
                    wfc_guidance_final_strength = gr.Slider(minimum=0, maximum=100, value=5, step=0.1, label="WFC guidance final strength")
                    map_width = gr.Slider(minimum=32, maximum=max_map_size, step=32, value=64, label="Width")
                    map_height = gr.Slider(minimum=32, maximum=max_map_size, step=32, value=64, label="Height")
                    show_raw_image = gr.Checkbox(label="Show raw generated image")
                with gr.Column():
                    btn = gr.Button("Start Diffusion!")
                    output_image = gr.Image(label="Output Image")
                    output_file = gr.File(label="Output Starcraft Map")
            btn.click(fn=run_demo, inputs=[
                prompt,
                auto_add_dreambooth,
                neg_prompt,
                tileset,
                steps,
                cfg_scale,
                wfc_guidance_start_step,
                wfc_guidance_strength,
                wfc_guidance_final_step,
                wfc_guidance_final_strength,
                map_width,
                map_height,
                show_raw_image
            ], outputs=[
                output_image,
                output_file
            ])
        with gr.Tab("img2img"): # copypaste programming ftw
            with gr.Row():
                with gr.Column():
                    i2i_prompt = gr.Textbox(label="Prompt", value="Lost Temple")
                    i2i_auto_add_dreambooth = gr.Checkbox(value=True, label="Automatically add the dreambooth prompt (e.g. 'isometric scspace terrain')")
                    i2i_neg_prompt = gr.Textbox(label="Negative Prompt")
                    i2i_image_pil = gr.Image(type="pil", label="Image")
                    i2i_tileset = gr.Dropdown(SUPPORTED_TILESETS, value=DEFAULT_TILESET, label="Tileset")
                    i2i_brightness = gr.Slider(minimum=-2, maximum=2, step=0.01, value=0, label="Brightness fix (Turn down if image is too bright)")
                    i2i_steps = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Scheduler steps")
                    i2i_cfg_scale = gr.Slider(minimum=0, maximum=15, step=0.1, value=6.5, label="CFG Guidance Scale")
                    i2i_wfc_guidance_start_step = gr.Slider(minimum=0, maximum=100, value=30, step=1, label="WFC guidance starting step")
                    i2i_wfc_guidance_strength = gr.Slider(minimum=0, maximum=100, value=5, step=0.1, label="WFC guidance strength")
                    i2i_wfc_guidance_final_step = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="WFC guidance final steps")
                    i2i_wfc_guidance_final_strength = gr.Slider(minimum=0, maximum=100, value=5, step=0.1, label="WFC guidance final strength")
                    i2i_map_width = gr.Slider(minimum=32, maximum=max_map_size, step=32, value=64, label="Width")
                    i2i_map_height = gr.Slider(minimum=32, maximum=max_map_size, step=32, value=64, label="Height")
                    i2i_show_raw_image = gr.Checkbox(label="Show raw generated image")
                with gr.Column():
                    i2i_btn = gr.Button("Start Diffusion!")
                    i2i_output_image = gr.Image(label="Output Image")
                    i2i_output_file = gr.File(label="Output Starcraft Map")
            i2i_btn.click(fn=run_demo_img2img, inputs=[
                i2i_prompt,
                i2i_auto_add_dreambooth,
                i2i_neg_prompt,
                i2i_image_pil,
                i2i_tileset,
                i2i_brightness,
                i2i_steps,
                i2i_cfg_scale,
                i2i_wfc_guidance_start_step,
                i2i_wfc_guidance_strength,
                i2i_wfc_guidance_final_step,
                i2i_wfc_guidance_final_strength,
                i2i_map_width,
                i2i_map_height,
                i2i_show_raw_image
            ], outputs=[
                i2i_output_image,
                i2i_output_file
            ])
        theme()

    if args.colab:
        demo.launch(debug=True)
    else:
        demo.launch()