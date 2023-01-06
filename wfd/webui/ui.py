import gradio as gr
import torch

from ..wf_diffusers import AutoencoderTile
from ..wf_diffusers import WaveFunctionDiffusionPipeline
from ..wf_diffusers import WaveFunctionDiffusionImg2ImgPipeline

from ..scmap import tiles_to_scx, demo_map_image
from .doki import theme, theme_settings
import random, time

from PIL import Image
import numpy as np

DEFAULT_TILESET = "platform"
SUPPORTED_TILESETS = ["platform"]

current_tileset = ""
current_mode = ""
current_tilenet = None
current_pipeline = None

def get_pretrained_path(tileset):
    if tileset == "platform":
        return "wdcqc/starcraft-platform-terrain-32x32"
    raise NotImplementedError

def get_wfc_data_path(tileset):
    if tileset == "platform":
        return "tile_data/wfc/platform_32x32.npz"
    raise NotImplementedError

def get_dreambooth_prompt(tileset):
    if tileset == "platform":
        return "isometric scspace terrain, "
    raise NotImplementedError

def run_demo(
    prompt,
    auto_add_dreambooth,
    neg_prompt,
    tileset,
    steps,
    wfc_guidance_start_step,
    wfc_guidance_strength,
    wfc_guidance_final_step,
    wfc_guidance_final_strength,
    show_raw_image
):
    global current_tileset, current_mode, current_tilenet, current_pipeline

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
        tilenet = AutoencoderTile.from_pretrained(
            get_pretrained_path(tileset),
            subfolder="tile_vae"
        ).to(device)
        pipeline = WaveFunctionDiffusionPipeline.from_pretrained(
            get_pretrained_path(tileset),
            tile_vae = tilenet,
            wfc_data_path = wfc_data_path
        )
        pipeline.to(device)

        current_mode = "txt2img"
        current_tileset = tileset
        current_tilenet = tilenet
        current_pipeline = pipeline
        

    if auto_add_dreambooth:
        prompt = get_dreambooth_prompt(tileset) + prompt

    # 2. Pipe
    pipeline_output = pipeline(
        prompt,
        negative_prompt = neg_prompt,
        num_inference_steps = steps,
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

    return image, gen_map

def run_demo_img2img(
    prompt,
    auto_add_dreambooth,
    neg_prompt,
    image_pil,
    tileset,
    brightness,
    steps,
    wfc_guidance_start_step,
    wfc_guidance_strength,
    wfc_guidance_final_step,
    wfc_guidance_final_strength,
    show_raw_image
):
    global current_tileset, current_mode, current_tilenet, current_pipeline

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
        tilenet = AutoencoderTile.from_pretrained(
            get_pretrained_path(tileset),
            subfolder="tile_vae"
        ).to(device)
        pipeline = WaveFunctionDiffusionImg2ImgPipeline.from_pretrained(
            get_pretrained_path(tileset),
            tile_vae = tilenet,
            wfc_data_path = wfc_data_path
        )
        pipeline.to(device)

        current_mode = "img2img"
        current_tileset = tileset
        current_tilenet = tilenet
        current_pipeline = pipeline

    if auto_add_dreambooth:
        prompt = get_dreambooth_prompt(tileset) + prompt

    image = image_pil.resize((512, 512))

    if brightness != 0:
        img_np = np.array(image, dtype = np.float32)
        img_np += brightness * 127.5
        img_np = img_np.astype(np.uint8)
        image = Image.fromarray(img_np)

    # 2. Pipe
    pipeline_output = pipeline(
        prompt,
        negative_prompt = neg_prompt,
        image = image,
        num_inference_steps = steps,
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

    return image, gen_map

def preinstall():
    global current_tileset, current_mode, current_tilenet, current_pipeline

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tileset = DEFAULT_TILESET
    tilenet = AutoencoderTile.from_pretrained(
        get_pretrained_path(tileset),
        subfolder="tile_vae"
    ).to(device)
    wfc_data_path = get_wfc_data_path(tileset)
    pipeline = WaveFunctionDiffusionPipeline.from_pretrained(
        get_pretrained_path(tileset),
        tile_vae = tilenet,
        wfc_data_path = wfc_data_path
    )
    pipeline.to(device)

    current_mode = "txt2img"
    current_tileset = tileset
    current_tilenet = tilenet
    current_pipeline = pipeline

def start_demo(args):
    if args.colab:
        preinstall()
    with gr.Blocks(analytics_enabled=False, title="WaveFunctionDiffusion demo page") as demo:
        if args.link_to_colab:
            gr.Markdown("Run this demo in [Google Colab](https://colab.research.google.com/github/wdcqc/WaveFunctionDiffusion/blob/remaster/colab/WaveFunctionDiffusion_Demo.ipynb)")
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="Lost Temple")
                    auto_add_dreambooth = gr.Checkbox(value=True, label="Automatically add the dreambooth prompt ('isometric scspace terrain')")
                    neg_prompt = gr.Textbox(label="Negative Prompt")
                    tileset = gr.Dropdown(SUPPORTED_TILESETS, value=DEFAULT_TILESET, label="Tileset (currently only platform is trained)")
                    steps = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Scheduler steps")
                    wfc_guidance_start_step = gr.Slider(minimum=0, maximum=50, value=20, step=1, label="WFC guidance starting step")
                    wfc_guidance_strength = gr.Slider(minimum=0, maximum=100, value=5, step=0.1, label="WFC guidance strength")
                    wfc_guidance_final_step = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="WFC guidance final steps")
                    wfc_guidance_final_strength = gr.Slider(minimum=0, maximum=100, value=10, step=0.1, label="WFC guidance final strength")
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
                wfc_guidance_start_step,
                wfc_guidance_strength,
                wfc_guidance_final_step,
                wfc_guidance_final_strength,
                show_raw_image
            ], outputs=[
                output_image,
                output_file
            ])
        with gr.Tab("img2img"): # copypaste programming ftw
            with gr.Row():
                with gr.Column():
                    i2i_prompt = gr.Textbox(label="Prompt", value="Lost Temple")
                    i2i_auto_add_dreambooth = gr.Checkbox(value=True, label="Automatically add the dreambooth prompt ('isometric scspace terrain')")
                    i2i_neg_prompt = gr.Textbox(label="Negative Prompt")
                    i2i_image_pil = gr.Image(type="pil", label="Image")
                    i2i_tileset = gr.Dropdown(SUPPORTED_TILESETS, value=DEFAULT_TILESET, label="Tileset (currently only platform is trained)")
                    i2i_brightness = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label="Brightness fix")
                    i2i_steps = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Scheduler steps")
                    i2i_wfc_guidance_start_step = gr.Slider(minimum=0, maximum=50, value=30, step=1, label="WFC guidance starting step")
                    i2i_wfc_guidance_strength = gr.Slider(minimum=0, maximum=100, value=5, step=0.1, label="WFC guidance strength")
                    i2i_wfc_guidance_final_step = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="WFC guidance final steps")
                    i2i_wfc_guidance_final_strength = gr.Slider(minimum=0, maximum=100, value=10, step=0.1, label="WFC guidance final strength")
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
                i2i_wfc_guidance_start_step,
                i2i_wfc_guidance_strength,
                i2i_wfc_guidance_final_step,
                i2i_wfc_guidance_final_strength,
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