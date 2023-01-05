## WaveFunctionDiffusion

Based on a mix of Wave Function Collapse (WFC) and Stable Diffusion (SD) algorithms, this repository generates a tile map (demonstrated with Starcraft:Remastered maps) from a simple text prompt (txt2img) or a given image (img2img).

It uses a dreamboothed Stable Diffusion model trained with images of tile maps, and a custom VAE model (`AutoencoderTile`) to encode and decode the latent variables to and from tile probabilities ("waves").

A WFC Guidance is also added to the sampling process, which pushes the generated tile map closer to the WFC transition rules. For more information about how guidance works, check out this tutorial: [Fine-Tuning, Guidance and Conditioning](https://github.com/huggingface/diffusion-models-class/tree/main/unit2)

The model is trained with 265 melee maps on the Space Platform tileset, which are mostly downloaded from Battle.net, bounding.net (scmscx.com) and broodwarmaps.net.

### Run with Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdcqc/WaveFunctionDiffusion/blob/remaster/colab/WaveFunctionDiffusion_Demo.ipynb)

### Installation

1. Install Python 3.9 (3.8 or 3.10 should also work)
1. Install CUDA and PyTorch
2. Install requirements with `pip install -r requirements.txt`
3. (Optional) Install xformers (this is pretty complicated but it increases diffusion efficiency)

### Local demo

Start the web UI demo:

```bash
python demo.py
```

Then open up the browser and navigate to the displayed URL to start diffusing!

In local demo, generated maps (.scx files) are saved in the `outputs` folder.

### Using the 🧨diffusers pipeline

Sample code:

```python
# Load pipeline
from wfd.wf_diffusers import WaveFunctionDiffusionPipeline
from wfd.wf_diffusers import AutoencoderTile

wfc_data_path = "tile_data/wfc/platform_32x32.npz"

# Use CUDA (otherwise it will take 15 minutes)
device = "cuda"

tilenet = AutoencoderTile.from_pretrained(
    "wdcqc/starcraft-platform-terrain-32x32",
    subfolder="tile_vae"
).to(device)
pipeline = WaveFunctionDiffusionPipeline.from_pretrained(
    "wdcqc/starcraft-platform-terrain-32x32",
    tile_vae = tilenet,
    wfc_data_path = wfc_data_path
)
pipeline.to(device)

# Generate pipeline output
# need to include the dreambooth keyword "isometric scspace terrain"
pipeline_output = pipeline(
    "isometric scspace terrain, corgi",
    num_inference_steps = 50,
    wfc_guidance_start_step = 20,
    wfc_guidance_strength = 5,
    wfc_guidance_final_steps = 20,
    wfc_guidance_final_strength = 10,
)
image = pipeline_output.images[0]

# Display raw generated image
from IPython.display import display
display(image)

# Display generated image as tiles
wave = pipeline_output.waves[0]
tile_result = wave.argmax(axis=2)

from wfd.scmap import demo_map_image
display(demo_map_image(tile_result, wfc_data_path = wfc_data_path))

# Generate map file
from wfd.scmap import tiles_to_scx
import random, time

tiles_to_scx(
    tile_result,
    "outputs/generated_{}_{:04d}.scx".format(time.strftime("%Y%m%d_%H%M%S"), random.randint(0, 1e4)),
    wfc_data_path = wfc_data_path
)
```

### Training with other tile maps

To train with other tile maps, first check out the dataset classes in `wfd/wfd/datasets.py`. You should modify the dataset classes so they output samples of your own tileset and tile images.

Starcraft has 5k+ tiles each tileset, so I did a lot of game-specific shenanigans to simplify it down. Under an easier tileset it should be fine to just randomly generate samples and convert them to images, possibly using the original WFC algorithm.

Then:

1. Train the Tile VAE (Check the config files, you can modify the hyperparameters there)

```bash
python train_tile_vae.py --image_vae <path_to_stable_diffusion_vae> --config "configs/tilenet/tilenet_sc_space_32x32.json" --output "weights/tilenet" --train_config "configs/train/train_config_32x32.json" --save_dataset_wfc <path_to_save_wfc_metadata> --data_dir <path_to_training_data> --device cuda
```

`path_to_stable_diffusion_vae` should be a downloaded `vae` folder on the original Stable Diffusion huggingface repository.

`path_to_save_wfc_metadata` is the WFC metadata that should be plugged into the pipeline, see `wfc_data_path` above.

2. Dreambooth the original `runwayml/stable-diffusion-v1-5` (For the meaning of the options you can check out the diffusers dreambooth documentation)

```bash
accelerate launch train_dreambooth.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir=<path_to_training_data> --class_data_dir=<path_to_class_images> --output_dir="checkpoints" --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="an image of isometric scspace terrain" --class_prompt="an image of isometric terrain" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=2 --gradient_checkpointing --use_8bit_adam --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=1000 --max_train_steps=10000 --checkpointing_steps 500
```

Note that I added custom options like `--append_map_name`, `--brightness_fix` to the dreambooth script. Check `python train_dreambooth.py --help` for more information.

### Never-Asked Questions

__Q: Is it necessary to train the entire VAE? Looks like the encoder can totally be skipped, only the decoding step is necessary.__

A: It's not. In hindsight, training only the decoder should be sufficient. But I already trained the encoder anyways so why not lol.

__Q: Why take argmax of the generated wave, instead of using it as a prior distribution for WFC? It should make the result more accurate.__

A: Because it doesn't work. It frequently generates impossible wave states, making WFC unsolvable (ㅠㅠㅠ)

__Q: How is this algorithm related to quantum physics?__

A: It's not. I named it wave function diffusion because Wave Function Collapse did it in the first place lmao

__Q: You seriously wasted over three weeks of time to create a terrain generator for a dead game???__

A: Technically, it can be used to generate tile maps for other games as well, should anyone train it on other datasets. However, deep down in the heart I know no one will do it anyways, so this is just my learning process of the diffusion algorithm.

In programming the best way to learn is to build something yourself, so it's not a waste of time. Plus, it's all blizz's fault anyways.

__Q: This will replace game level designing jobs and make everyone jobless!!!!🤬🤬🤬__

A: While I really don't think this thing'd be able to do it, at this point we should probably stop doing jobs and wait for our AI overlords to overtake us all

__Q: Can this algorithm solve sudoku?__

A: You should do it yourself. It's a game.
