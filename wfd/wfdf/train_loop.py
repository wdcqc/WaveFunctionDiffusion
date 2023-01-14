# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import torch, torch.nn as nn
import numpy as np
import time
from .losses import ReconstructionLoss
from .utils import offset_tensor, default_loss_weights, output_train_result

def train_loop(net,
               vae,
               train_dataset,
               epochs = 10,
               batch_size = 4,
               lr = 0.001,
               tile_count = 2048,
               clamp = 10,
               loss_weights = None,
               recon_temperature = 0.01,
               max_nudge = 5,
               nudge_loss_type = "ce",
               grad_accum = 1,
               log_interval = 100,
               log_interval_time = None,
               device = "cuda",
               autosave = 0,
               autosave_path = None,
               max_steps = None,
              ):
    r"""
    The train loop to train the VAE.

    Args:
        net (`AutoencoderTile`):
            The Tile VAE to be trained.
        vae (`AutoencoderKL`):
            The Stable Diffusion VAE which encodes/decodes images to/from latents.
        train_dataset (`torch.utils.data.Dataset` or generally any iterable):
            The dataset to train the VAE. Should return two values on __getitem__(index), where the first is the
            tile map as (h, w) sized int ndarray, and the second is the tile image as normalized (c, h, w) float
            array.
        epochs (`int`, *optional*, defaults to 10):
            Total iterations through the entire dataset. Note that the starcraft map dataset has 1m+ data points
            so it is impossible to even reach end of epoch 1, though it might be useful when training on other
            datasets.
        batch_size (`int`, *optional*, defaults to 4):
            The images used in each batch. Be careful not to burst the GPU memory.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate. TODO: try a learning rate scheduler
        tile_count (`int`, *optional*, defaults to 2048):
            The total number of tiles, which should be the in_channels and out_channels of the Tile VAE `net`.
            It can be acquired with `net.encoder.conv_in.in_channels`.
        clamp (`float`, *optional*, defaults to 10):
            Clamps the gradient to avoid it going too high and breaking the training.
        loss_weights (`dict`, *optional*):
            The weights of each type of loss. If the weight is 0, the loss is not computed. It should include
            5 keys (`match`, `kl`, `recon`, `nudge`, `ce`) corresponding to each loss.
        recon_temperature (`float`, *optional*, defaults to 0.01):
            The temperature value of reconstruction loss. Not fully understood (by me).
        max_nudge (`int`, *optional*, defaults to 5):
            The max amount to nudge the tile image for nudge_loss. Should be 5 for 32x32 and 3 for 64x64.
        nudge_loss_type (`"ce"` or `"recon"`, *optional*, defaults to `"ce"`):
            The type of loss used for nudge_loss.
        grad_accum (`int`, *optional*, defaults to 1):
            Gradient accumulation value.
        log_interval (`int`, *optional*, defaults to 100):
            Logs training result every N steps. If `log_interval_time` is used set this to None.
        log_interval_time (`float`, *optional*):
            Logs training result every N seconds. Do not set together with `log_interval`.
        device (`str`, *optional*, defaults to `"cuda"`):
            The thing that makes a dev cool.
        autosave (`int`, *optional*, defaults to 0):
            Autosave every X steps.
        autosave_path (`str`):
            Autosave path to save the model. Must be set if autosave is not 0.
        max_steps (`int`, *optional*):
            Max steps to train each epoch.
    """
    if log_interval is not None and log_interval_time is not None:
        raise ValueError("please only set one of log_interval and log_interval_time")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )
    
    if loss_weights is None:
        loss_weights = default_loss_weights()

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    recon = ReconstructionLoss()

    if log_interval_time is not None:
        prev_time = time.time()

    need_encoder = (loss_weights["kl"] != 0 or loss_weights["match"] != 0 or loss_weights["ce"] != 0 or loss_weights["recon"] != 0)
    need_full_autoencoder = (loss_weights["match"] != 0 or loss_weights["ce"] != 0 or loss_weights["recon"] != 0)

    if need_encoder: # why I feel there is a loophole here
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(net.decoder.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        train_loss, train_count = 0.0, 0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            tiles, images = data
            tiles = tiles.to(device).to(torch.int64)
            tiles_1h = torch.nn.functional.one_hot(tiles, num_classes = tile_count)
            tiles_1h = tiles_1h.permute(0, 3, 1, 2)
            images = images.to(device)
            
            if need_encoder:
                net_encode_result = net.encode(tiles_1h.float())
                if need_full_autoencoder:
                    net_decode_result = net.decode(net_encode_result.latent_dist.sample())
                else:
                    net_decode_result = 0
            else:
                net_encode_result = 0
                net_decode_result = 0
            
            # calculate losses
            total_loss = 0
            
            if loss_weights["match"] != 0:
                with torch.no_grad():
                    img_encode_result = vae.encode(images)
                    img_mean = img_encode_result.latent_dist.mean
                match_loss = mse(net_encode_result.latent_dist.mean, img_mean)
                total_loss += match_loss * loss_weights["match"]
            else:
                match_loss = 0
            
            if loss_weights["kl"] != 0:
                kl_loss = net_encode_result.latent_dist.kl().mean() * 1e-6
                total_loss += kl_loss * loss_weights["kl"]
            else:
                kl_loss = 0
            
            if loss_weights["nudge"] != 0:
                images = offset_tensor(images, max_nudge)
                with torch.no_grad():
                    img_encode_result = vae.encode(images)
                    img_latents = img_encode_result.latent_dist.sample()
                    
                latents_decode_result = net.decode(img_latents)
                if nudge_loss_type == "ce":
                    nudge_loss = ce(latents_decode_result.sample, tiles)
                elif nudge_loss_type == "recon":
                    nudge_loss = recon(latents_decode_result.sample, tiles_1h, recon_temperature)
                else:
                    raise NotImplementedError
                total_loss += nudge_loss * loss_weights["nudge"]
            else:
                nudge_loss = 0
            
            if loss_weights["ce"] != 0:
                ce_loss = ce(net_decode_result.sample, tiles)
                total_loss += ce_loss * loss_weights["ce"]
            else:
                ce_loss = 0
            
            if loss_weights["recon"] != 0:
                recon_loss = recon(net_decode_result.sample, tiles_1h, recon_temperature)
                total_loss += recon_loss * loss_weights["recon"]
            else:
                recon_loss = 0

            # backward
            total_loss.backward()
            
            # clamp gradients
            if clamp is not None:
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad.clamp_(-clamp, clamp)
                    
            optimizer.step()
            
            # zero the parameter gradients
            if i % grad_accum == 0:
                optimizer.zero_grad()

            # print statistics
            train_loss += total_loss.item()
            train_count += 1
                
            should_log_on_step = (log_interval is not None) and ((i + 1) % log_interval == 0)
            should_log_on_time = (log_interval_time is not None) and (time.time() > log_interval_time + prev_time)
            if should_log_on_step or should_log_on_time:
                output_train_result(
                    epoch = epoch + 1,
                    step = i + 1,
                    loss_weights = loss_weights,
                    match_loss = match_loss,
                    kl_loss = kl_loss,
                    nudge_loss = nudge_loss,
                    ce_loss = ce_loss,
                    recon_loss = recon_loss,
                    train_loss = train_loss / train_count,
                    log_handler = print,
                )

                prev_time = time.time()
            
                train_loss, train_count = 0.0, 0

            if autosave > 0 and (i + 1) % autosave == 0:
                net.save_pretrained(autosave_path)

            if max_steps is not None and i >= max_steps:
                break

    # End training
    if autosave > 0:
        net.save_pretrained(autosave_path)