# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import torch, torch.nn as nn
import numpy as np
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
               device = "cuda",
               autosave = 0,
               autosave_path = None,
              ):
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
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        train_loss, train_count = 0.0, 0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            tiles, images = data
            tiles = tiles.to(device).to(torch.int64)
            tiles_1h = torch.nn.functional.one_hot(tiles, num_classes = tile_count)
            tiles_1h = tiles_1h.permute(0, 3, 1, 2)
            images = images.to(device)
            
            net_encode_result = net.encode(tiles_1h.float())
            net_decode_result = net.decode(net_encode_result.latent_dist.sample())
            
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
                    param.grad.clamp_(-clamp, clamp)
                    
            optimizer.step()
            
            # zero the parameter gradients
            if i % grad_accum == 0:
                optimizer.zero_grad()

            # print statistics
            train_loss += total_loss.item()
            train_count += 1
                
            if (i + 1) % log_interval == 0:
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
            
                train_loss, train_count = 0.0, 0

            if autosave > 0 and (i + 1) % autosave == 0:
                net.save_pretrained(autosave_path)