# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import torch
import numpy as np

def offset_tensor(image_tensor, max_offset):
    x, y = np.random.randint(-max_offset, max_offset+1, size=(2,))
    if y != 0:
        image_tensor = torch.cat([image_tensor[:, :, y:, :], image_tensor[:, :, :y, :]], dim=2)
    if x != 0:
        image_tensor = torch.cat([image_tensor[:, :, :, x:], image_tensor[:, :, :, :x]], dim=3)
    return image_tensor

def default_loss_weights():
    return {
        "ce": 0.5,
        "match": 0.5,
        "recon": 0,
        "kl": 0,
        "nudge": 0,
    }

def output_train_result(
    epoch, step, loss_weights, match_loss, kl_loss, nudge_loss, ce_loss, recon_loss, train_loss,
    log_handler = print,
):
    result = "[EP{:3d} STEP{:6d}]".format(epoch, step)
    if loss_weights["match"] != 0:
        result += " match_loss: {:.6f}".format(match_loss.item())
    if loss_weights["kl"] != 0:
        result += " kl_loss: {:.6f}".format(kl_loss.item())
    if loss_weights["nudge"] != 0:
        result += " nudge_loss: {:.6f}".format(nudge_loss.item())
    if loss_weights["ce"] != 0:
        result += " ce_loss: {:.6f}".format(ce_loss.item())
    if loss_weights["recon"] != 0:
        result += " recon_loss: {:.6f}".format(recon_loss.item())
    result += " total_loss: {:.6f}".format(train_loss)
    log_handler(result)

def softload_weights(model, path, excluded_layers = []):
    excluded_weights = [lay + ".weight" for lay in excluded_layers] + [lay + ".bias" for lay in excluded_layers]
    state_dict = torch.load(path)

    for w in excluded_weights:
        if w in state_dict:
            del state_dict[w]

    load_result = model.load_state_dict(state_dict, strict=False)
    return load_result