# -*- coding: utf-8 -*-
# Copyright @ 2023 wdcqc/aieud project.
# Open-source under license obtainable in project root (LICENSE.md).

import torch
import einops as eo
import numpy as np

class WFCLoss(torch.nn.Module):
    """
    Computes a value on a wave that corresponds to the expectation of violation to WFC rules.
    This is slow, use the other versions instead.

    Parameters:
        wfc_mat_x (`2d tensor-like`):
            Transition matrix over horizontal terrain grids (x[i,j] == 1 means tile j is allowed at right side of tile i).
        wfc_mat_y (`2d tensor-like`):
            Transition matrix over vertical terrain grids (y[i,j] == 1 means tile j is allowed at bottom of tile i).
    """
    def __init__(self, wfc_mat_x, wfc_mat_y):
        super().__init__()
        cx = torch.as_tensor(wfc_mat_x, dtype=torch.float32)
        cy = torch.as_tensor(wfc_mat_y, dtype=torch.float32)
        self.register_buffer("cx", eo.rearrange(1 - cx, "a b -> 1 1 1 a b"))
        self.register_buffer("cy", eo.rearrange(1 - cy, "a b -> 1 1 1 a b"))
        self.count = cx.size(0)
        assert cx.size(0) == cx.size(1) and cx.size(1) == cy.size(0) and cy.size(0) == cy.size(1), "WFC matrix size mismatch"
        
    def forward(self, t, softmax=False):
        if softmax:
            t = torch.softmax(t, dim=1)
            
        x_l = eo.rearrange(t[:, :self.count, :, :-1], "b i y x -> b y x 1 i")
        x_r = eo.rearrange(t[:, :self.count, :, 1:], "b i y x -> b y x i 1")
        sum_x = torch.sum(x_l @ self.cx @ x_r, dim=(1, 2, 3, 4))

        y_t = eo.rearrange(t[:, :self.count, :-1, :], "b i y x -> b y x 1 i")
        y_b = eo.rearrange(t[:, :self.count,  1:, :], "b i y x -> b y x i 1")
        sum_y = torch.sum(y_t @ self.cy @ y_b, dim=(1, 2, 3, 4))
        
        return (sum_x + sum_y) / (x_l.size(1) * x_l.size(2) + y_t.size(1) * y_t.size(2))

class WFCLossEinsum(torch.nn.Module):
    """
    Computes a value on a wave that corresponds to the expectation of violation to WFC rules.
    Einsum version, which is faster than the original WFCLoss.

    Parameters:
        wfc_mat_x (`2d tensor-like`):
            Transition matrix over horizontal terrain grids (x[i,j] == 1 means tile j is allowed at right side of tile i).
        wfc_mat_y (`2d tensor-like`):
            Transition matrix over vertical terrain grids (y[i,j] == 1 means tile j is allowed at bottom of tile i).
    """
    def __init__(self, wfc_mat_x, wfc_mat_y):
        super().__init__()
        cx = torch.as_tensor(wfc_mat_x, dtype=torch.float32)
        cy = torch.as_tensor(wfc_mat_y, dtype=torch.float32)
        self.register_buffer("cx", eo.rearrange(1 - cx, "a b -> 1 a b 1 1"))
        self.register_buffer("cy", eo.rearrange(1 - cy, "a b -> 1 a b 1 1"))
        self.count = cx.size(0)
        assert cx.size(0) == cx.size(1) and cx.size(1) == cy.size(0) and cy.size(0) == cy.size(1), "WFC matrix size mismatch"
        
    def forward(self, t, softmax=False):
        if softmax:
            t = torch.softmax(t, dim=1)
            
        x_l = t[:, :self.count, :, :-1]
        x_r = t[:, :self.count, :, 1:]
        ein_x = torch.einsum("biyx, bjyx, bijyx -> b", x_l, x_r, self.cx)

        y_t = t[:, :self.count, :-1, :]
        y_b = t[:, :self.count,  1:, :]
        ein_y = torch.einsum("biyx, bjyx, bijyx -> b", y_t, y_b, self.cy)
        
        return (ein_x + ein_y) / (x_l.size(2) * x_l.size(3) + y_t.size(2) * y_t.size(3))

class WFCLossBilinear(torch.nn.Module):
    """
    Computes a value on a wave that corresponds to the expectation of violation to WFC rules.
    Bilinear layer version, which is faster than the original WFCLoss.

    Parameters:
        wfc_mat_x (`2d tensor-like`):
            Transition matrix over horizontal terrain grids (x[i,j] == 1 means tile j is allowed at right side of tile i).
        wfc_mat_y (`2d tensor-like`):
            Transition matrix over vertical terrain grids (y[i,j] == 1 means tile j is allowed at bottom of tile i).
    """
    def __init__(self, wfc_mat_x, wfc_mat_y):
        super().__init__()
        cx = torch.as_tensor(wfc_mat_x, dtype=torch.float32)
        cy = torch.as_tensor(wfc_mat_y, dtype=torch.float32)
        self.bilinear_x = torch.nn.Bilinear(cx.shape[0], cx.shape[1], 1, bias=False)
        self.bilinear_y = torch.nn.Bilinear(cy.shape[0], cy.shape[1], 1, bias=False)
        self.count = cx.size(0)
        assert cx.size(0) == cx.size(1) and cx.size(1) == cy.size(0) and cy.size(0) == cy.size(1), "WFC matrix size mismatch"
        
        with torch.no_grad():
            next(self.bilinear_x.parameters())[:] = 1 - cx
            next(self.bilinear_y.parameters())[:] = 1 - cy
        
        self.bilinear_x.requires_grad_(False)
        self.bilinear_y.requires_grad_(False)
        
    def forward(self, t, softmax=False):
        if softmax:
            t = torch.softmax(t, dim=1)
            
        x_l = eo.rearrange(t[:, :self.count, :, :-1], "b i y x -> b y x i")
        x_r = eo.rearrange(t[:, :self.count, :,  1:], "b i y x -> b y x i")
        bil_x = self.bilinear_x(x_l, x_r)
        sum_x = torch.sum(bil_x, dim=(1, 2, 3))
        
        y_t = eo.rearrange(t[:, :self.count, :-1, :], "b i y x -> b y x i")
        y_b = eo.rearrange(t[:, :self.count,  1:, :], "b i y x -> b y x i")
        bil_y = self.bilinear_y(y_t, y_b)
        sum_y = torch.sum(bil_y, dim=(1, 2, 3))
        
        return (sum_x + sum_y) / (x_l.size(1) * x_l.size(2) + y_t.size(1) * y_t.size(2))

class ReconstructionLoss(torch.nn.Module):
    """
    Computes the reconstruction loss.
    Taken from DD-VAE (Deterministic Decoding for Discrete Data in Variational Autoencoders) codebase by Daniil Polykovskiy and Dmitry Vetrov.
    """
    def __init__(self):
        super().__init__()

    def smoothed_log_indicator(self, x, temperature):
        return torch.nn.functional.softplus(-x/temperature + np.log(1/temperature - 1))

    def forward(self, y_pred, y_true, temperature=0.01):
        if len(y_true.shape) == 3:
            y_hot = torch.nn.functional.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2)
        else:
            y_hot = y_true
        p_correct = torch.sum(y_pred * y_hot, dim=1)
        delta = p_correct - (1 - p_correct)
        reconstruction_loss = self.smoothed_log_indicator(delta, temperature).mean()
        return reconstruction_loss