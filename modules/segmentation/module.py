from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Union

import torch
from monai.data import DataLoader as MonaiDataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader as PyTorchDataLoader

DataLoader = Union[MonaiDataLoader, PyTorchDataLoader]


class SegmentationModule(nn.Module):
    alias = "SegNet"

    def __init__(
        self,
        net: nn.Module,
        roi_size: tuple,
        sw_batch_size: int,
        criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.net = net
        self.criterion = criterion
        self.lr = lr
        self.device = device

        params = self.net.parameters()
        differentiable_params = [p for p in params if p.requires_grad]
        # TODO: replace these assignment with partials
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)

        if pretrained:
            self.load(pretrained)

        self.to(device)

    def forward(self, x):
        y = self.net(x)
        return y

    def inference(self, x, modality):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(checkpoint_dir, "net.pth"))

    def load(self, checkpoint_dir):
        self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net.pth")))

    def print_info(self):
        print("Module:", self.net.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", repr(self.criterion))


class SegmentationEncoderDecoder(nn.Module):
    alias = "SegEncoderDecoder"

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        roi_size: tuple,
        sw_batch_size: int,
        criterion: _Loss = DiceCELoss(to_onehot_y=True, softmax=True),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
    ):
        super().__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.criterion = criterion
        self.lr = lr

        self.encoder = encoder
        self.decoder = decoder

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        differentiable_params = [p for p in params if p.requires_grad]
        # TODO: replace these assignment with partials
        if optimizer == "AdamW":
            self.optimizer = AdamW(differentiable_params, lr=self.lr)
        if optimizer == "Adam":
            self.optimizer = Adam(differentiable_params, lr=self.lr)
        if optimizer == "SGD":
            self.optimizer = SGD(differentiable_params, lr=self.lr)

    def forward(self, x):
        encoded = self.encoder(x)
        y = self.decoder(*encoded) if isinstance(encoded, (list, tuple)) else self.decoder(encoded)
        return y

    def inference(self, x):
        # Using sliding windows
        self.eval()
        return sliding_window_inference(x, self.roi_size, self.sw_batch_size, self.forward)

    def save(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint_dir, "encoder_state.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(checkpoint_dir, "decoder_state.pth"))

    def load(self, checkpoint_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "encoder_state.pth")))
        self.decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "decoder_state.pth")))

    def print_info(self):
        print("Module Encoder:", self.encoder.__class__.__name__)
        print("       Decoder:", self.decoder.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function:", repr(self.criterion))
