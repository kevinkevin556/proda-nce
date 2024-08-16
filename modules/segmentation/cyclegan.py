# TODO Rewrite the updater
# This is buggy.

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch import nn
from torch.nn.modules.loss import _Loss

from lib.cyclegan.utils import load_cyclegan
from lib.loss.target_adaptative_loss import TargetAdaptativeLoss
from networks.cyclegan.cycle_gan_model import CycleGANModel
from networks.unet import BasicUNet

from .module import SegmentationModule
from .updater import SegmentationUpdater


class CycleGanSegmentationModule(SegmentationModule):
    def __init__(
        self,
        cyclegan_checkpoints_dir: str,
        net: nn.Module = BasicUNet(spatial_dims=2, in_channels=1, out_channels=4, features=(32, 32, 64, 128, 256, 32)),
        roi_size: tuple = (512, 512),
        sw_batch_size: int = 1,
        ct_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 1, 2]),
        mr_criterion: _Loss = TargetAdaptativeLoss(num_classes=4, background_classes=[0, 3]),
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        device: Literal["cuda", "cpu"] = "cuda",
        pretrained: Path | None = None,
    ):
        super().__init__(
            net=net,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            criterion=None,
            optimizer=optimizer,
            lr=lr,
            device=device,
            pretrained=pretrained,
        )
        self.ct_criterion = ct_criterion
        self.mr_criterion = mr_criterion
        # self.cycle_gan = CycleGANModel()
        self.cyclegan = load_cyclegan(cyclegan_checkpoints_dir, which_epoch="latest")

    def print_info(self):
        print("Module:", self.net.__class__.__name__)
        print("Optimizer:", self.optimizer.__class__.__name__, f"(lr = {self.lr})")
        print("Loss function - CT:", repr(self.ct_criterion))
        print("Loss function - MR:", repr(self.mr_criterion))


class CycleGanSegmentationUpdater(SegmentationUpdater):
    def __init__(self):
        super().__init__()
        self.sampling_mode = "sequential"

    def check_module(self, module):
        assert isinstance(module, nn.Module), "The specified module should inherit torch.nn.Module."
        assert isinstance(
            module, CycleGanSegmentationModule
        ), "The specified module should inherit CycleGanSegmentationModule."
        for component in ("ct_criterion", "mr_criterion", "optimizer", "cyclegan"):
            assert getattr(
                module, component, False
            ), f"The specified module should incoporate component/method: {component}"

    def update(self, module, images, masks, modalities):
        module.optimizer.zero_grad()

        ct, mr = "A", "B"

        if modalities == 0:
            # Train network with fake MR scans (generated from CT)
            fake_mr = module.cyclegan.generate_image(input_image=images, from_domain=ct)
            ct_images, ct_mask = fake_mr, masks
            ct_output = module.net(ct_images)
            seg_loss = module.ct_criterion(ct_output, ct_mask)
            seg_loss += module.ct_criterion(images, ct_mask)
        else:
            # Train network with real MR scans
            fake_ct = module.cyclegan.generate_image(input_image=images, from_domain=mr)
            mr_images, mr_mask = fake_ct, masks
            mr_output = module.net(mr_images)
            seg_loss = module.mr_criterion(mr_output, mr_mask)
            seg_loss += module.mr_criterion(images, mr_mask)

        # Back-prop
        seg_loss.backward()
        module.optimizer.step()
        return seg_loss.item()
