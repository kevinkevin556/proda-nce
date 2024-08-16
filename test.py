from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from jsonargparse import CLI
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torch import nn

from lib.datasets.dataset_wrapper import Dataset
from modules.base.validator import BaseValidator
from modules.validator.seg_vizualizer import SegVizualizer
from modules.validator.summary import SummmaryValidator


def setup(
    ct_data: Dataset,
    mr_data: Dataset,
    module: nn.Module,
    pretrained: str | None = None,
    evaluator: BaseValidator | str = None,
    device: str = "cuda",
):
    module = module.to(device)

    # Load pretrained module / network
    if pretrained is not None:
        if getattr(module, "load", False):
            module.load(pretrained)
        else:
            module = module.to(device)
            module.load_state_dict(torch.load(pretrained))

    # default evaluator for testing set: SummaryValidator

    return ct_data, mr_data, module, evaluator, pretrained


def main():
    ct_data, mr_data, module, evaluator, pretrained = CLI(setup, parser_mode="omegaconf")
    ct_dataloader = ct_data.get_data()
    mr_dataloader = mr_data.get_data()

    if Path(pretrained).is_dir():
        pass
    else:
        pretrained = str(Path(pretrained).parents[0])

    if evaluator is None or evaluator == "dice" or evaluator == "all":
        num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
        dice_evaluator = SummmaryValidator(
            metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
            num_classes=num_classes,
        )
        dice = dice_evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
        print(dice)
        dice.to_csv(f"{pretrained}/dice.csv")

    if evaluator == "hausdorff" or evaluator == "all":
        num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
        hausdorff_evaluator = SummmaryValidator(
            metric=HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False),
            num_classes=num_classes,
        )
        hausdorff = hausdorff_evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
        print(hausdorff)
        hausdorff.to_csv(f"{pretrained}/hausdorff.csv")

    if evaluator == "image" or evaluator == "all":
        num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
        evaluator = SegVizualizer(
            num_classes=num_classes,
            output_dir=f"{pretrained}/images/",
            ground_truth=False,
        )
        performance = evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))


if __name__ == "__main__":
    main()
    # CLI(main, parser_mode="omegaconf", formatter_class=RichHelpFormatter)
