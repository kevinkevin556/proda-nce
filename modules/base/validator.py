from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import tqdm
from medaset.transforms import BackgroundifyClasses
from monai.data import DataLoader, decollate_batch
from monai.metrics import Metric
from monai.transforms import AsDiscrete, Compose
from torch import nn
from tqdm.auto import tqdm


class BaseValidator:
    """
    The base class of validators.
    It is intended to be a template for users to create their own validator class.
    """

    def __init__(
        self,
        metric: Metric,
        is_train: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
        pred_logits: bool = True,
    ):
        """
        Args:
            metric (monai.Metric): The metric used to evaluate the model's performance.
            is_train (bool): Flag indicating if the validator is for training. Defaults to False.
            device (Literal["cuda", "cpu"]): The device to use for validation ('cuda' or 'cpu'). Defaults to 'cuda'.
            pred_logits (bool): Flag indicating if the model's predictions are logits. Defaults to True.
        """
        self.metric = metric
        self.is_train = is_train
        self.device = device
        self.pred_logits = pred_logits  # if model's preds are logits, set this to true

        if is_train:
            self.pbar_description = "Validate ({global_step} Steps) (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"
        else:
            self.pbar_description = "Validate (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"

    def __call__(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> dict:
        return self.validation(module, dataloader, global_step)

    def validation(
        self,
        module: nn.Module,
        dataloader: DataLoader | Sequence[DataLoader],
        global_step: int | None = None,
    ) -> dict:
        """
        Perform the validation process.

        Args:
            module (nn.Module): The model to be validated.
            dataloader (DataLoader | Sequence[DataLoader]): The dataloader(s) providing the validation data.
            global_step (int | None): The current global step, if applicable.

        Returns:
            dict: A dictionary containing the mean validation metrics for 'ct' and 'mr' modalities and their overall mean.
        """

        module.eval()
        val_metrics = {"ct": [], "mr": []}
        metric_means = {"mean": None, "ct": None, "mr": None}

        if not isinstance(dataloader, (list, tuple)):
            dataloader = [dataloader]
        else:
            dataloader = [dl for dl in dataloader if dl is not None]
        data_iter = itertools.chain(*dataloader)
        pbar = tqdm(
            data_iter,
            total=sum(len(dl) for dl in dataloader),
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in pbar:
                # Infer, decollate data into list of samples, and postprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                modality_label = batch["modality"][0]
                num_classes = int(batch["num_classes"])
                background_classes = batch["background_classes"].numpy().flatten()

                assert modality_label in set(["ct", "mr"]), f"Unknown/Invalid modality {modality_label}"
                assert 0 in background_classes, "0 should be included in background_classes"

                # Get inferred / forwarded results of module
                if getattr(module, "inference", False):
                    infer_out = module.inference(images, modality=modality_label)
                else:
                    infer_out = module.forward(images)

                # Discretize the prediction and masks of ground truths
                samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})
                preds: list = discretize_and_backgroundify_preds(samples, num_classes, background_classes)
                masks: list = discretize_and_backgroundify_masks(samples, num_classes, background_classes)

                # Compute validation metrics
                self.metric(y_pred=preds, y=masks)
                batch_metric = self.metric.aggregate().item()
                val_metrics[modality_label] += [batch_metric]
                self.metric.reset()

                # Update progressbar
                info = {
                    "val_on_partial": set(background_classes) > set([0]),
                    "metric_name": self.metric.__class__.__name__,
                    "batch_metric": batch_metric,
                    "global_step": global_step,
                }
                desc = self.pbar_description.format(**info)
                pbar.set_description(desc)

        metric_means["mean"] = np.mean(val_metrics["ct"] + val_metrics["mr"])
        metric_means["ct"] = np.mean(val_metrics["ct"]) if len(val_metrics["ct"]) > 0 else np.nan
        metric_means["mr"] = np.mean(val_metrics["mr"]) if len(val_metrics["mr"]) > 0 else np.nan
        return metric_means


def discretize_and_backgroundify_preds(samples: Sequence, num_classes: int, background: Sequence = (0,)) -> list:
    """
    Argmax and one-hot-encoded the given prediction logits. Masked the background class if needed.

    Args:
        samples (Sequence): The input samples containing predictions and ground truth masks.
        num_classes (int): The number of classes for one-hot encoding.
        background (Sequence): The background classes to be considered during postprocessing. Defaults to (0,).

    Returns:
        List[torch.Tensor]: Processed predictions.
    """
    if isinstance(background, (np.ndarray, torch.Tensor)):
        background = background.tolist()
    if isinstance(background, tuple):
        background = list(background)
    assert isinstance(background, list)

    if background != [0]:
        postprocess_pred = Compose(
            AsDiscrete(argmax=True, to_onehot=num_classes),
            BackgroundifyClasses(channel_dim=0, classes=background),
        )
    else:
        postprocess_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    preds = [postprocess_pred(sample["prediction"]) for sample in samples]
    return preds


def discretize_and_backgroundify_masks(samples: Sequence, num_classes: int, background: Sequence = (0,)) -> list:
    """
    One-hot-encoded the given ground truth masks. Masked the background class if needed.

    Args:
        samples (Sequence): The input samples containing predictions and ground truth masks.
        num_classes (int): The number of classes for one-hot encoding.
        background (Sequence): The background classes to be considered during postprocessing. Defaults to (0,).

    Returns:
        List[torch.Tensor]: Processed masks.
    """
    if isinstance(background, (np.ndarray, torch.Tensor)):
        background = background.tolist()
    if isinstance(background, tuple):
        background = list(background)
    assert isinstance(background, list)

    if background != [0]:
        postprocess_mask = Compose(
            AsDiscrete(to_onehot=num_classes),
            BackgroundifyClasses(channel_dim=0, classes=background),
        )
    else:
        postprocess_mask = AsDiscrete(to_onehot=num_classes)

    masks = [postprocess_mask(sample["ground_truth"]) for sample in samples]
    return masks
