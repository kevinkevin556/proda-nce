from dataclasses import dataclass
from typing import Sequence, TypedDict

import numpy as np
import torch
from monai.networks.utils import one_hot
from torch import nn

from lib.tensor_shape import array, tensor


@dataclass
class ModuleInfo:
    module: nn.Module
    background: Sequence
    pretrained: str


class MultiNets(nn.Module):
    """
    A module used to synthesize the predictions from two domain adaptation networks.
    """

    def __init__(self, net1: ModuleInfo, net2: ModuleInfo, num_classes: int, device: str = "cuda"):
        super().__init__()

        # Load net 1
        self.net1 = net1.module.to(device)
        if getattr(self.net1, "load", False):
            self.net1.load(net1.pretrained)
        else:
            self.net1.load_state_dict(torch.load(net1.pretrained))
        self.net1.eval()

        # Load net 2
        self.net2 = net2.module.to(device)
        if getattr(self.net2, "load", False):
            self.net2.load(net2.pretrained)
        else:
            self.net2.load_state_dict(torch.load(net2.pretrained))
        self.net2.eval()

        # Register class info
        self.backgound_class = {"net1": net1.background, "net2": net2.background}
        self.num_classes = num_classes
        print(f"Net 1 background = {str(net1.background)}, Net 2 background = {str(net2.background)}")

    def inference(self, x: tensor["1 1 w h"], modality=None):
        # The inplementation is currently for 2d images

        # get predictions from both nets
        y1: tensor["1 c w h"] = self.net1.inference(x) if getattr(self.net1, "inference", False) else self.net1(x)
        y2: tensor["1 c w h"] = self.net2.inference(x) if getattr(self.net2, "inference", False) else self.net2(x)
        y1_argmax: array["w h"] = torch.argmax(y1, dim=1).cpu().numpy().squeeze()
        y2_argmax: array["w h"] = torch.argmax(y2, dim=1).cpu().numpy().squeeze()

        # Merge two predictions
        pred: array["w h"] = (y1_argmax + y2_argmax).astype(float)
        overlap: array["w h"] = np.logical_and(y1_argmax > 0, y2_argmax > 0)
        pred[overlap] = np.nan

        # If two predictions both mark a pixel positive,
        #   determine the class of the pixel by its neighborhoods.
        overlap_index = np.asarray(overlap).nonzero()
        if modality is None:
            for a, b in zip(*overlap_index):
                pred[a, b] = classify_undetermined_point(pred, (a, b))
        if modality == "ct":
            pred[overlap] = y1_argmax[overlap]
        if modality == "mr":
            pred[overlap] = y2_argmax[overlap]
        assert np.isnan(pred).sum() == 0, "Some points are still undertermined."

        # Obtain one-hot-encoded tensors
        pred: tensor["1 w h"] = torch.LongTensor(pred[None, :]).to(x.device)
        pred: tensor["1 c w h"] = one_hot(pred, num_classes=self.num_classes, dim=0)[None, :]
        return pred


def classify_undetermined_point(mask: np.ndarray, point: tuple, max_iter: int = 50) -> int:
    """Classifies the undetermined point by its nearby pixels.
      An undetermined pixel is labelled as the most frequent class among
      all its searched neighbors. For example,
      ```
      1    2    3
      1    *    NaN
      NaN  NaN  1
      ```
      The \*  pixel is labelled as class 1.

    Args:
        mask (np.ndarray): A mask of ground truth.
        point (tuple): The coordinate of the undetermined point
        max_iter (int, optional): The range to search.. Defaults to 30.

    Returns:
        int: the class of undetermined point.
    """
    x, y = point
    for k in range(max_iter):
        search_region = mask[(x - k) : (x + k + 1), (y - k) : (y + k + 1)]
        search_region = search_region[~np.isnan(search_region)]
        search_region = search_region[search_region != 0]
        found_class, counts = np.unique(search_region, return_counts=True)
        # Classes other than the undetermined type are found
        # return the one with the greatest count
        if len(found_class) != 0:
            return found_class[np.argmax(counts)]
