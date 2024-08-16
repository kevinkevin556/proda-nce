import datetime
import inspect
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

from jsonargparse import CLI, ArgumentParser
from jsonargparse.typing import Path_fr
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils import set_determinism
from ruamel.yaml import YAML
from torch import nn

from lib.datasets.dataset_wrapper import Dataset
from modules.base.trainer import BaseTrainer
from modules.base.updater import BaseUpdater
from modules.base.validator import BaseValidator
from modules.validator.seg_vizualizer import SegVizualizer
from modules.validator.summary import SummmaryValidator


def setup(
    ct_data: Dataset,
    mr_data: Dataset,
    module: nn.Module,
    updater: BaseUpdater,
    trainer: BaseTrainer,
    evaluator: BaseValidator = None,
    device: str = "cuda",
    dev: bool = False,
    deterministic: bool = False,
):
    if deterministic:
        set_determinism(seed=0)
    if dev:
        os.environ["MONAI_DEBUG"] = "True"

    # checkpoint name
    suffix = "{time}_{module}_{trainer}_{updater}_LR{lr}_{optimizer}_{ct_dataset}_{mr_dataset}_Step{step}"
    info = {
        "time": datetime.now().strftime("%Y%m%d_%H%M"),
        "module": getattr(module, "alias", module.__class__.__name__),
        "trainer": trainer.get_alias(),
        "updater": updater.get_alias(),
        "lr": module.lr,
        "optimizer": module.optimizer.__class__.__name__,
        "ct_dataset": ct_data.__class__.__name__ if ct_data.in_use else "null",
        "mr_dataset": mr_data.__class__.__name__ if mr_data.in_use else "null",
        "step": trainer.max_iter,
    }
    trainer.checkpoint_dir += suffix.format(**info)

    # default evaluator for testing set: SummaryValidator
    if evaluator is None:
        num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
        evaluator = SummmaryValidator(
            metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
            num_classes=num_classes,
        )
    return ct_data, mr_data, module, trainer, updater, evaluator


def save_config_to(dir_path):
    Path(dir_path).mkdir(exist_ok=True, parents=True)
    target_path = os.path.join(dir_path, "config.yml")

    parser = ArgumentParser()
    parser.add_argument("--config", type=Path_fr)
    cfg_path = parser.parse_args().config

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    with open(cfg_path, "r", encoding="utf8") as stream:
        cfg_data = yaml.load(stream)
    with open(target_path, "w", encoding="utf8") as file:
        yaml.dump(cfg_data, file)


def save_source_to(dir_path, objects):
    dir_path = Path(dir_path) / "source"
    dir_path.mkdir(exist_ok=True, parents=True)
    source_files = set(Path(inspect.getsourcefile(obj.__class__)) for obj in objects if obj is not None)
    for file in source_files:
        shutil.copy(file, dir_path / file.name)


def main():
    ct_data, mr_data, module, trainer, updater, evaluator = CLI(setup, parser_mode="omegaconf")
    num_classes = getattr(ct_data, "num_classes", getattr(mr_data, "num_classes", None))
    assert num_classes is not None

    components = [module, trainer, updater, evaluator]
    if ct_data.in_use:
        components += [ct_data, ct_data.train_transform, ct_data.test_transform]
    if mr_data.in_use:
        components = [mr_data, mr_data.train_transform, mr_data.test_transform]
    save_config_to(trainer.checkpoint_dir)
    save_source_to(trainer.checkpoint_dir, objects=components)
    ct_dataloader = ct_data.get_data()
    mr_dataloader = mr_data.get_data()
    trainer.train(
        module,
        updater,
        ct_dataloader=ct_dataloader,
        mr_dataloader=mr_dataloader,
    )
    # performance = evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
    # print(performance)

    dice_evaluator = SummmaryValidator(
        metric=DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        num_classes=num_classes,
    )
    dice = dice_evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
    print(dice)
    dice.to_csv(f"{trainer.checkpoint_dir}/dice.csv")

    print("---------------------------")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hausdorff_evaluator = SummmaryValidator(
            metric=HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False),
            num_classes=num_classes,
        )
        hausdorff = hausdorff_evaluator.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))
        print(hausdorff)
        hausdorff.to_csv(f"{trainer.checkpoint_dir}/hausdorff.csv")

    visualizer = SegVizualizer(
        num_classes=num_classes,
        output_dir=f"{trainer.checkpoint_dir}/images",
        ground_truth=True,
    )
    visualizer.validation(module, dataloader=(ct_dataloader[2], mr_dataloader[2]))


if __name__ == "__main__":
    main()
    # CLI(main, parser_mode="omegaconf", formatter_class=RichHelpFormatter)
