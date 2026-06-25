from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from custom_dataset import CustomObjectDataset


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "data" / "custom_object"
WEIGHTS_DIR = PROJECT_DIR / "weights"
EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "custom_smoke"
LOG_PATH = EXPERIMENT_DIR / "training_log.csv"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"

NUM_CLASSES = 2
NUM_EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 0.0025
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
RANDOM_SEED = 42

LOSS_KEYS = [
    "total_loss",
    "loss_classifier",
    "loss_box_reg",
    "loss_mask",
    "loss_objectness",
    "loss_rpn_box_reg",
]


def set_random_seed(seed: int) -> None:
    """固定随机种子，让冒烟训练尽量可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """优先使用 CUDA；不可用时回退到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple:
    """把 detection 数据整理成 images list 和 targets list。"""

    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes: int) -> torch.nn.Module:
    """创建 COCO 预训练 Mask R-CNN，并替换为自制数据集类别数。"""

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_in_features,
        num_classes,
    )

    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features,
        hidden_layer,
        num_classes,
    )

    return model


def move_targets_to_device(
    targets: tuple[dict[str, torch.Tensor], ...],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """把 target 中的所有张量移动到训练设备。"""

    return [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]


def validate_loss(loss_value: torch.Tensor, batch_index: int) -> None:
    """发现 NaN 或无穷 loss 时立即停止。"""

    if not torch.isfinite(loss_value):
        raise RuntimeError(
            f"第 {batch_index} 个 batch 出现非法 loss：{float(loss_value.detach().cpu())}"
        )


def average_losses(loss_sums: dict[str, float], batch_count: int) -> dict[str, float]:
    """计算每个 epoch 的平均 loss。"""

    return {
        key: loss_sums[key] / batch_count if batch_count else math.nan
        for key in LOSS_KEYS
    }


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """训练一个 epoch，并返回各项平均 loss。"""

    model.train()
    loss_sums = {key: 0.0 for key in LOSS_KEYS}
    batch_count = 0

    for batch_index, (images, targets) in enumerate(data_loader, start=1):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        validate_loss(total_loss, batch_index)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_count += 1
        loss_sums["total_loss"] += float(total_loss.detach().cpu())
        for key in LOSS_KEYS[1:]:
            loss_sums[key] += float(loss_dict[key].detach().cpu())

        print(
            f"epoch {epoch} batch {batch_index}/{len(data_loader)} "
            f"total_loss={float(total_loss.detach().cpu()):.4f}"
        )

    return average_losses(loss_sums, batch_count)


def save_training_log(rows: list[dict[str, Any]]) -> None:
    """保存每轮训练的 loss 日志。"""

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "batch_count", *LOSS_KEYS, "epoch_time_sec", "weight_path"]
    with LOG_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_config(device: torch.device, train_count: int) -> None:
    """保存本次冒烟训练配置。"""

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_dir": str(DATASET_DIR),
        "num_classes": NUM_CLASSES,
        "class_names": {"0": "background", "1": "custom_object"},
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "train_sample_count": train_count,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "note": "2 epoch smoke training for pipeline validation, not final training.",
    }
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """主流程：加载自制数据集，完成 2 个 epoch 冒烟训练。"""

    set_random_seed(RANDOM_SEED)
    device = get_device()
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"CUDA 设备：{torch.cuda.get_device_name(0)}")

    dataset = CustomObjectDataset(DATASET_DIR, split="train")
    if len(dataset) == 0:
        raise RuntimeError("训练集为空，无法进行冒烟训练。")

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"训练样本数：{len(dataset)}")
    print("正在建立 COCO 预训练 Mask R-CNN……")
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    save_config(device=device, train_count=len(dataset))

    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()
        losses = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        epoch_time_sec = time.perf_counter() - epoch_start

        weight_path = WEIGHTS_DIR / f"custom_smoke_epoch_{epoch}.pth"
        torch.save(model.state_dict(), weight_path)
        print(f"已保存权重：{weight_path}")

        row = {
            "epoch": epoch,
            "batch_count": len(data_loader),
            **{key: f"{value:.6f}" for key, value in losses.items()},
            "epoch_time_sec": f"{epoch_time_sec:.2f}",
            "weight_path": str(weight_path),
        }
        log_rows.append(row)
        save_training_log(log_rows)

        print(
            f"epoch {epoch} 完成："
            f"total_loss={losses['total_loss']:.4f}, "
            f"loss_classifier={losses['loss_classifier']:.4f}, "
            f"loss_box_reg={losses['loss_box_reg']:.4f}, "
            f"loss_mask={losses['loss_mask']:.4f}, "
            f"loss_objectness={losses['loss_objectness']:.4f}, "
            f"loss_rpn_box_reg={losses['loss_rpn_box_reg']:.4f}"
        )

    print("\n冒烟训练完成")
    print(f"训练日志：{LOG_PATH}")
    print(f"配置文件：{CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
