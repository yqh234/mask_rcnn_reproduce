from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import to_tensor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CustomSample:
    """记录一张原图和它对应的实例 mask。"""

    name: str
    image_path: Path
    mask_path: Path


class CustomObjectDataset(Dataset):
    """TorchVision Mask R-CNN 可用的自制单类实例分割数据集。"""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: Callable | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split
        self.transforms = transforms
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.split_path = self.root / "splits" / f"{split}.txt"
        self.samples = self._load_samples()

    def __len__(self) -> int:
        """返回当前 split 中的样本数量。"""

        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """读取一张图片，生成 Mask R-CNN 需要的 target。"""

        sample = self.samples[index]
        image = self._read_image(sample.image_path)
        mask_image = self._read_mask(sample.mask_path)

        if image.size != mask_image.size:
            raise ValueError(
                f"原图和 mask 宽高不一致：{sample.name}，"
                f"image={image.size}, mask={mask_image.size}"
            )

        masks = self._build_instance_masks(mask_image, sample.name)
        boxes = masks_to_boxes(masks)
        self._validate_boxes(boxes, sample.name)

        labels = torch.ones((masks.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([index], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)

        image_tensor = to_tensor(image)
        target: dict[str, torch.Tensor] = {
            "boxes": boxes.to(torch.float32),
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area.to(torch.float32),
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

    def _load_samples(self) -> list[CustomSample]:
        """从 split 文件读取样本名，并检查图片和 mask 是否存在。"""

        self._validate_dataset_dirs()

        if not self.split_path.exists():
            raise FileNotFoundError(
                f"找不到 split 文件：{self.split_path}。"
                "请先运行 scripts/create_custom_splits.py。"
            )

        names = [
            line.strip()
            for line in self.split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not names:
            raise ValueError(f"split 文件为空：{self.split_path}")

        samples: list[CustomSample] = []
        for name in names:
            image_path = self._find_image_path(name)
            mask_path = self._find_mask_path(name)
            samples.append(
                CustomSample(
                    name=name,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

        return samples

    def _validate_dataset_dirs(self) -> None:
        """检查自制数据集关键目录是否存在。"""

        for directory in [self.root, self.image_dir, self.mask_dir]:
            if not directory.exists():
                raise FileNotFoundError(f"找不到目录：{directory}")

    def _find_image_path(self, name: str) -> Path:
        """根据 split 中的样本名查找原图。"""

        matches = [
            self.image_dir / f"{name}{suffix}"
            for suffix in IMAGE_SUFFIXES
            if (self.image_dir / f"{name}{suffix}").exists()
        ]
        if not matches:
            raise FileNotFoundError(
                f"找不到原图：{name}，目录：{self.image_dir}"
            )
        return sorted(matches)[0]

    def _find_mask_path(self, name: str) -> Path:
        """根据 split 中的样本名查找实例 mask。"""

        preferred_mask = self.mask_dir / f"{name}_mask.png"
        if preferred_mask.exists():
            return preferred_mask

        fallback_mask = self.mask_dir / f"{name}.png"
        if fallback_mask.exists():
            return fallback_mask

        raise FileNotFoundError(
            f"找不到 mask：{name}_mask.png，目录：{self.mask_dir}"
        )

    def _read_image(self, image_path: Path) -> Image.Image:
        """读取 RGB 原图，并处理无法读取的文件。"""

        try:
            return Image.open(image_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"无法读取原图：{image_path}") from error

    def _read_mask(self, mask_path: Path) -> Image.Image:
        """读取单通道实例 mask，并处理无法读取的文件。"""

        try:
            return Image.open(mask_path).convert("L")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"无法读取 mask：{mask_path}") from error

    def _build_instance_masks(
        self,
        mask_image: Image.Image,
        sample_name: str,
    ) -> torch.Tensor:
        """把单张实例编号 mask 转成 [N, H, W] 二值 mask。"""

        width, height = mask_image.size
        mask_tensor = torch.tensor(
            list(mask_image.tobytes()),
            dtype=torch.uint8,
        ).reshape(height, width)

        instance_ids = torch.unique(mask_tensor)
        instance_ids = instance_ids[instance_ids != 0]
        if instance_ids.numel() == 0:
            raise ValueError(f"mask 为空，没有非零实例：{sample_name}")

        masks = mask_tensor == instance_ids[:, None, None]
        areas = masks.flatten(1).sum(dim=1)
        valid = areas > 0
        if not bool(valid.all()):
            invalid_ids = instance_ids[~valid].tolist()
            raise ValueError(
                f"存在面积为 0 的实例：{sample_name}, ids={invalid_ids}"
            )

        return masks.to(torch.uint8)

    def _validate_boxes(self, boxes: torch.Tensor, sample_name: str) -> None:
        """检查由 mask 计算出的边界框是否有效。"""

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 0) & (heights > 0)
        if not bool(valid.all()):
            raise ValueError(f"存在零面积边界框：{sample_name}")
