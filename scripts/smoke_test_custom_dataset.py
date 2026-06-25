from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from custom_dataset import CustomObjectDataset

DATASET_DIR = PROJECT_DIR / "data" / "custom_object"


def main() -> int:
    """读取一张自制数据样本，并打印 Mask R-CNN 需要的字段形状。"""

    dataset = CustomObjectDataset(DATASET_DIR, split="train")
    print(f"dataset length: {len(dataset)}")

    image, target = dataset[0]
    print(f"image shape: {tuple(image.shape)}")
    print(f"boxes shape: {tuple(target['boxes'].shape)}")
    print(f"labels: {target['labels'].tolist()}")
    print(f"masks shape: {tuple(target['masks'].shape)}")
    print(f"instance count: {int(target['labels'].numel())}")
    print(f"image_id: {target['image_id'].item()}")
    print(f"area shape: {tuple(target['area'].shape)}")
    print(f"iscrowd shape: {tuple(target['iscrowd'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
