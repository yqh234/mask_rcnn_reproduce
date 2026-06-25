from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "data" / "custom_object"
IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"
SPLIT_DIR = DATASET_DIR / "splits"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CustomSample:
    """记录一张原图和它对应的实例 mask。"""

    name: str
    image_path: Path
    mask_path: Path


def ensure_directories() -> None:
    """确保自制数据集的必要目录都存在。"""

    for directory in [IMAGE_DIR, MASK_DIR, SPLIT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def find_image_paths() -> list[Path]:
    """按文件名读取所有支持格式的原图。"""

    if not IMAGE_DIR.exists():
        return []

    return sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def find_mask_for_image(image_path: Path) -> Path | None:
    """根据原图 stem 查找对应的实例 mask。"""

    preferred_mask = MASK_DIR / f"{image_path.stem}_mask.png"
    if preferred_mask.exists():
        return preferred_mask

    fallback_mask = MASK_DIR / f"{image_path.stem}.png"
    if fallback_mask.exists():
        return fallback_mask

    return None


def collect_samples() -> tuple[list[CustomSample], list[str]]:
    """收集可用于划分的数据，并记录缺少 mask 的图片。"""

    samples: list[CustomSample] = []
    errors: list[str] = []

    for image_path in find_image_paths():
        mask_path = find_mask_for_image(image_path)
        if mask_path is None:
            errors.append(
                f"找不到对应 mask：{image_path.name}，"
                f"推荐命名为 {image_path.stem}_mask.png"
            )
            continue

        samples.append(
            CustomSample(
                name=image_path.stem,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    return samples, errors


def calculate_split_counts(total_count: int) -> tuple[int, int, int]:
    """按 70/20/10 计算 train、val、test 数量。"""

    if total_count < 3:
        raise ValueError(
            "自制数据集至少需要 3 张有配对 mask 的图片，"
            "才能划分 train、val、test。"
        )

    train_count = max(1, int(total_count * TRAIN_RATIO))
    val_count = max(1, int(total_count * VAL_RATIO))
    test_count = total_count - train_count - val_count

    if test_count < 1:
        test_count = 1
        train_count = total_count - val_count - test_count

    if train_count < 1:
        raise ValueError("训练集数量不足，请增加图片后再划分。")

    return train_count, val_count, test_count


def split_samples(
    samples: list[CustomSample],
) -> tuple[list[CustomSample], list[CustomSample], list[CustomSample]]:
    """使用固定随机种子划分 train、val、test。"""

    shuffled = samples[:]
    random.Random(RANDOM_SEED).shuffle(shuffled)

    train_count, val_count, _ = calculate_split_counts(len(shuffled))
    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count : train_count + val_count]
    test_samples = shuffled[train_count + val_count :]

    validate_no_overlap(train_samples, val_samples, test_samples)
    return train_samples, val_samples, test_samples


def validate_no_overlap(
    train_samples: list[CustomSample],
    val_samples: list[CustomSample],
    test_samples: list[CustomSample],
) -> None:
    """检查同一张图片不会进入多个集合。"""

    split_names = {
        "train": {sample.name for sample in train_samples},
        "val": {sample.name for sample in val_samples},
        "test": {sample.name for sample in test_samples},
    }

    train_val = split_names["train"] & split_names["val"]
    train_test = split_names["train"] & split_names["test"]
    val_test = split_names["val"] & split_names["test"]

    if train_val or train_test or val_test:
        raise RuntimeError(
            "数据划分出现重复："
            f"train/val={sorted(train_val)}, "
            f"train/test={sorted(train_test)}, "
            f"val/test={sorted(val_test)}"
        )


def write_split_file(split_name: str, samples: list[CustomSample]) -> Path:
    """把一个 split 的样本名写入 txt 文件。"""

    output_path = SPLIT_DIR / f"{split_name}.txt"
    lines = [sample.name for sample in samples]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def print_dataset_hint(errors: list[str]) -> None:
    """输出空数据或缺少 mask 时的友好提示。"""

    print("没有生成划分文件。")
    print(f"请把原图放入：{IMAGE_DIR}")
    print(f"请把实例 mask 放入：{MASK_DIR}")
    print("推荐命名示例：")
    print("  images/0001.jpg")
    print("  masks/0001_mask.png")

    if errors:
        print("\n发现的问题：")
        for error in errors:
            print(f"- {error}")


def main() -> int:
    """主流程：收集样本，划分数据集，并写出 split 文件。"""

    ensure_directories()
    samples, errors = collect_samples()

    if errors:
        print_dataset_hint(errors)
        return 1

    if not samples:
        print("当前没有找到可划分的自制数据。")
        print_dataset_hint(errors)
        return 1

    try:
        train_samples, val_samples, test_samples = split_samples(samples)
    except ValueError as error:
        print(f"数据不足：{error}")
        print_dataset_hint(errors)
        return 1

    output_paths = [
        write_split_file("train", train_samples),
        write_split_file("val", val_samples),
        write_split_file("test", test_samples),
    ]

    print("自制数据集划分完成")
    print(f"随机种子：{RANDOM_SEED}")
    print(f"样本总数：{len(samples)}")
    print(f"训练集 train：{len(train_samples)}")
    print(f"验证集 val：{len(val_samples)}")
    print(f"测试集 test：{len(test_samples)}")
    print("输出文件：")
    for output_path in output_paths:
        print(f"- {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
