from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from custom_dataset import CustomObjectDataset


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "data" / "custom_object"
PREVIEW_DIR = DATASET_DIR / "annotations_preview"
REPORT_PATH = DATASET_DIR / "annotation_report.csv"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_IMAGES = 10

COLORS = [
    (255, 80, 80),
    (80, 180, 255),
    (80, 220, 120),
    (255, 190, 70),
    (190, 100, 255),
    (255, 120, 200),
]


@dataclass
class ReportRow:
    """记录一张图片的标注检查结果。"""

    image_name: str
    image_width: int | str
    image_height: int | str
    instance_count: int | str
    status: str
    error_message: str = ""


def read_split_names(split_name: str) -> set[str]:
    """读取某个 split 文件中的样本名，用于检查重复。"""

    split_path = DATASET_DIR / "splits" / f"{split_name}.txt"
    if not split_path.exists():
        return set()

    return {
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def check_split_overlaps() -> list[str]:
    """检查 train、val、test 中是否存在重复样本。"""

    split_names = {
        split: read_split_names(split)
        for split in ["train", "val", "test"]
    }
    messages: list[str] = []
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        overlap = split_names[left] & split_names[right]
        if overlap:
            messages.append(
                f"{left} 和 {right} 存在重复样本：{sorted(overlap)}"
            )
    return messages


def blend_mask(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    """把一个二值 mask 以半透明颜色叠加到图片上。"""

    overlay = Image.new("RGB", image.size, color)
    mask_image = to_pil_image((mask > 0).to(torch.uint8) * 255)
    return Image.composite(
        Image.blend(image, overlay, alpha),
        image,
        mask_image,
    )


def draw_preview(
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    output_path: Path,
    sample_name: str,
) -> tuple[int, int, int]:
    """绘制单张图片的 mask、边界框、实例编号和实例总数。"""

    result = to_pil_image(image).convert("RGB")
    draw = ImageDraw.Draw(result)
    boxes = target["boxes"].cpu()
    masks = target["masks"].cpu()
    instance_count = int(masks.shape[0])

    for index, mask in enumerate(masks):
        color = COLORS[index % len(COLORS)]
        result = blend_mask(result, mask, color)
        draw = ImageDraw.Draw(result)
        x1, y1, x2, y2 = boxes[index].tolist()
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 3, y1 + 3), f"id {index + 1}", fill=(255, 255, 255))

    draw.rectangle((8, 8, 8 + 360, 44), fill=(0, 0, 0))
    draw.text(
        (18, 18),
        f"{sample_name} | instances: {instance_count}",
        fill=(255, 255, 255),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    width, height = result.size
    return width, height, instance_count


def write_report(rows: list[ReportRow]) -> None:
    """写出 annotation_report.csv。"""

    fieldnames = [
        "image_name",
        "image_width",
        "image_height",
        "instance_count",
        "status",
        "error_message",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def check_dataset(split: str, max_images: int) -> list[ReportRow]:
    """逐张检查数据集标注，单张失败时继续后面的样本。"""

    rows: list[ReportRow] = []
    dataset = CustomObjectDataset(DATASET_DIR, split=split)
    count = min(len(dataset), max_images)

    for index in range(count):
        sample = dataset.samples[index]
        try:
            image, target = dataset[index]
            output_path = PREVIEW_DIR / f"{split}_{sample.name}_preview.png"
            width, height, instance_count = draw_preview(
                image=image,
                target=target,
                output_path=output_path,
                sample_name=sample.name,
            )
            rows.append(
                ReportRow(
                    image_name=sample.image_path.name,
                    image_width=width,
                    image_height=height,
                    instance_count=instance_count,
                    status="ok",
                )
            )
            print(f"检查完成：{sample.image_path.name}，实例数 {instance_count}")
        except Exception as error:
            rows.append(
                ReportRow(
                    image_name=sample.image_path.name,
                    image_width="",
                    image_height="",
                    instance_count="",
                    status="failed",
                    error_message=f"{type(error).__name__}: {error}",
                )
            )
            print(f"检查失败：{sample.image_path.name} -> {error}")

    return rows


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="检查自制实例分割数据集标注并生成预览图。"
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "val", "test"],
        help="要检查的数据划分，默认 train。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help="最多检查多少张图片，默认 10。",
    )
    return parser.parse_args()


def main() -> int:
    """主流程：检查重复 split，生成预览图和 CSV 报告。"""

    args = parse_args()
    overlap_messages = check_split_overlaps()
    if overlap_messages:
        print("发现 split 重复：")
        for message in overlap_messages:
            print(f"- {message}")
    else:
        print("train/val/test 没有发现重复样本。")

    rows = check_dataset(split=args.split, max_images=args.max_images)
    write_report(rows)

    ok_count = sum(1 for row in rows if row.status == "ok")
    failed_count = sum(1 for row in rows if row.status == "failed")
    print("\n检查完成")
    print(f"成功：{ok_count}")
    print(f"失败：{failed_count}")
    print(f"预览图目录：{PREVIEW_DIR}")
    print(f"CSV 报告：{REPORT_PATH}")
    return 0 if failed_count == 0 and not overlap_messages else 1


if __name__ == "__main__":
    raise SystemExit(main())
