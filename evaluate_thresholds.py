from __future__ import annotations

import csv
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from infer_trained import get_model_instance_segmentation

IMAGE_DIR = PROJECT_DIR / "data" / "PennFudanPed" / "PNGImages"
MASK_DIR = PROJECT_DIR / "data" / "PennFudanPed" / "PedMasks"
WEIGHT_PATH = PROJECT_DIR / "weights" / "mask_rcnn_pennfudan.pth"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "threshold_evaluation"

THRESHOLDS = [0.30, 0.50, 0.70]
MASK_THRESHOLD = 0.50
NUM_CLASSES = 2
SAMPLE_COUNT = 10


@dataclass
class ReportRow:
    """保存单张图片在某个阈值下的统计结果。"""

    image_name: str
    threshold: float
    ground_truth_count: int | str
    predicted_count: int | str
    absolute_count_error: int | str
    highest_score: float | str
    inference_time_ms: float | str
    status: str = "ok"
    error_message: str = ""


def get_device() -> torch.device:
    """自动选择 CUDA；不可用时回退到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device) -> torch.nn.Module:
    """建立训练时一致的模型结构，并加载训练后的权重。"""

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"找不到权重文件：{WEIGHT_PATH}")

    model = get_model_instance_segmentation(NUM_CLASSES)
    state_dict = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def select_sample_images() -> list[Path]:
    """按文件名排序，选择前 10 张 Penn-Fudan 图片。"""

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"找不到图片目录：{IMAGE_DIR}")

    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    if len(image_paths) < SAMPLE_COUNT:
        raise RuntimeError(f"图片数量不足 {SAMPLE_COUNT} 张：{IMAGE_DIR}")

    return image_paths[:SAMPLE_COUNT]


def mask_path_for(image_path: Path) -> Path:
    """根据图片文件名找到对应的真实实例掩膜。"""

    return MASK_DIR / f"{image_path.stem}_mask.png"


def count_ground_truth_instances(mask_path: Path) -> int:
    """统计真实掩膜中非零实例编号的数量。"""

    if not mask_path.exists():
        raise FileNotFoundError(f"找不到真实掩膜：{mask_path}")

    mask_image = Image.open(mask_path)
    mask_tensor = torch.tensor(list(mask_image.tobytes()), dtype=torch.uint8)
    instance_ids = torch.unique(mask_tensor)
    return int((instance_ids != 0).sum().item())


def run_inference(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    """对单张图片执行一次模型推理，并返回耗时。"""

    image_tensor = to_tensor(image).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time_ms = (time.perf_counter() - start_time) * 1000
    return prediction, inference_time_ms


def filter_prediction(
    prediction: dict[str, torch.Tensor],
    threshold: float,
) -> dict[str, torch.Tensor]:
    """按类别和置信度阈值筛选行人预测结果。"""

    keep = (prediction["labels"] == 1) & (prediction["scores"] >= threshold)
    return {
        "boxes": prediction["boxes"][keep].detach().cpu(),
        "scores": prediction["scores"][keep].detach().cpu(),
        "masks": (prediction["masks"][keep, 0].detach().cpu() >= MASK_THRESHOLD),
    }


def draw_count_banner(
    image: Image.Image,
    ground_truth_count: int,
    predicted_count: int,
    threshold: float,
) -> Image.Image:
    """在可视化图片上写入真实人数和预测人数。"""

    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    text = (
        f"threshold {threshold:.2f} | "
        f"ground truth: {ground_truth_count} | predicted: {predicted_count}"
    )
    bbox = draw.textbbox((0, 0), text)
    padding = 8
    panel = (
        8,
        8,
        8 + (bbox[2] - bbox[0]) + padding * 2,
        8 + (bbox[3] - bbox[1]) + padding * 2,
    )
    draw.rectangle(panel, fill=(0, 0, 0))
    draw.text((8 + padding, 8 + padding), text, fill=(255, 255, 255))
    return result


def save_visualization(
    image: Image.Image,
    filtered: dict[str, torch.Tensor],
    output_path: Path,
    ground_truth_count: int,
    threshold: float,
) -> None:
    """保存包含边界框、实例掩膜、类别和置信度的可视化结果。"""

    result_tensor = pil_to_tensor(image.convert("RGB"))
    masks = filtered["masks"]
    boxes = filtered["boxes"]
    scores = filtered["scores"]

    if len(masks) > 0:
        result_tensor = draw_segmentation_masks(
            result_tensor,
            masks=masks,
            alpha=0.45,
        )

    if len(boxes) > 0:
        labels = [f"person {score:.2f}" for score in scores.tolist()]
        result_tensor = draw_bounding_boxes(
            result_tensor,
            boxes=boxes,
            labels=labels,
            width=3,
        )

    result_image = to_pil_image(result_tensor)
    result_image = draw_count_banner(
        result_image,
        ground_truth_count=ground_truth_count,
        predicted_count=len(boxes),
        threshold=threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(output_path)


def build_ok_row(
    image_name: str,
    threshold: float,
    ground_truth_count: int,
    filtered: dict[str, torch.Tensor],
    inference_time_ms: float,
) -> ReportRow:
    """生成成功处理图片后的 CSV 行。"""

    predicted_count = int(len(filtered["boxes"]))
    highest_score: float | str = ""
    if len(filtered["scores"]) > 0:
        highest_score = float(filtered["scores"].max().item())

    return ReportRow(
        image_name=image_name,
        threshold=threshold,
        ground_truth_count=ground_truth_count,
        predicted_count=predicted_count,
        absolute_count_error=abs(ground_truth_count - predicted_count),
        highest_score=highest_score,
        inference_time_ms=inference_time_ms,
    )


def build_failed_rows(image_name: str, error: Exception) -> list[ReportRow]:
    """生成失败图片的记录，避免单张图片中断整个测试。"""

    message = f"{type(error).__name__}: {error}"
    return [
        ReportRow(
            image_name=image_name,
            threshold=threshold,
            ground_truth_count="",
            predicted_count="",
            absolute_count_error="",
            highest_score="",
            inference_time_ms="",
            status="failed",
            error_message=message,
        )
        for threshold in THRESHOLDS
    ]


def write_csv(rows: list[ReportRow], output_path: Path) -> None:
    """把所有阈值下的逐图统计写入 CSV。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "threshold",
        "ground_truth_count",
        "predicted_count",
        "absolute_count_error",
        "highest_score",
        "inference_time_ms",
        "status",
        "error_message",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize_threshold(
    rows: list[ReportRow],
    threshold: float,
) -> dict[str, Any]:
    """汇总某个置信度阈值下的整体表现。"""

    threshold_rows = [
        row
        for row in rows
        if row.threshold == threshold and row.status == "ok"
    ]
    image_count = len(threshold_rows)
    predicted_total = sum(int(row.predicted_count) for row in threshold_rows)
    gt_total = sum(int(row.ground_truth_count) for row in threshold_rows)
    error_total = sum(int(row.absolute_count_error) for row in threshold_rows)
    time_total = sum(float(row.inference_time_ms) for row in threshold_rows)
    exact_count = sum(
        1
        for row in threshold_rows
        if int(row.absolute_count_error) == 0
    )

    return {
        "threshold": threshold,
        "image_count": image_count,
        "predicted_total": predicted_total,
        "ground_truth_total": gt_total,
        "mean_absolute_error": error_total / image_count if image_count else 0.0,
        "exact_count": exact_count,
        "mean_inference_time_ms": time_total / image_count if image_count else 0.0,
    }


def choose_best_threshold(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """按平均绝对数量误差选择本次数量预测最好的阈值。"""

    valid_summaries = [
        summary for summary in summaries if summary["image_count"] > 0
    ]
    if not valid_summaries:
        return None

    return min(
        valid_summaries,
        key=lambda item: (
            item["mean_absolute_error"],
            -item["exact_count"],
            item["threshold"],
        ),
    )


def write_summary(rows: list[ReportRow], output_path: Path) -> None:
    """生成 Markdown 格式的阈值汇总报告。"""

    summaries = [
        summarize_threshold(rows, threshold)
        for threshold in THRESHOLDS
    ]
    best = choose_best_threshold(summaries)
    failed_rows = [row for row in rows if row.status == "failed"]
    failed_images = sorted({row.image_name for row in failed_rows})

    lines = [
        "# 阈值评估汇总",
        "",
        "| 置信度阈值 | 测试图片数 | 预测目标总数 | 真实目标总数 | 平均绝对数量误差 | 完全预测正确图片数 | 平均推理时间(ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        lines.append(
            "| "
            f"{summary['threshold']:.2f} | "
            f"{summary['image_count']} | "
            f"{summary['predicted_total']} | "
            f"{summary['ground_truth_total']} | "
            f"{summary['mean_absolute_error']:.3f} | "
            f"{summary['exact_count']} | "
            f"{summary['mean_inference_time_ms']:.2f} |"
        )

    lines.extend(["", "## 最佳阈值", ""])
    if best is None:
        lines.append("没有成功处理的图片，无法选择最佳阈值。")
    else:
        lines.append(
            f"本次 10 张图片上，数量预测最好的是阈值 {best['threshold']:.2f}，"
            f"平均绝对数量误差为 {best['mean_absolute_error']:.3f}，"
            f"完全预测正确 {best['exact_count']} 张。"
        )

    if failed_images:
        lines.extend(["", "## 失败图片", ""])
        for image_name in failed_images:
            error_message = next(
                row.error_message
                for row in failed_rows
                if row.image_name == image_name
            )
            lines.append(f"- {image_name}: {error_message}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def evaluate_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> list[ReportRow]:
    """处理单张图片，并为三个阈值分别保存结果。"""

    print(f"处理图片：{image_path.name}")
    mask_path = mask_path_for(image_path)
    ground_truth_count = count_ground_truth_instances(mask_path)
    image = Image.open(image_path).convert("RGB")
    prediction, inference_time_ms = run_inference(model, image, device)

    rows: list[ReportRow] = []
    for threshold in THRESHOLDS:
        filtered = filter_prediction(prediction, threshold)
        output_path = (
            OUTPUT_DIR
            / f"threshold_{threshold:.2f}"
            / f"{image_path.stem}_threshold_{threshold:.2f}.png"
        )
        save_visualization(
            image=image,
            filtered=filtered,
            output_path=output_path,
            ground_truth_count=ground_truth_count,
            threshold=threshold,
        )
        rows.append(
            build_ok_row(
                image_name=image_path.name,
                threshold=threshold,
                ground_truth_count=ground_truth_count,
                filtered=filtered,
                inference_time_ms=inference_time_ms,
            )
        )

    return rows


def main() -> None:
    """主流程：加载模型、批量推理、保存图片和统计报告。"""

    device = get_device()
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"CUDA 设备：{torch.cuda.get_device_name(0)}")

    for threshold in THRESHOLDS:
        (OUTPUT_DIR / f"threshold_{threshold:.2f}").mkdir(
            parents=True,
            exist_ok=True,
        )

    image_paths = select_sample_images()
    model = load_model(device)

    rows: list[ReportRow] = []
    for image_path in image_paths:
        try:
            rows.extend(evaluate_image(image_path, model, device))
        except Exception as error:
            print(f"处理失败：{image_path.name} -> {error}")
            traceback.print_exc()
            rows.extend(build_failed_rows(image_path.name, error))

    write_csv(rows, OUTPUT_DIR / "threshold_report.csv")
    write_summary(rows, OUTPUT_DIR / "summary.md")

    ok_images = {
        row.image_name
        for row in rows
        if row.status == "ok"
    }
    print("\n测试完成")
    print(f"成功处理图片数：{len(ok_images)}")
    print(f"CSV 报告：{OUTPUT_DIR / 'threshold_report.csv'}")
    print(f"Markdown 汇总：{OUTPUT_DIR / 'summary.md'}")
    print(f"可视化目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
