from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "data" / "custom_object"
IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"
MANIFEST_PATH = DATASET_DIR / "source_manifest.csv"

TARGET_SAMPLE_COUNT = 12
MAX_DOWNLOAD_PER_QUERY = 12
MAX_IMAGE_SIZE = 640
SCORE_THRESHOLD = 0.65
MASK_THRESHOLD = 0.50
MIN_MASK_AREA = 800

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "mask-rcnn-reproduce-demo/1.0 (learning project)"

SEARCH_QUERIES = [
    "banana fruit photo",
    "apple fruit photo",
    "orange fruit photo",
    "coffee mug photo",
    "ceramic bowl photo",
    "water bottle photo",
    "soda can photo",
    "spoon photo",
]

ALLOWED_COCO_CLASSES = {
    "banana",
    "apple",
    "orange",
    "cup",
    "bowl",
    "bottle",
    "spoon",
}


@dataclass(frozen=True)
class CandidateImage:
    """记录从 Wikimedia Commons 找到的候选图片。"""

    title: str
    url: str
    original_url: str
    width: int
    height: int


@dataclass(frozen=True)
class SavedSample:
    """记录已保存到自制数据集的一组图片和 mask。"""

    image_name: str
    mask_name: str
    source_title: str
    source_url: str
    detected_labels: str
    scores: str


def get_device() -> torch.device:
    """优先使用 CUDA，无法使用时回退到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_directories() -> None:
    """确保输出目录存在。"""

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)


def request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """从 Wikimedia Commons API 请求 JSON 数据。"""

    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search_commons(query: str) -> list[CandidateImage]:
    """按关键词搜索 Commons 图片候选。"""

    payload = request_json(
        COMMONS_API_URL,
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": MAX_DOWNLOAD_PER_QUERY,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": MAX_IMAGE_SIZE,
        },
    )

    candidates: list[CandidateImage] = []
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        image_info = page.get("imageinfo", [{}])[0]
        mime = image_info.get("mime", "")
        if mime not in {"image/jpeg", "image/png"}:
            continue

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if width < 160 or height < 160:
            continue

        candidates.append(
            CandidateImage(
                title=str(page.get("title", "")),
                url=str(image_info.get("thumburl") or image_info.get("url", "")),
                original_url=str(image_info.get("url", "")),
                width=width,
                height=height,
            )
        )

    return candidates


def download_image(candidate: CandidateImage) -> Image.Image:
    """下载一张候选图片并转成 RGB。"""

    request = urllib.request.Request(
        candidate.url,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
    return image


def safe_stem(text: str) -> str:
    """把来源标题转换成适合作为文件名的短 stem。"""

    stem = text.replace("File:", "")
    stem = Path(stem).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem[:48] or "sample"


def load_coco_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """加载 COCO 预训练 Mask R-CNN，用于生成伪标注。"""

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    categories = list(weights.meta["categories"])
    return model, categories


def predict_instances(
    model: torch.nn.Module,
    categories: list[str],
    image: Image.Image,
    device: torch.device,
) -> tuple[Image.Image | None, list[str], list[float]]:
    """用 COCO 模型筛选可抓取物体，并合成实例编号 mask。"""

    image_tensor = to_tensor(image).to(device)
    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    masks = prediction["masks"].detach().cpu()

    instance_mask = Image.new("L", image.size, 0)
    instance_pixels = instance_mask.load()
    kept_labels: list[str] = []
    kept_scores: list[float] = []
    next_instance_id = 1

    for index, label_tensor in enumerate(labels):
        score = float(scores[index].item())
        label_name = categories[int(label_tensor.item())]
        if score < SCORE_THRESHOLD or label_name not in ALLOWED_COCO_CLASSES:
            continue

        binary_mask = masks[index, 0] >= MASK_THRESHOLD
        area = int(binary_mask.sum().item())
        if area < MIN_MASK_AREA:
            continue

        mask_image = Image.fromarray(
            (binary_mask.numpy().astype("uint8") * 255),
            mode="L",
        )
        mask_pixels = mask_image.load()
        width, height = image.size
        for y in range(height):
            for x in range(width):
                if mask_pixels[x, y] > 0 and instance_pixels[x, y] == 0:
                    instance_pixels[x, y] = next_instance_id

        kept_labels.append(label_name)
        kept_scores.append(score)
        next_instance_id += 1

        if next_instance_id > 250:
            break

    if not kept_labels:
        return None, [], []

    return instance_mask, kept_labels, kept_scores


def save_sample(
    image: Image.Image,
    mask: Image.Image,
    candidate: CandidateImage,
    labels: list[str],
    scores: list[float],
    index: int,
) -> SavedSample:
    """保存一组原图、实例 mask 和来源信息。"""

    stem = f"web_{index:03d}_{safe_stem(candidate.title)}"
    image_name = f"{stem}.jpg"
    mask_name = f"{stem}_mask.png"
    image_path = IMAGE_DIR / image_name
    mask_path = MASK_DIR / mask_name

    image.save(image_path, quality=95)
    mask.save(mask_path)

    return SavedSample(
        image_name=image_name,
        mask_name=mask_name,
        source_title=candidate.title,
        source_url=candidate.original_url or candidate.url,
        detected_labels=";".join(labels),
        scores=";".join(f"{score:.3f}" for score in scores),
    )


def write_manifest(samples: list[SavedSample]) -> None:
    """写出样本来源清单，方便以后追溯。"""

    fieldnames = [
        "image_name",
        "mask_name",
        "source_title",
        "source_url",
        "detected_labels",
        "scores",
        "annotation_type",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    **sample.__dict__,
                    "annotation_type": "pseudo_mask_from_coco_mask_rcnn",
                }
            )


def existing_image_count() -> int:
    """统计当前 images 目录中已有的真实图片数量。"""

    return len(
        [
            path
            for path in IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def read_existing_source_urls() -> set[str]:
    """读取已经保存过的来源 URL，避免重复下载。"""

    if not MANIFEST_PATH.exists():
        return set()

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return {
            row.get("source_url", "")
            for row in reader
            if row.get("source_url")
        }


def read_existing_manifest_rows() -> list[SavedSample]:
    """读取已有 manifest，支持继续补充样本。"""

    if not MANIFEST_PATH.exists():
        return []

    rows: list[SavedSample] = []
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                SavedSample(
                    image_name=row["image_name"],
                    mask_name=row["mask_name"],
                    source_title=row["source_title"],
                    source_url=row["source_url"],
                    detected_labels=row["detected_labels"],
                    scores=row["scores"],
                )
            )
    return rows


def main() -> int:
    """主流程：找公开图片，自动生成伪 mask，并保存成项目格式。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ensure_directories()

    current_image_count = existing_image_count()
    if current_image_count > 0 and not MANIFEST_PATH.exists():
        print("data/custom_object/images 中已经有图片。")
        print("为避免覆盖你的手工数据，本脚本不会继续写入。")
        return 1

    if current_image_count >= TARGET_SAMPLE_COUNT:
        print(f"当前已有 {current_image_count} 张图片，不需要继续补充。")
        return 0

    device = get_device()
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"CUDA 设备：{torch.cuda.get_device_name(0)}")

    print("正在加载 COCO 预训练 Mask R-CNN，用于生成伪标注……")
    model, categories = load_coco_model(device)

    saved_samples = read_existing_manifest_rows()
    seen_urls = read_existing_source_urls()

    for query in SEARCH_QUERIES:
        if len(saved_samples) >= TARGET_SAMPLE_COUNT:
            break

        print(f"\n搜索公开图片：{query}")
        try:
            candidates = search_commons(query)
        except Exception as error:
            print(f"搜索失败：{query} -> {error}")
            continue

        for candidate in candidates:
            if len(saved_samples) >= TARGET_SAMPLE_COUNT:
                break
            source_url = candidate.original_url or candidate.url
            if not candidate.url or source_url in seen_urls:
                continue
            seen_urls.add(source_url)

            try:
                print(f"下载并生成伪 mask：{candidate.title}")
                image = download_image(candidate)
                mask, labels, scores = predict_instances(
                    model=model,
                    categories=categories,
                    image=image,
                    device=device,
                )
                if mask is None:
                    print("  跳过：COCO 模型没有检测到合适的可抓取物体")
                    continue

                sample = save_sample(
                    image=image,
                    mask=mask,
                    candidate=candidate,
                    labels=labels,
                    scores=scores,
                    index=len(saved_samples) + 1,
                )
                saved_samples.append(sample)
                print(f"  保存：{sample.image_name} / {sample.mask_name}")
                time.sleep(0.8)
            except Exception as error:
                print(f"  跳过：处理失败 -> {error}")
                continue

    if not saved_samples:
        print("没有成功生成任何样本。")
        return 1

    write_manifest(saved_samples)
    print("\n样本生成完成")
    print(f"成功样本数：{len(saved_samples)}")
    print(f"图片目录：{IMAGE_DIR}")
    print(f"mask 目录：{MASK_DIR}")
    print(f"来源清单：{MANIFEST_PATH}")
    print("注意：这些 mask 是 COCO 预训练模型生成的伪标注，正式训练前建议人工检查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
