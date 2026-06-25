from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data" / "PennFudanPed"
IMAGE_DIR = DATA_DIR / "PNGImages"
MASK_DIR = DATA_DIR / "PedMasks"
WEIGHT_PATH = PROJECT_DIR / "weights" / "mask_rcnn_pennfudan.pth"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "verify_project_result.png"
TEST_IMAGE_PATH = IMAGE_DIR / "FudanPed00001.png"
SCORE_THRESHOLD = 0.70
MASK_THRESHOLD = 0.50
NUM_CLASSES = 2


def check(condition: bool, message: str) -> None:
    """检查一个条件，不满足时抛出清晰错误。"""

    if not condition:
        raise RuntimeError(message)
    print(f"[OK] {message}")


def import_core_modules() -> None:
    """检查核心 Python 模块是否可以导入。"""

    modules = [
        "torch",
        "torchvision",
        "PIL",
        "matplotlib",
        "pycocotools",
        "infer_trained",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
        print(f"[OK] 模块可导入：{module_name}")


def get_device() -> torch.device:
    """优先使用 CUDA，不可用时回退到 CPU。"""

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"[OK] CUDA 可用：{torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("[WARN] CUDA 不可用，使用 CPU 验证。")
    return torch.device("cpu")


def check_files() -> None:
    """检查数据目录、测试图片和权重是否存在。"""

    check(DATA_DIR.exists(), f"数据目录存在：{DATA_DIR}")
    check(IMAGE_DIR.exists(), f"图片目录存在：{IMAGE_DIR}")
    check(MASK_DIR.exists(), f"mask 目录存在：{MASK_DIR}")
    check(TEST_IMAGE_PATH.exists(), f"测试图片存在：{TEST_IMAGE_PATH}")
    check(WEIGHT_PATH.exists(), f"权重文件存在：{WEIGHT_PATH}")
    check(WEIGHT_PATH.stat().st_size > 0, "权重文件非空")


def load_model(device: torch.device) -> torch.nn.Module:
    """建立模型结构并加载本地训练权重。"""

    from infer_trained import get_model_instance_segmentation

    model = get_model_instance_segmentation(NUM_CLASSES)
    state_dict = torch.load(
        WEIGHT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[OK] 模型权重加载成功")
    return model


def run_single_image_inference(
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """对一张图片做推理并保存可视化结果。"""

    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_tensor = to_tensor(image).to(device)

    start_time = time.perf_counter()
    with torch.inference_mode():
        prediction = model([image_tensor])[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"[OK] 输入图片张量形状：{tuple(image_tensor.shape)}")
    print(f"[OK] boxes 形状：{tuple(prediction['boxes'].shape)}")
    print(f"[OK] labels 形状：{tuple(prediction['labels'].shape)}")
    print(f"[OK] scores 形状：{tuple(prediction['scores'].shape)}")
    print(f"[OK] masks 形状：{tuple(prediction['masks'].shape)}")
    print(f"[OK] 推理耗时：{elapsed_ms:.2f} ms")

    keep = prediction["scores"] >= SCORE_THRESHOLD
    boxes = prediction["boxes"][keep].detach().cpu()
    scores = prediction["scores"][keep].detach().cpu()
    masks = prediction["masks"][keep, 0].detach().cpu() >= MASK_THRESHOLD

    result = pil_to_tensor(image)
    if len(masks) > 0:
        result = draw_segmentation_masks(result, masks=masks, alpha=0.45)
    if len(boxes) > 0:
        labels = [f"person {score:.2f}" for score in scores.tolist()]
        result = draw_bounding_boxes(result, boxes=boxes, labels=labels, width=3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(result).save(OUTPUT_PATH)
    check(OUTPUT_PATH.exists(), f"输出图片已生成：{OUTPUT_PATH}")
    check(OUTPUT_PATH.stat().st_size > 0, "输出图片非空")


def main() -> int:
    """完整验证项目是否能在当前机器恢复运行。"""

    print("开始验证 Mask R-CNN 项目……")
    import_core_modules()
    device = get_device()
    check_files()
    model = load_model(device)
    run_single_image_inference(model, device)
    print("\n项目验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
