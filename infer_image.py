from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import (
    pil_to_tensor,
    to_pil_image,
    to_tensor,
)
from torchvision.utils import (
    draw_bounding_boxes,
    draw_segmentation_masks,
)

from infer_trained import get_model_instance_segmentation


PROJECT_DIR = Path(__file__).resolve().parent

# 先使用 Penn-Fudan 数据集中的一张图片测试
IMAGE_PATH = (
    PROJECT_DIR
    / "data"
    / "PennFudanPed"
    / "PNGImages"
    / "FudanPed00001.png"
)

WEIGHT_PATH = (
    PROJECT_DIR
    / "weights"
    / "mask_rcnn_pennfudan.pth"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "trained_result.png"
)

SCORE_THRESHOLD = 0.70
MASK_THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"找不到测试图片：{IMAGE_PATH}"
        )

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型权重：{WEIGHT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Penn-Fudan：
    # 0 = 背景
    # 1 = 行人
    num_classes = 2

    print("正在建立模型……")

    model = get_model_instance_segmentation(
        num_classes
    )

    print("正在加载训练权重……")

    state_dict = torch.load(
        WEIGHT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 读取原图
    image = Image.open(IMAGE_PATH).convert("RGB")

    # 模型输入：[3, H, W]，数值范围0到1
    image_tensor = to_tensor(image).to(DEVICE)

    print("输入图片形状：", image_tensor.shape)

    # 执行推理
    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    print("\n原始预测结果：")
    print("boxes：", prediction["boxes"].shape)
    print("labels：", prediction["labels"].shape)
    print("scores：", prediction["scores"].shape)
    print("masks：", prediction["masks"].shape)

    # 根据置信度筛选目标
    keep = prediction["scores"] >= SCORE_THRESHOLD

    boxes = prediction["boxes"][keep].cpu()
    scores = prediction["scores"][keep].cpu()

    # [N, 1, H, W] → [N, H, W]
    masks = (
        prediction["masks"][keep, 0].cpu()
        >= MASK_THRESHOLD
    )

    print(
        f"\n置信度不低于 {SCORE_THRESHOLD} "
        f"的行人数：{len(boxes)}"
    )

    labels = [
        f"person {score:.2f}"
        for score in scores.tolist()
    ]

    # 原图转换为uint8张量，用于绘图
    result = pil_to_tensor(image)

    if len(masks) > 0:
        result = draw_segmentation_masks(
            result,
            masks=masks,
            alpha=0.45,
        )

    if len(boxes) > 0:
        result = draw_bounding_boxes(
            result,
            boxes=boxes,
            labels=labels,
            width=3,
        )

    result_image = to_pil_image(result)
    result_image.save(OUTPUT_PATH)
    print("\n推理完成")
    print("使用设备：", DEVICE)
    print("结果保存在：", OUTPUT_PATH)


if __name__ == "__main__":
    main()