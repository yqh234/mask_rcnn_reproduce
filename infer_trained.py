from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# 当前脚本所在目录
PROJECT_DIR = Path(__file__).resolve().parent

# 训练得到的权重
WEIGHT_PATH = (
    PROJECT_DIR
    / "weights"
    / "mask_rcnn_pennfudan.pth"
)

# 自动选择显卡或CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_model_instance_segmentation(
    num_classes: int,
) -> torch.nn.Module:
    """
    创建与训练时完全相同的 Mask R-CNN 模型结构。
    """

    # 不加载官方权重，因为后面会加载自己训练的权重
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    # 替换边界框分类头
    box_in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_in_features,
        num_classes,
    )

    # 替换掩膜预测头
    mask_in_features = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features,
        hidden_layer,
        num_classes,
    )

    return model


def main() -> None:
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(
            f"找不到权重文件：{WEIGHT_PATH}"
        )

    # Penn-Fudan：
    # 类别0 = 背景
    # 类别1 = 行人
    num_classes = 2

    print("正在建立模型结构……")

    model = get_model_instance_segmentation(
        num_classes=num_classes
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

    print("\n权重加载成功")
    print("使用设备：", DEVICE)
    print("权重路径：", WEIGHT_PATH)
    print("模型状态：", "推理模式" if not model.training else "训练模式")


if __name__ == "__main__":
    main()