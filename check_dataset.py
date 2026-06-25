from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_PATH = Path(
    "data/PennFudanPed/PNGImages/FudanPed00001.png"
)

MASK_PATH = Path(
    "data/PennFudanPed/PedMasks/FudanPed00001_mask.png"
)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"找不到图片：{IMAGE_PATH.resolve()}"
        )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"找不到mask：{MASK_PATH.resolve()}"
        )

    image = Image.open(IMAGE_PATH).convert("RGB")
    mask = Image.open(MASK_PATH)

    print("原图尺寸：", image.size)
    print("mask尺寸：", mask.size)
    print("原图模式：", image.mode)
    print("mask模式：", mask.mode)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Instance Mask")
    plt.imshow(mask)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()