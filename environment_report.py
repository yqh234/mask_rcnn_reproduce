from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch
import torchvision


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "environment_report.json"


def collect_environment() -> dict[str, str | bool]:
    """收集当前 Python、PyTorch、CUDA 和系统信息。"""

    cuda_available = torch.cuda.is_available()
    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "N/A",
        "operating_system": platform.platform(),
    }


def main() -> None:
    """打印环境信息，并保存为 JSON 文件。"""

    report = collect_environment()
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("环境报告：")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"\n已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
