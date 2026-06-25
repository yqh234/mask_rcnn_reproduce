# 项目迁移说明

本文档基于当前已经提交到 GitHub 的项目状态编写。目标是在另一台 Windows 电脑上恢复当前 Mask R-CNN 项目，让它能加载本地权重、读取数据、完成推理，并继续使用自制数据集检查流程。

GitHub 仓库：

```text
https://github.com/yqh234/mask_rcnn_reproduce
```

## 1. 当前仓库已经包含什么

当前 GitHub 仓库已经包含：

```text
核心源码：
- train_pennfudan.py
- infer_trained.py
- infer_image.py
- evaluate_thresholds.py
- custom_dataset.py
- check_custom_dataset.py
- engine.py
- utils.py
- coco_utils.py
- coco_eval.py
- transforms.py

迁移和验证：
- NEW_COMPUTER_SETUP.md
- MIGRATION.md
- setup_environment.ps1
- environment_report.py
- verify_project.py
- requirements-base.txt
- requirements-lock.txt

文档：
- AGENTS.md
- docs/current_baseline.md
- docs/pennfudan_baseline.md

数据：
- data/PennFudanPed/
- data/custom_object/
```

其中 `data/custom_object/` 里包含 5 组演示图片、伪 mask、划分文件和标注预览图。它们适合用来跑通自制数据集读取和检查流程，但不是人工精修的正式训练数据。

## 2. 仍然需要人工单独备份什么

最重要的是训练权重：

```text
weights/mask_rcnn_pennfudan.pth
```

这个文件没有进入 Git，因为它大约 176 MB，超过 GitHub 普通单文件 100 MB 限制。

如果你希望保留旧电脑上的运行结果，也要单独备份：

```text
outputs/
experiments/
```

这些目录不是新电脑恢复运行的必需项，但里面可能有你已经生成的结果图、报告和实验记录。

如果以后自制数据集变大，也建议单独备份：

```text
data/custom_object/images/
data/custom_object/masks/
```

当前这 5 组演示样本已经提交进 Git，不需要额外备份。

## 3. 旧电脑迁移前检查

在旧电脑项目根目录运行：

```powershell
git status
& ".\.venv\Scripts\python.exe" environment_report.py
& ".\.venv\Scripts\python.exe" verify_project.py
```

如果旧电脑没有项目内 `.venv`，可以用当前实际可用的 Python 解释器运行：

```powershell
& "你的Python路径\python.exe" environment_report.py
& "你的Python路径\python.exe" verify_project.py
```

确认 `verify_project.py` 通过后，再备份：

```text
weights/mask_rcnn_pennfudan.pth
```

## 4. 新电脑恢复步骤

### 4.1 克隆项目

```powershell
git clone https://github.com/yqh234/mask_rcnn_reproduce.git
cd mask_rcnn_reproduce
```

### 4.2 创建虚拟环境并安装基础依赖

```powershell
.\setup_environment.ps1
```

这个脚本会创建：

```text
.venv/
```

并安装：

```text
requirements-base.txt
```

如果新电脑上 `python` 不是 Python 3.12，可以指定解释器：

```powershell
.\setup_environment.ps1 -Python "C:\Path\To\Python312\python.exe"
```

### 4.3 放回权重

把旧电脑备份的权重复制到：

```text
weights/mask_rcnn_pennfudan.pth
```

如果 `weights/` 目录不存在，就手动创建：

```powershell
mkdir weights
```

最终应是：

```text
mask_rcnn_reproduce/
└── weights/
    └── mask_rcnn_pennfudan.pth
```

### 4.4 生成环境报告

```powershell
& ".\.venv\Scripts\python.exe" environment_report.py
```

它会生成：

```text
environment_report.json
```

报告内容包括：

- Python 版本
- torch 版本
- torchvision 版本
- torch CUDA 版本
- CUDA 是否可用
- GPU 名称
- 操作系统

### 4.5 验证项目

```powershell
& ".\.venv\Scripts\python.exe" verify_project.py
```

它会检查：

- 核心 Python 模块是否能导入。
- CUDA 是否可用。
- Penn-Fudan 数据目录是否存在。
- 权重是否存在且非空。
- 模型是否能加载。
- 至少一张图片是否能完成推理。
- 输出图片是否真实生成。

成功时会看到：

```text
项目验证通过。
```

并生成：

```text
outputs/verify_project_result.png
```

## 5. 自制数据集相关验证

当前仓库已经包含一个小型演示自制数据集：

```text
data/custom_object/
├── images/
├── masks/
├── splits/
├── annotations_preview/
├── README.md
├── annotation_report.csv
└── source_manifest.csv
```

验证自制数据集读取：

```powershell
& ".\.venv\Scripts\python.exe" scripts/smoke_test_custom_dataset.py
```

预期会打印类似：

```text
dataset length: 3
image shape: (3, H, W)
boxes shape: (N, 4)
labels: [...]
masks shape: (N, H, W)
```

重新检查标注并生成预览图：

```powershell
& ".\.venv\Scripts\python.exe" check_custom_dataset.py
```

输出位置：

```text
data/custom_object/annotations_preview/
data/custom_object/annotation_report.csv
```

注意：当前自制数据集里的 mask 是用 COCO 预训练模型自动生成的伪标注，主要用于跑通流程。正式训练前，建议你用人工检查和修正后的数据替换它们。

## 6. 哪些文件应该进入 Git

应该进入 Git：

```text
*.py
*.md
*.ps1
requirements-base.txt
requirements-lock.txt
.gitignore
AGENTS.md
docs/
scripts/
data/PennFudanPed/
data/custom_object/README.md
data/custom_object/splits/
小规模演示图片和 mask
```

当前仓库已经提交了小规模演示图片和 mask。

## 7. 哪些文件不应该进入 Git

不建议进入 Git：

```text
.venv/
__pycache__/
outputs/
experiments/
environment_report.json
weights/*.pth
weights/*.pt
weights/*.ckpt
*.zip
```

原因：

- `.venv/` 和缓存可以在新电脑重新生成。
- `outputs/` 是运行产物。
- `experiments/` 可能包含较大的实验记录。
- `environment_report.json` 是当前机器生成的本地报告。
- 权重文件通常很大，应单独备份。

## 8. 常见错误和恢复方法

### 找不到权重

错误表现：

```text
权重文件存在
```

这条检查没有通过，或者提示找不到：

```text
weights/mask_rcnn_pennfudan.pth
```

恢复方法：

把旧电脑备份的权重复制回 `weights/`。

### CUDA 不可用

如果看到：

```text
CUDA 不可用，使用 CPU 验证。
```

项目仍可运行，只是推理会变慢。需要 GPU 时检查：

- NVIDIA 驱动是否安装。
- PyTorch 是否安装了 CUDA 版本。
- `torch.cuda.is_available()` 是否为 `True`。

### PowerShell 不允许运行脚本

如果 `setup_environment.ps1` 被执行策略拦截，在当前 PowerShell 窗口临时允许：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_environment.ps1
```

### 缺少 pycocotools

```powershell
& ".\.venv\Scripts\python.exe" -m pip install pycocotools
```

### 依赖版本冲突

优先使用基础依赖：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-base.txt
```

如果想尽量复刻旧电脑环境，再使用：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` 是旧电脑完整冻结环境，可能和新电脑的 Python、CUDA 或显卡驱动不完全匹配。

### 找不到 Penn-Fudan 数据集

检查是否存在：

```text
data/PennFudanPed/
```

里面应包含：

```text
PNGImages/
PedMasks/
Annotation/
```

当前 GitHub 仓库已经包含 Penn-Fudan 解压后的数据。如果 clone 后缺失，先检查 Git 是否完整拉取。

## 9. 推荐验证顺序

新电脑上完整顺序：

```powershell
git clone https://github.com/yqh234/mask_rcnn_reproduce.git
cd mask_rcnn_reproduce
.\setup_environment.ps1
```

复制权重：

```text
weights/mask_rcnn_pennfudan.pth
```

验证环境和推理：

```powershell
& ".\.venv\Scripts\python.exe" environment_report.py
& ".\.venv\Scripts\python.exe" verify_project.py
```

验证自制数据集：

```powershell
& ".\.venv\Scripts\python.exe" scripts/smoke_test_custom_dataset.py
& ".\.venv\Scripts\python.exe" check_custom_dataset.py
```

如果这些命令都通过，说明项目已经在新电脑恢复完成。

## 10. 当前迁移原则

- 不改变模型结构。
- 不改变类别数。
- 不覆盖已有权重。
- 所有路径都从项目根目录或脚本所在位置推导。
- Git 保存代码、文档、小数据和检查脚本。
- 大权重和运行产物单独备份。
