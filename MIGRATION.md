# 项目迁移说明

本文档说明如何把当前 Mask R-CNN 项目迁移到另一台电脑，并恢复到可以加载权重、读取数据、完成推理的状态。

## 1. 旧电脑如何备份

旧电脑上的项目目录：

```text
mask_rcnn_reproduce/
```

建议分两类备份：

1. 用 Git 备份代码、文档、轻量配置和小规模示例数据。
2. 用网盘、移动硬盘或压缩包单独备份大文件，例如训练权重和大量输出结果。

推荐先运行：

```powershell
git status
& ".\.venv\Scripts\python.exe" environment_report.py
& ".\.venv\Scripts\python.exe" verify_project.py
```

如果旧电脑没有项目内 `.venv`，也可以用当前可用的 Python 解释器运行上面两个脚本。

## 2. 哪些文件应该进入 Git

建议提交：

```text
*.py
*.md
*.ps1
requirements-base.txt
requirements-lock.txt
.gitignore
AGENTS.md
data/PennFudanPed/
data/custom_object/README.md
data/custom_object/splits/
docs/
scripts/
```

如果自制数据集很小，也可以提交：

```text
data/custom_object/images/
data/custom_object/masks/
data/custom_object/source_manifest.csv
```

## 3. 哪些文件不应该进入 Git

不建议提交：

```text
.venv/
__pycache__/
outputs/
experiments/
weights/*.pth
weights/*.pt
weights/*.ckpt
*.zip
```

原因：

- 虚拟环境和缓存可以在新电脑重新生成。
- `outputs/` 是运行结果，不是源码。
- 权重文件通常较大，例如 `mask_rcnn_pennfudan.pth` 超过 GitHub 普通单文件 100MB 限制。

## 4. 数据和权重如何单独备份

至少需要单独备份：

```text
weights/mask_rcnn_pennfudan.pth
```

如果数据没有进入 Git，也要单独备份：

```text
data/PennFudanPed/
data/custom_object/images/
data/custom_object/masks/
```

恢复时保持同样的目录结构：

```text
mask_rcnn_reproduce/
├── data/
│   ├── PennFudanPed/
│   └── custom_object/
└── weights/
    └── mask_rcnn_pennfudan.pth
```

## 5. 新电脑如何创建虚拟环境

进入项目根目录：

```powershell
cd path\to\mask_rcnn_reproduce
```

创建并安装基础依赖：

```powershell
.\setup_environment.ps1
```

如果系统里 `python` 不是 Python 3.12，可以指定 Python：

```powershell
.\setup_environment.ps1 -Python "C:\Path\To\Python312\python.exe"
```

## 6. 如何安装依赖

基础安装：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-base.txt
```

如果要尽量复刻旧电脑环境，可以使用：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
```

注意：`requirements-lock.txt` 是旧电脑的完整冻结环境，可能包含只适合当前 CUDA、Python 或 Windows 版本的包。新电脑如果 CUDA 或 Python 版本不同，优先使用 `requirements-base.txt`，再按报错补装。

## 7. 如何验证项目

先生成环境报告：

```powershell
& ".\.venv\Scripts\python.exe" environment_report.py
```

再运行完整验证：

```powershell
& ".\.venv\Scripts\python.exe" verify_project.py
```

验证脚本会检查：

- 核心 Python 模块能否导入。
- CUDA 是否可用。
- 数据目录是否存在。
- 权重是否存在且非空。
- 模型是否能加载。
- 至少一张图片能否完成推理。
- 输出图片是否真实生成。

验证输出图片：

```text
outputs/verify_project_result.png
```

## 8. 常见错误和恢复方法

### 找不到权重

错误表现：

```text
权重文件不存在
```

恢复方法：

把旧电脑备份的权重放回：

```text
weights/mask_rcnn_pennfudan.pth
```

### CUDA 不可用

错误表现：

```text
CUDA 不可用，使用 CPU 验证
```

这不一定会导致项目失败，只是推理会变慢。需要 GPU 时，请检查：

- NVIDIA 驱动是否安装。
- PyTorch 是否是 CUDA 版本。
- `torch.cuda.is_available()` 是否为 `True`。

### 缺少 pycocotools

恢复方法：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install pycocotools
```

### 找不到数据集

错误表现：

```text
data/PennFudanPed 不存在
```

恢复方法：

把旧电脑备份的数据目录放回：

```text
data/PennFudanPed/
```

### 输出目录不存在

不用手动创建。`verify_project.py` 会自动创建：

```text
outputs/
```

## 9. 当前迁移原则

- 不改变模型结构。
- 不改变类别数。
- 不覆盖已有权重。
- 所有项目路径都从脚本所在位置或项目根目录推导。
- 新电脑验证以 `verify_project.py` 为准。
