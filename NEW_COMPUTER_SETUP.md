# 新电脑恢复运行步骤

这份文档只写“换一台电脑后要怎么做”。目标是让项目在新电脑上恢复到可以加载 Mask R-CNN 权重、读取 Penn-Fudan 数据并完成一次推理验证的状态。

## 需要先准备的东西

新电脑需要：

- Windows
- PowerShell
- Python 3.12.x
- NVIDIA 显卡和驱动，推荐有 CUDA 支持
- Git

旧电脑需要单独备份的大文件：

```text
weights/mask_rcnn_pennfudan.pth
```

如果自制数据集、输出结果或实验记录没有放进 Git，也要单独备份：

```text
data/custom_object/images/
data/custom_object/masks/
outputs/
experiments/
```

其中 `outputs/` 和 `experiments/` 不是运行必需，只是结果和记录。

## 第一步：克隆项目

在新电脑上打开 PowerShell：

```powershell
git clone https://github.com/yqh234/mask_rcnn_reproduce.git
cd mask_rcnn_reproduce
```

## 第二步：创建虚拟环境并安装依赖

推荐直接运行项目里的安装脚本：

```powershell
.\setup_environment.ps1
```

它会创建：

```text
.venv/
```

并安装：

```text
requirements-base.txt
```

如果 `python` 命令不是 Python 3.12，可以指定解释器：

```powershell
.\setup_environment.ps1 -Python "C:\Path\To\Python312\python.exe"
```

## 第三步：恢复权重

把旧电脑备份的权重文件放回：

```text
weights/mask_rcnn_pennfudan.pth
```

最终目录应类似：

```text
mask_rcnn_reproduce/
├── weights/
│   └── mask_rcnn_pennfudan.pth
├── data/
│   └── PennFudanPed/
└── verify_project.py
```

注意：权重文件没有放进 Git，因为它超过 GitHub 普通单文件 100MB 限制。

## 第四步：生成环境报告

运行：

```powershell
& ".\.venv\Scripts\python.exe" environment_report.py
```

它会打印并保存：

```text
environment_report.json
```

里面包含：

- Python 版本
- torch 版本
- torchvision 版本
- torch CUDA 版本
- CUDA 是否可用
- GPU 名称
- 操作系统

## 第五步：验证项目是否恢复成功

运行：

```powershell
& ".\.venv\Scripts\python.exe" verify_project.py
```

验证脚本会检查：

- 核心模块能否导入
- CUDA 是否可用
- 数据目录是否存在
- 权重是否存在且非空
- 模型是否能够加载
- 至少一张图片能否完成推理
- 输出图片是否真实生成

成功后会看到：

```text
项目验证通过。
```

并生成：

```text
outputs/verify_project_result.png
```

## 常见问题

### 1. 找不到权重

检查文件是否在这里：

```text
weights/mask_rcnn_pennfudan.pth
```

如果没有，把旧电脑备份的权重复制回来。

### 2. CUDA 不可用

如果看到：

```text
CUDA 不可用，使用 CPU 验证
```

项目仍然可以运行，只是推理会慢。需要 GPU 时检查：

- NVIDIA 驱动是否安装
- PyTorch 是否为 CUDA 版本
- `torch.cuda.is_available()` 是否为 `True`

### 3. 缺少依赖

先尝试：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-base.txt
```

如果想尽量复刻旧电脑环境，再尝试：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` 是旧电脑完整环境，可能和新电脑 CUDA 或 Python 版本不完全匹配。

### 4. 找不到数据集

检查：

```text
data/PennFudanPed/
```

里面应该有：

```text
PNGImages/
PedMasks/
Annotation/
```

### 5. PowerShell 不允许运行脚本

如果 `setup_environment.ps1` 被策略拦截，可以在当前 PowerShell 窗口临时允许：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_environment.ps1
```

## 推荐的新电脑验证命令顺序

完整顺序如下：

```powershell
git clone https://github.com/yqh234/mask_rcnn_reproduce.git
cd mask_rcnn_reproduce
.\setup_environment.ps1
```

然后手动复制权重：

```text
weights/mask_rcnn_pennfudan.pth
```

再运行：

```powershell
& ".\.venv\Scripts\python.exe" environment_report.py
& ".\.venv\Scripts\python.exe" verify_project.py
```

如果最后看到 `项目验证通过。`，说明新电脑已经恢复成功。
