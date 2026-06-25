# 当前 Penn-Fudan Mask R-CNN 基线

## 项目环境

- 操作系统：Windows
- 终端：PowerShell
- Python：3.12.7
- GPU：NVIDIA RTX 4060
- Python 解释器：`C:\Users\86136\detectron2\.venv\Scripts\python.exe`
- 深度学习框架：PyTorch 2.11.0+cu128，TorchVision 0.26.0+cu128
- 当前使用：TorchVision Mask R-CNN，不是 Detectron2

运行脚本时使用：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" 脚本名.py
```

## 目录结构摘要

```text
mask_rcnn_reproduce/
├── data/
│   └── PennFudanPed/
│       ├── PNGImages/    170 张原图
│       ├── PedMasks/     170 张实例 mask
│       └── Annotation/   170 个标注文件
├── weights/
│   └── mask_rcnn_pennfudan.pth
├── outputs/
├── experiments/
├── train_pennfudan.py
├── infer_trained.py
├── infer_image.py
├── evaluate_thresholds.py
├── engine.py
├── utils.py
├── coco_utils.py
├── coco_eval.py
└── transforms.py
```

## 类别定义

```text
0 = 背景
1 = 行人
```

模型类别数：

```text
num_classes = 2
```

## 模型结构

训练脚本使用 TorchVision 的 Mask R-CNN：

```python
torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
```

随后替换两个预测头：

- `FastRCNNPredictor`：边界框分类和回归。
- `MaskRCNNPredictor`：实例分割 mask 预测。

训练轮数：

```text
num_epochs = 10
```

训练权重保存位置：

```text
weights/mask_rcnn_pennfudan.pth
```

当前权重大小：

```text
176,229,189 bytes
```

注意：该权重文件大于 GitHub 普通单文件 100MB 限制，因此不建议直接提交到 GitHub。

## 已验证脚本

### 权重加载

命令：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" infer_trained.py
```

输入：

- `weights/mask_rcnn_pennfudan.pth`

输出：

- 控制台确认模型结构创建成功、权重加载成功、模型处于推理模式。

验证结果：

```text
权重加载成功
使用设备： cuda
模型状态： 推理模式
```

### 单图推理

命令：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" infer_image.py
```

输入：

- `data/PennFudanPed/PNGImages/FudanPed00001.png`
- `weights/mask_rcnn_pennfudan.pth`

输出：

- `outputs/trained_result.png`

张量形状：

```text
输入图片形状：torch.Size([3, 536, 559])
boxes：torch.Size([2, 4])
labels：torch.Size([2])
scores：torch.Size([2])
masks：torch.Size([2, 1, 536, 559])
```

说明：

- 输入图片张量是 `[C, H, W]`。
- `C=3` 表示 RGB 三个通道。
- `H=536`，`W=559`。
- `boxes` 中每一行是 `[x1, y1, x2, y2]`。
- `masks` 中每个实例 mask 的大小和原图一致。

### 多图多阈值评估

命令：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" evaluate_thresholds.py
```

输入：

- `data/PennFudanPed/PNGImages` 中按文件名排序的前 10 张图片。
- 对应的 `data/PennFudanPed/PedMasks` 实例 mask。
- `weights/mask_rcnn_pennfudan.pth`。

输出：

```text
outputs/threshold_evaluation/
├── threshold_0.30/
├── threshold_0.50/
├── threshold_0.70/
├── threshold_report.csv
└── summary.md
```

结果摘要：

| 置信度阈值 | 测试图片数 | 预测目标总数 | 真实目标总数 | 平均绝对数量误差 | 完全预测正确图片数 | 平均推理时间(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 10 | 25 | 18 | 0.700 | 5 | 130.83 |
| 0.50 | 10 | 22 | 18 | 0.400 | 7 | 130.83 |
| 0.70 | 10 | 21 | 18 | 0.300 | 7 | 130.83 |

当前 10 张快速测试图片上，数量预测最好的阈值是：

```text
0.70
```

## 数据流

单图推理的数据流：

```text
PNG 图片
-> PIL.Image RGB
-> torch.Tensor，形状 [3, H, W]，数值范围 0 到 1
-> Mask R-CNN
-> boxes / labels / scores / masks
-> 置信度筛选
-> mask 二值化
-> 绘制边界框和 mask
-> 保存结果图
```

多阈值评估的数据流：

```text
前 10 张 PNG 图片
-> 对应 PedMasks 统计真实行人数
-> 模型只加载一次
-> 每张图片推理一次
-> 用 0.30、0.50、0.70 三个阈值分别筛选
-> 保存三组可视化图
-> 写入 CSV
-> 汇总 Markdown 报告
```

## 已发现问题

1. `train_pennfudan.py` 中仍有官方教程残留的 `os.system("wget ...")` 代码。当前 Windows 项目规则要求不使用 `wget`，所以不建议直接运行该训练脚本。
2. `train_pennfudan.py` 的训练逻辑主要位于顶层代码，没有清晰的 `main()` 入口保护。导入或运行时可能触发下载、训练等副作用。
3. 项目目录原本不是独立 Git 仓库；上层 `C:\Users\86136` 是 Git 根目录，会导致 `git status` 混入大量用户目录文件。
4. 训练权重文件超过 GitHub 普通单文件大小限制，上传代码仓库时应忽略该文件，并在本地保留。

## 建议的最小整理方案

1. 保留现有可运行推理脚本，不改变模型结构。
2. 用本文档记录当前可复现基线。
3. 将项目目录初始化为独立 Git 仓库。
4. 使用 `.gitignore` 排除权重、缓存、虚拟环境和输出图。
5. 后续如需整理训练脚本，应单独处理，并保持模型结构和训练行为不变。
