# Mask R-CNN 项目阶段报告

生成时间：2026-06-25  
项目目录：`C:\Users\86136\mask_rcnn_reproduce`  
远程仓库：`https://github.com/yqh234/mask_rcnn_reproduce`

## 1. 当前目标

这个项目用于快速复现 TorchVision 版 Mask R-CNN。当前已经完成三件事：

1. 在 Penn-Fudan 行人数据集上加载已训练权重，并做多图片、多阈值推理测试。
2. 把项目整理成可以迁移到另一台电脑继续运行的结构。
3. 准备了一个很小的自定义数据集流程，并完成 2 轮冒烟训练，确认训练链路可以跑通。

这里的“冒烟训练”不是正式训练。它的作用是检查数据集读取、模型创建、loss 计算、反向传播、权重保存这些步骤是否正常。

## 2. 环境记录

当前验证环境如下，来自 `environment_report.json`：

| 项目 | 当前值 |
|---|---|
| Python | 3.12.7 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torch CUDA | 12.8 |
| CUDA 是否可用 | true |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 操作系统 | Windows 11 |

运行 Python 脚本时使用：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" 脚本名.py
```

## 3. 已完成的主要文件

### Penn-Fudan 训练和推理

| 文件 | 作用 |
|---|---|
| `train_pennfudan.py` | Penn-Fudan 训练脚本 |
| `infer_trained.py` | 加载训练后权重并推理 |
| `infer_image.py` | 单张图片推理 |
| `evaluate_thresholds.py` | 10 张图片、3 个置信度阈值的批量评估 |
| `weights/mask_rcnn_pennfudan.pth` | 已训练好的 Penn-Fudan 权重，本地保留，不提交 Git |

`evaluate_thresholds.py` 复用了项目里的模型结构，读取：

- 图片：`data/PennFudanPed/PNGImages`
- 真实掩膜：`data/PennFudanPed/PedMasks`
- 权重：`weights/mask_rcnn_pennfudan.pth`

输出目录：

```text
outputs/threshold_evaluation/
├── threshold_0.30/
├── threshold_0.50/
├── threshold_0.70/
├── threshold_report.csv
└── summary.md
```

### 项目迁移文件

| 文件 | 作用 |
|---|---|
| `requirements-base.txt` | 基础依赖列表 |
| `requirements-lock.txt` | 当前环境依赖记录 |
| `environment_report.py` | 输出 Python、torch、CUDA、GPU、系统信息 |
| `verify_project.py` | 检查项目能否在当前电脑跑通 |
| `setup_environment.ps1` | 新电脑创建环境和安装依赖的 PowerShell 脚本 |
| `MIGRATION.md` | 中文迁移说明 |
| `NEW_COMPUTER_SETUP.md` | 新电脑恢复步骤 |
| `.gitignore` | 排除数据、权重、实验输出等大文件 |

### 自定义数据集流程

| 文件 | 作用 |
|---|---|
| `custom_dataset.py` | 读取自定义目标检测/实例分割数据集 |
| `check_custom_dataset.py` | 检查自定义数据和标注，并生成预览图 |
| `scripts/bootstrap_custom_samples.py` | 生成少量自定义样本 |
| `scripts/create_custom_splits.py` | 生成 train/val/test 划分 |
| `scripts/smoke_test_custom_dataset.py` | 快速检查 Dataset 输出 |
| `train_custom.py` | 对自定义数据集做 2 轮冒烟训练 |

当前自定义数据集位于：

```text
data/custom_object/
├── images/
├── masks/
├── annotations_preview/
└── splits/
```

## 4. Penn-Fudan 阈值评估结果

评估样本：按文件名排序后的前 10 张 Penn-Fudan 图片。  
置信度阈值：`0.30`、`0.50`、`0.70`。  
掩膜二值化阈值固定为：`0.50`。

汇总结果：

| 置信度阈值 | 测试图片数 | 预测目标总数 | 真实目标总数 | 平均绝对数量误差 | 完全预测正确图片数 | 平均推理时间(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 10 | 25 | 18 | 0.700 | 5 | 179.15 |
| 0.50 | 10 | 22 | 18 | 0.400 | 7 | 179.15 |
| 0.70 | 10 | 21 | 18 | 0.300 | 7 | 179.15 |

这 10 张图里，`0.70` 的数量预测最好，平均绝对数量误差最低，为 `0.300`。  
`0.30` 阈值更宽松，会保留更多预测框，所以预测总数最高，也更容易多报。  
`0.50` 和 `0.70` 的完全预测正确图片数都是 7 张，但 `0.70` 的总数量误差更小。

结果图片已经生成：

```text
outputs/threshold_evaluation/threshold_0.30/  10 张
outputs/threshold_evaluation/threshold_0.50/  10 张
outputs/threshold_evaluation/threshold_0.70/  10 张
```

## 5. 自定义数据集冒烟训练结果

运行命令：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" train_custom.py
```

控制台确认使用 CUDA：

```text
使用设备：cuda
CUDA 设备：NVIDIA GeForce RTX 4060 Laptop GPU
训练样本数：3
```

训练日志：

| epoch | batch 数 | total_loss | loss_classifier | loss_box_reg | loss_mask | loss_objectness | loss_rpn_box_reg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 4.912454 | 0.455325 | 0.395692 | 3.967107 | 0.052372 | 0.041957 |
| 2 | 3 | 2.003402 | 0.241176 | 0.381475 | 1.319494 | 0.012242 | 0.049016 |

第 2 轮的 `total_loss` 低于第 1 轮，说明训练流程在这组小样本上可以正常更新参数。这个结果只能证明链路能跑通，不能说明模型已经具备可靠泛化能力。

本地生成文件：

```text
weights/custom_smoke_epoch_1.pth
weights/custom_smoke_epoch_2.pth
experiments/custom_smoke/training_log.csv
experiments/custom_smoke/config.json
```

这些文件没有提交到 GitHub。它们属于运行产物，应该在本机或移动硬盘里单独备份。

## 6. 数据如何流动

### Penn-Fudan 推理评估

1. `evaluate_thresholds.py` 从 `PNGImages` 读取前 10 张图片。
2. 同名掩膜从 `PedMasks` 读取。
3. 真实人数通过掩膜中不同的非零像素值计算。
4. 模型只加载一次，放到 CUDA 或 CPU。
5. 每张图只推理一次，得到 `boxes`、`labels`、`scores`、`masks`。
6. 对同一份推理结果分别应用 `0.30`、`0.50`、`0.70` 三个置信度阈值。
7. 保存可视化图片，同时写入 CSV 和 Markdown 汇总。

常见张量形状：

| 名称 | 形状 | 含义 |
|---|---|---|
| 输入图片 | `[3, H, W]` | 3 通道 RGB 图片 |
| boxes | `[N, 4]` | N 个预测框，格式为 `x1, y1, x2, y2` |
| labels | `[N]` | 每个预测框的类别 |
| scores | `[N]` | 每个预测框的置信度 |
| masks | `[N, 1, H, W]` | 每个实例的预测掩膜 |

### 自定义数据集训练

1. `CustomObjectDataset` 读取 `data/custom_object`。
2. 每张图片转成 `[3, H, W]` 的 `Tensor`。
3. 每个 target 包含 `boxes`、`labels`、`masks`、`image_id`、`area`、`iscrowd`。
4. `DataLoader` 使用 `collate_fn`，把 batch 整理成 `images list` 和 `targets list`。
5. `train_custom.py` 创建 COCO 预训练 Mask R-CNN，并把分类头和掩膜头替换为 2 类。
6. 模型前向返回多个 loss，脚本求和后反向传播。
7. 每个 epoch 保存一次权重，并把 loss 写入 CSV。

## 7. GitHub 提交记录

当前已经推送到 `main` 分支的提交：

```text
9b2c42f Add custom dataset smoke training
bf72704 Update migration guide for submitted project state
0cfb13b Add custom dataset bootstrap and checks
916cc8e Add migration setup guide
0ef0fe2 Add Penn-Fudan Mask R-CNN baseline
```

GitHub 保存的是源码、脚本和文档。数据集、权重和实验输出没有进入 Git，这是为了避免仓库过大，也避免误覆盖本地训练产物。

## 8. 新电脑恢复顺序

在新电脑上建议按这个顺序做：

1. 克隆仓库。

```powershell
git clone https://github.com/yqh234/mask_rcnn_reproduce.git
cd mask_rcnn_reproduce
```

2. 创建并激活虚拟环境。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. 安装依赖。

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt
```

4. 单独恢复数据和权重。

需要从旧电脑或备份盘拷贝：

```text
data/
weights/
experiments/   可选，主要用于保留实验记录
outputs/       可选，主要用于保留历史输出图片和报告
```

5. 验证项目。

```powershell
python environment_report.py
python verify_project.py
```

如果要重新跑阈值评估：

```powershell
python evaluate_thresholds.py
```

如果要重新跑自定义数据集冒烟训练：

```powershell
python train_custom.py
```

## 9. 仍需注意的地方

- `weights/mask_rcnn_pennfudan.pth` 是重要权重，必须单独备份。
- `data/` 目录没有提交到 GitHub，新电脑需要手动复制。
- `weights/custom_smoke_epoch_*.pth` 只是冒烟训练权重，不建议当成正式模型使用。
- 自定义数据集样本很少，目前只适合验证流程，不适合评估真实效果。
- 如果新电脑没有 NVIDIA GPU，脚本会回退到 CPU，但训练和推理会慢很多。
- 如果安装 PyTorch 时 CUDA 版本不匹配，先运行 `environment_report.py` 看 `cuda_available` 是否为 `true`。

## 10. 当前结论

当前项目已经具备三个基础能力：

1. Penn-Fudan 已训练权重可以加载，并能批量评估不同置信度阈值。
2. 项目路径基本改成基于项目根目录的相对路径，更适合迁移。
3. 自定义数据集从检查、划分、预览到 2 轮训练都已经跑通。

下一步如果继续做，建议不要直接长时间训练。更稳妥的顺序是：先扩大自定义数据集，人工检查标注质量，再设置正式训练轮数和验证指标。
