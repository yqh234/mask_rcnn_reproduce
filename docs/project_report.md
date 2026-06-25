# Mask R-CNN 项目阶段报告

生成时间：2026-06-25  
项目目录：`C:\Users\86136\mask_rcnn_reproduce`

## 1. 项目目前做了什么

这个项目主要是复现 TorchVision 里的 Mask R-CNN，并在 Penn-Fudan 行人数据集上做训练后推理测试。

目前我已经完成了下面几部分：

1. 用已经训练好的 Penn-Fudan 权重做推理。
2. 对 10 张图片测试了 3 个不同的置信度阈值。
3. 整理了项目迁移需要的环境检查脚本。
4. 准备了一个小的自定义数据集流程。
5. 用自定义数据集跑了 2 轮冒烟训练，确认训练流程可以正常运行。

这里的冒烟训练不是正式训练，只是用很少的数据检查代码能不能跑通。

## 2. 当前运行环境

当前电脑的环境如下：

| 项目 | 内容 |
|---|---|
| Python | 3.12.7 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA | 12.8 |
| CUDA 是否可用 | true |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 系统 | Windows 11 |

运行脚本时使用这个 Python：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" 脚本名.py
```

## 3. 主要文件说明

Penn-Fudan 相关文件：

| 文件 | 说明 |
|---|---|
| `train_pennfudan.py` | 原来的 Penn-Fudan 训练脚本 |
| `infer_trained.py` | 加载训练好的权重并推理 |
| `infer_image.py` | 单张图片推理 |
| `evaluate_thresholds.py` | 批量测试 10 张图片和 3 个阈值 |
| `weights/mask_rcnn_pennfudan.pth` | 已经训练好的 Penn-Fudan 权重 |

迁移和检查相关文件：

| 文件 | 说明 |
|---|---|
| `requirements-base.txt` | 基础依赖 |
| `requirements-lock.txt` | 当前环境依赖记录 |
| `environment_report.py` | 输出 Python、torch、CUDA、GPU 等环境信息 |
| `verify_project.py` | 检查项目是否能正常加载模型并推理 |
| `setup_environment.ps1` | 新电脑上创建环境用的脚本 |
| `MIGRATION.md` | 项目迁移说明 |
| `NEW_COMPUTER_SETUP.md` | 新电脑运行步骤 |

自定义数据集相关文件：

| 文件 | 说明 |
|---|---|
| `custom_dataset.py` | 读取自定义数据集 |
| `check_custom_dataset.py` | 检查标注并生成预览图 |
| `scripts/bootstrap_custom_samples.py` | 准备少量自定义样本 |
| `scripts/create_custom_splits.py` | 划分 train、val、test |
| `scripts/smoke_test_custom_dataset.py` | 快速检查 Dataset 输出 |
| `train_custom.py` | 用自定义数据集做 2 轮冒烟训练 |

## 4. Penn-Fudan 阈值测试结果

这次测试从 `data/PennFudanPed/PNGImages` 里按文件名排序，取前 10 张图片。对应的真实掩膜来自 `data/PennFudanPed/PedMasks`。

测试的置信度阈值是：

```text
0.30
0.50
0.70
```

掩膜二值化阈值固定为 `0.50`。

测试结果如下：

| 置信度阈值 | 测试图片数 | 预测目标总数 | 真实目标总数 | 平均绝对数量误差 | 完全预测正确图片数 | 平均推理时间(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 10 | 25 | 18 | 0.700 | 5 | 179.15 |
| 0.50 | 10 | 22 | 18 | 0.400 | 7 | 179.15 |
| 0.70 | 10 | 21 | 18 | 0.300 | 7 | 179.15 |

从这 10 张图片看，`0.70` 的数量预测最好，平均绝对数量误差是 `0.300`。  
`0.30` 阈值比较低，所以模型会留下更多预测框，预测人数也最多，但多报的情况更明显。  
`0.50` 和 `0.70` 都有 7 张图片预测人数完全正确，不过 `0.70` 的总误差更小。

结果保存位置：

```text
outputs/threshold_evaluation/
├── threshold_0.30/
├── threshold_0.50/
├── threshold_0.70/
├── threshold_report.csv
└── summary.md
```

每个阈值目录里都有 10 张可视化图片。

## 5. 自定义数据集冒烟训练

自定义数据集放在：

```text
data/custom_object/
├── images/
├── masks/
├── annotations_preview/
└── splits/
```

我用下面的命令跑了 2 轮训练：

```powershell
& "C:\Users\86136\detectron2\.venv\Scripts\python.exe" train_custom.py
```

运行时控制台显示使用了 CUDA：

```text
使用设备：cuda
CUDA 设备：NVIDIA GeForce RTX 4060 Laptop GPU
训练样本数：3
```

训练日志如下：

| epoch | batch 数 | total_loss | loss_classifier | loss_box_reg | loss_mask | loss_objectness | loss_rpn_box_reg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 4.912454 | 0.455325 | 0.395692 | 3.967107 | 0.052372 | 0.041957 |
| 2 | 3 | 2.003402 | 0.241176 | 0.381475 | 1.319494 | 0.012242 | 0.049016 |

第 2 轮的 `total_loss` 比第 1 轮低，说明模型参数确实在更新，训练流程可以跑通。不过这个数据集太小，所以这个结果不能说明模型已经训练好了，只能说明代码流程没有问题。

训练后生成的文件：

```text
weights/custom_smoke_epoch_1.pth
weights/custom_smoke_epoch_2.pth
experiments/custom_smoke/training_log.csv
experiments/custom_smoke/config.json
```

这些是本地运行结果，没有放进 GitHub。

## 6. 数据流和张量形状

### Penn-Fudan 推理

推理时，程序先读取图片，再把图片转成张量。输入图片的形状一般是：

```text
[3, H, W]
```

其中 `3` 表示 RGB 三个通道，`H` 和 `W` 是图片高度和宽度。

模型输出主要包括：

| 名称 | 形状 | 含义 |
|---|---|---|
| `boxes` | `[N, 4]` | N 个预测框，每个框是 `x1, y1, x2, y2` |
| `labels` | `[N]` | 每个目标的类别 |
| `scores` | `[N]` | 每个目标的置信度 |
| `masks` | `[N, 1, H, W]` | 每个目标的实例分割掩膜 |

在 `evaluate_thresholds.py` 里，同一张图片只推理一次。得到结果后，再分别用 `0.30`、`0.50`、`0.70` 三个阈值筛选预测框，这样比较三个阈值时更公平。

### 自定义数据集训练

训练时，每个样本包括一张图片和一个 target。

图片张量形状是：

```text
[3, H, W]
```

target 里主要有：

| 字段 | 形状 | 含义 |
|---|---|---|
| `boxes` | `[N, 4]` | 目标框 |
| `labels` | `[N]` | 类别编号 |
| `masks` | `[N, H, W]` | 实例掩膜 |
| `image_id` | `[1]` | 图片编号 |
| `area` | `[N]` | 每个框的面积 |
| `iscrowd` | `[N]` | COCO 格式需要的字段 |

`DataLoader` 通过 `collate_fn` 把数据整理成 `images list` 和 `targets list`。然后模型根据这些数据计算多个 loss，例如分类 loss、框回归 loss、mask loss 等。脚本把这些 loss 加起来，反向传播更新模型参数。

## 小结

目前这个项目已经可以完成 Penn-Fudan 数据集上的推理和阈值比较，也能跑通一个很小的自定义数据集训练流程。

我觉得目前最有用的结果是阈值测试。因为它能看出不同置信度阈值对预测人数的影响：阈值低时预测更多，容易多报；阈值高时会过滤掉一些不确定结果，在这 10 张图片上 `0.70` 的数量误差最小。

自定义数据集部分还只是开始。后面如果要继续做，应该先增加数据量，并人工检查标注质量，然后再进行正式训练。
