# Penn-Fudan 官方基线封存

本文档封存当前已经跑通的 TorchVision Mask R-CNN Penn-Fudan 基线，方便后续和自制数据集、YOLO 或机械臂抓取流程做对照。

## 数据集

- 数据集：Penn-Fudan Pedestrian
- 本地位置：`data/PennFudanPed/`
- 原图目录：`data/PennFudanPed/PNGImages/`
- 实例 mask 目录：`data/PennFudanPed/PedMasks/`
- 标注目录：`data/PennFudanPed/Annotation/`
- 当前数量：
  - 原图：170 张
  - 实例 mask：170 张
  - 标注文件：170 个

## 类别

```text
0 = 背景
1 = 行人
```

模型类别数：

```text
num_classes = 2
```

## 模型

- 模型来源：TorchVision
- 模型结构：Mask R-CNN ResNet-50 FPN
- 训练脚本：`train_pennfudan.py`
- 推理时复用的模型构造函数：`infer_trained.get_model_instance_segmentation()`

训练脚本中的核心结构：

```python
torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
```

然后替换：

- `FastRCNNPredictor`
- `MaskRCNNPredictor`

## 权重

本地权重：

```text
weights/mask_rcnn_pennfudan.pth
```

当前权重文件信息：

```text
大小：176,229,189 bytes
修改时间：2026/6/25 13:53:13
```

注意：该权重文件超过 GitHub 普通单文件 100MB 限制，已在 `.gitignore` 中忽略。上传仓库时只提交代码、数据和文档，不提交该权重。

## 推理命令

权重加载验证：

```powershell
& ".\.venv\Scripts\python.exe" infer_trained.py
```

单图推理：

```powershell
& ".\.venv\Scripts\python.exe" infer_image.py
```

多图片、多阈值评估：

```powershell
& ".\.venv\Scripts\python.exe" evaluate_thresholds.py
```

## 多阈值评估输出

输出目录：

```text
outputs/threshold_evaluation/
├── threshold_0.30/
├── threshold_0.50/
├── threshold_0.70/
├── threshold_report.csv
└── summary.md
```

已检查结果：

```text
threshold_0.30：10 张结果图
threshold_0.50：10 张结果图
threshold_0.70：10 张结果图
threshold_report.csv：30 行结果
summary.md：非空
```

## 三个阈值结果

| 置信度阈值 | 测试图片数 | 预测目标总数 | 真实目标总数 | 平均绝对数量误差 | 完全预测正确图片数 |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 10 | 25 | 18 | 0.700 | 5 |
| 0.50 | 10 | 22 | 18 | 0.400 | 7 |
| 0.70 | 10 | 21 | 18 | 0.300 | 7 |

当前 10 张快速测试图片上，数量预测最好的阈值是：

```text
0.70
```

原因：`0.70` 的平均绝对数量误差最低，为 `0.300`；完全预测正确图片数为 `7`，与 `0.50` 持平，但误差更小。

## 已知失败案例

本次 `threshold_report.csv` 中没有 `status = failed` 的图片记录。也就是说，前 10 张快速测试图片都成功完成推理和可视化保存。

需要注意的是，这里说的“没有失败案例”只表示脚本运行没有失败，不代表模型在每张图上的预测都完全正确。按数量误差看：

- `0.30` 有 5 张图片数量完全正确。
- `0.50` 有 7 张图片数量完全正确。
- `0.70` 有 7 张图片数量完全正确。

## 数据流

多阈值评估的数据流：

```text
PNGImages 中前 10 张图片
-> PIL 读取为 RGB
-> 转成 Tensor，形状 [3, H, W]
-> Mask R-CNN 推理一次
-> 得到 boxes / labels / scores / masks
-> 分别用 0.30、0.50、0.70 三个置信度阈值筛选
-> mask 固定用 0.50 二值化
-> 从 PedMasks 统计真实行人数
-> 保存可视化图片
-> 写入 threshold_report.csv
-> 汇总 summary.md
```

输出中的主要张量含义：

- `boxes`：形状 `[N, 4]`，每一行是一个目标框 `[x1, y1, x2, y2]`。
- `labels`：形状 `[N]`，类别编号。当前只关心 `1 = 行人`。
- `scores`：形状 `[N]`，模型对每个目标的置信度。
- `masks`：形状 `[N, 1, H, W]`，每个目标对应一张实例 mask。

## Git 与提交建议

`.gitignore` 已检查，当前会忽略：

- `__pycache__/`
- `outputs/`
- `experiments/`
- `weights/*.pth`
- `weights/*.pt`
- `weights/*.ckpt`
- `*.zip`
- 常见虚拟环境和编辑器缓存

建议提交文件清单：

```text
docs/pennfudan_baseline.md
```

如果后续希望把本次文档同步到 GitHub，可以执行：

```powershell
git add docs/pennfudan_baseline.md
git commit -m "Document Penn-Fudan baseline results"
git push
```

当前步骤不自动执行 `git commit`，等待用户确认。

## 后续注意事项

1. 不要直接提交 `weights/mask_rcnn_pennfudan.pth`。
2. 不要把 `outputs/threshold_evaluation/` 当作源码提交；它是运行产物。
3. `train_pennfudan.py` 仍有官方教程残留的 `wget` 代码，后续整理训练脚本时应单独处理。
4. 这个基线只覆盖 Penn-Fudan 行人实例分割，后续自制物体数据集需要重新建立类别、数据读取器和训练脚本。
