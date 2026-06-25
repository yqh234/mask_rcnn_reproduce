# 自制单类实例分割数据集说明

这个目录用于放置你自己拍摄并标注的可抓取物体数据。当前阶段只建立数据结构和命名规范，不训练模型，也不生成虚假图片。

## 目录结构

```text
data/custom_object/
├── images/               原图
├── masks/                实例 mask
├── splits/               train/val/test 划分文件
├── annotations_preview/  标注检查预览图
└── README.md
```

## 图片和 mask 命名

推荐一张原图对应一张实例 mask，文件名一一对应：

```text
images/0001.jpg
masks/0001_mask.png

images/0002.png
masks/0002_mask.png
```

规则：

- 原图文件名可以是 `0001.jpg`、`0001.png` 这类简单名字。
- mask 文件名推荐使用 `原图文件名_stem + _mask.png`。
- 例如原图是 `cup_001.jpg`，对应 mask 应为 `cup_001_mask.png`。

## 推荐图片格式

原图推荐使用：

- `.jpg`
- `.jpeg`
- `.png`

mask 推荐使用：

- `.png`

原因是 PNG 是无损格式，适合保存像素级标注。

## mask 像素值规则

mask 是单通道灰度图，每个像素值表示它属于哪个实例。

```text
背景 = 0
物体1 = 1
物体2 = 2
物体3 = 3
```

重要规则：

1. 背景像素值必须为 `0`。
2. 同一张图中，不同物体实例必须使用不同的非零整数值。
3. 同一个物体实例内部应使用同一个像素值。
4. mask 不能是彩色语义图，也不要用随机颜色代表实例。

## 宽高要求

原图和 mask 的宽高必须完全一致。

例如：

```text
images/0001.jpg      1280 x 720
masks/0001_mask.png  1280 x 720
```

如果宽高不一致，训练时 box 和 mask 会对不上。

## 建议数据划分

建议按下面比例划分：

```text
70% 训练集 train
20% 验证集 val
10% 测试集 test
```

本项目提供了脚本：

```powershell
& ".\.venv\Scripts\python.exe" scripts/create_custom_splits.py
```

脚本会按固定随机种子生成：

```text
data/custom_object/splits/train.txt
data/custom_object/splits/val.txt
data/custom_object/splits/test.txt
```

每一行记录一个样本名，也就是原图的 `stem`，例如：

```text
0001
cup_001
```

## 如何检查标注

训练前应该检查：

1. 每张原图是否都有对应 mask。
2. 每张 mask 是否至少包含一个非零实例。
3. 原图和 mask 宽高是否一致。
4. mask 中不同物体是否使用不同实例编号。
5. 可视化预览中，mask 是否覆盖在正确物体上。
6. 自动计算出的边界框是否包住目标。

后续可以使用 `check_custom_dataset.py` 生成预览图和 CSV 报告。

## 常见错误标注

下面这些都属于错误标注：

- mask 全黑，只有背景 `0`。
- 原图存在，但对应 mask 不存在。
- mask 存在，但对应原图不存在。
- 原图和 mask 宽高不一致。
- 多个不同物体用了同一个非零像素值。
- 背景不是 `0`。
- mask 是彩色图，像素值不是清晰的实例编号。
- 物体边缘大面积偏离真实物体。
- 同一个物体被断成多个不连续编号，但实际不想区分为多个实例。
- 文件名无法一一对应，例如 `0001.jpg` 配了 `abc_mask.png`。

## 当前阶段

当前只建立数据目录和规范。请先拍摄并标注真实图片，再进入数据读取器、标注检查和训练步骤。

## 可选：自动找少量演示样本

如果只是为了先跑通数据读取和 smoke test，可以使用：

```powershell
& ".\.venv\Scripts\python.exe" scripts/bootstrap_custom_samples.py
```

这个脚本会从 Wikimedia Commons 搜索少量公开图片，再用 TorchVision 的 COCO 预训练 Mask R-CNN 自动生成实例 mask。它生成的是“伪标注”，适合学习流程和检查代码，不等同于人工精修数据。

生成后会得到：

```text
data/custom_object/images/
data/custom_object/masks/
data/custom_object/source_manifest.csv
```

正式训练前，请务必用标注可视化检查器确认 mask 是否贴合物体。
