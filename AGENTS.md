# 项目说明

你正在维护一个面向初学者的计算机视觉与机械臂抓取复现项目。协作时优先保持代码简单、可运行、容易解释，不做不必要的大改。

## 当前项目

- 项目目录：当前仓库根目录
- 推荐 Python 解释器：`.\.venv\Scripts\python.exe`

## 当前环境

- Windows
- PowerShell
- Python 3.12.7
- NVIDIA RTX 4060
- PyTorch 与 TorchVision 已安装
- 当前使用 TorchVision Mask R-CNN，不是 Detectron2
- Penn-Fudan 模型已经完成训练、保存和加载

运行 Python 文件时使用：

```powershell
& ".\.venv\Scripts\python.exe" 脚本名.py
```

## 现有重要文件

- `train_pennfudan.py`
- `infer_trained.py`
- `infer_image.py`
- `evaluate_thresholds.py`（如果已经生成）
- `weights/mask_rcnn_pennfudan.pth`
- `data/PennFudanPed/`

## 当前模型

- 模型：TorchVision Mask R-CNN
- 数据集：Penn-Fudan
- 类别数：2
- 类别 0：背景
- 类别 1：行人
- 权重：`weights/mask_rcnn_pennfudan.pth`

## 工作规则

1. 每次任务开始前先检查相关文件和目录。
2. 先说明准备新增、修改哪些文件，再开始修改。
3. 优先复用已有函数，避免复制出多套不一致的模型结构。
4. 路径统一使用 `pathlib.Path`，并基于 `__file__` 计算。
5. 兼容 Windows 和 PowerShell。
6. 运行 Python 时使用指定解释器。
7. 优先使用 CUDA，同时支持 CPU 回退。
8. 不删除、覆盖已有权重。
9. 不随意安装新的大型依赖。
10. 不使用 `wget`。
11. 不大范围重写已经能够工作的代码。
12. 对重要函数添加类型标注和简短中文注释。
13. 修改后必须实际运行验证，不能只生成代码。
14. 检查输出文件是否真实存在、非空并能打开。
15. 遇到错误时先定位原因，做最小修改，不要直接推倒重写。
16. 除非用户明确授权，否则每次只完成当前任务，不要自动进入下一阶段。

## 验证规则

修改代码后必须：

1. 使用指定 Python 解释器实际运行。
2. 检查退出码。
3. 检查输出文件是否真实存在。
4. 检查输出内容不是空文件。
5. 在结束前自行检查代码差异。

## 面向用户的说明

用户是初学者。完成后用中文说明：

- 修改了哪些文件
- 执行了哪些命令
- 输入和输出是什么
- 张量形状是什么（涉及模型、图片、mask 时必须说明）
- 数据流是什么
- 文件保存在哪里
- 如何重新运行
- 验证结果是什么
- 可能出现什么错误
- 还存在哪些问题
