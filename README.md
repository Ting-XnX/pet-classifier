# Oxford-IIIT Pet 细粒度分类

王诗雯｜W125301282

我使用 ImageNet 预训练 ResNet-18 完成 37 类宠物分类，并比较普通交叉熵与 Label Smoothing（0.1）。两个实验共享数据划分、初始化种子、增强、优化器和训练轮数，只改变损失函数中的平滑系数。

## 我的实验结果

我的仓库：https://github.com/Ting-XnX/pet-classifier

我在相同划分和种子下完成两组各 10 轮训练，按验证 Top-1 选择模型后统一评估 1,103 张测试图像。

| 配置 | 最佳验证 Top-1 | 测试 Top-1 | 测试 Top-5 | Macro-F1 | 最佳轮次 |
| --- | --- | --- | --- | --- | --- |
| Baseline | 92.29% | 90.21% | 99.73% | 0.9006 | 2 |
| Label Smoothing 0.1 | 93.10% | 92.11% | 98.82% | 0.9204 | 9 |

本次平滑组的测试 Top-1 提高 1.90 个百分点，Macro-F1 提高 0.0198，但 Top-5 下降。我没有将单次实验表述为统计显著提升。三页报告见 [技术报告](report/【考核】王诗雯_W125301282_宠物分类.pdf)，原始 CSV、TensorBoard 和评估文件见 [实验记录](runs/)。

## 复现

推荐 Python 3.12 和支持 CUDA 的 GPU。Windows 的 DataLoader 入口已有主进程保护。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python run_all.py
```

首次运行会下载数据集及预训练权重。没有 NVIDIA GPU 时可使用 CPU 版 PyTorch，但训练耗时会增加。报告由真实生成的 `history.csv` 和 `evaluation/results.json` 自动汇总，不填入预设成绩。

已有提交日志保存在 `runs/` 时，为防止覆盖，复现前将该目录改名为 `runs_submitted/`。新运行会重新生成 `runs/`。

```bash
tensorboard --logdir runs
```

分步运行：

```bash
python train.py --output runs/baseline --label_smoothing 0
python train.py --output runs/label_smoothing --label_smoothing 0.1
python evaluate.py --checkpoint runs/baseline/best_model.pth
python evaluate.py --checkpoint runs/label_smoothing/best_model.pth
python make_report.py
```

## 实验规范

- 我合并 torchvision 的 `trainval` 与 `test` 后进行分层随机划分，种子为 42；划分索引固定存储在 `runs/split.json`。本结果不与官方划分排行榜直接比较。
- 训练变换为 Resize(256)、RandomCrop(224)、RandomHorizontalFlip、ImageNet Normalize；验证和测试使用 CenterCrop，保持输入确定。
- 两组均全参数微调 10 轮，batch size 32，AdamW，学习率 0.0001，weight decay 0.01，无学习率调度。
- 我仅用验证 Top-1 选择最佳模型，相同准确率时保留较早轮次。全部训练完成后才评估测试集，不据此再调参。
- 平滑组的训练目标损失与普通交叉熵不是同一量。我额外记录 `train_ce`，两组 `val_loss` 均为普通交叉熵。
- Top-1/Top-5 使用百分比，Macro-F1 使用 0～1，固定纳入全部 37 类。
- 单个随机种子的两组实验只支持本次对照观察，不能据此宣称统计显著。

## 文件说明

```text
data/dataset.py          数据加载、固定分层划分与独立变换
models/model.py          预训练 ResNet-18 与 37 类分类头
utils/metrics.py         统一评估与混淆矩阵
utils/gradcam.py         对预测类别计算 Grad-CAM
train.py                训练、CSV/TensorBoard、最佳模型保存
evaluate.py             独立测试、逐样本预测及正确/错误案例
run_all.py              两组训练、评估、报告的一键入口
make_report.py          根据实际结果制作曲线和三页 PDF
runs/                   配置、固定划分、训练日志和评估结果
report/                 报告与可视化
```

Grad-CAM 选取测试顺序中的第一个正确样本和第一个错误样本，目标为模型预测类别，防止根据热力图美观程度挑选案例。热力图是模型敏感区域的定性提示，不构成因果解释。

数据、缓存和模型权重不纳入 Git；本地交付包可保留训练所得权重。数据集使用条件参见原始发布页。

## 参考资料

1. [Oxford-IIIT Pet 原始数据集](https://www.robots.ox.ac.uk/~vgg/data/pets/)
2. [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
3. [Torchvision OxfordIIITPet](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html)

## 工具使用说明

我使用 AI 工具辅助编写模块、检查实验流程和整理报告。实验指标来源于训练日志与独立评估文件；我在答辩中需要解释数据划分、损失函数、梯度清零及 Grad-CAM 的计算过程。

## Kaggle

我也提供 `Kaggle启动.ipynb`。将 `pet_project_code.zip` 添加为私有输入数据集，开启 GPU 和 Internet 后运行全部单元格，结果可下载为 ZIP。Notebook 使用 Kaggle 自带的 GPU 版 PyTorch，不重新安装显卡运行库；跨平台结果可能有细小数值差异。
