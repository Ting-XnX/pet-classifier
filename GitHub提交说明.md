# 我的仓库与复现说明

王诗雯｜W125301282

我的仓库：https://github.com/Ting-XnX/pet-classifier

我已上传完整模块代码、README、技术报告、图表、两组训练日志和评估记录，材料按实际目录组织，并保留逐项提交记录。

## 我的提交材料

- `report/【考核】王诗雯_W125301282_宠物分类.pdf`：三页正式技术报告。
- `runs/baseline/tensorboard/`：基线完整训练日志。
- `runs/label_smoothing/tensorboard/`：标签平滑完整训练日志。
- `runs/`：固定划分、训练配置、CSV、预测和评估结果。
- `答辩准备.md`：实验方法与关键代码原理。

## 我的后续复现

```bash
git clone https://github.com/Ting-XnX/pet-classifier.git
cd pet-classifier
```

我按照 README 安装依赖，并先将仓库中的 `runs` 改名为 `runs_submitted`，保存原始提交结果后再运行 `python run_all.py`。

本地交付包保留了训练所得权重和独立的实验 Git 历史。后续维护以新克隆的远程仓库为起点，不将本地独立历史强制推送覆盖远程仓库。
