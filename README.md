# FFT-STGAN-IDS 实验工程

本项目根据 `WSTGAN-IDS.docx`、`Framework.png` 和参考目录
`A:\DATAS\rt-iot2022` 中 STGAN-IDS 对 UNSW-NB15、CIC-IDS2017、TON_IoT
三个数据集的处理方式搭建。

核心约束：

- 只修改本项目目录内容。
- Python 环境使用 `D:\new_1\envs\new_conda1\python.exe`。
- 全部模型网络基于 PyTorch。
- 实验过程不需要、也不允许执行 `pip install` 或 `pip uninstall`。
- 原框架中的 DWT/小波分支被替换为 FFT 频带先验分支。

## 方法变化

`Framework.png` 中原始频率分支使用 DWT 得到多尺度小波先验 `{W1,W2,W3,W4}`。
本项目将其替换为 **Learnable FFT Band Prior**：

1. 对输入结构图像执行 `torch.fft.fft2`。
2. 按频率半径构造低频、中频、高频、全频四组频带响应。
3. 通过可学习 1x1/3x3 卷积和 Sigmoid gate 生成每个 encoder stage 的频带先验。
4. 在空间分支、时序 GRU 分支和频带分支间执行融合。
5. 损失和异常分数中增加频带一致性项，替代原小波一致性项。

这个替代分支不依赖任何小波库或新增第三方包，作用上承担多尺度频率上下文、去噪和结构增强。

## 快速运行

先注册本地参考数据集：

```powershell
& 'D:\new_1\envs\new_conda1\python.exe' .\scripts\prepare_datasets.py
```

正式训练：

```powershell
& 'D:\new_1\envs\new_conda1\python.exe' .\train.py --dataset all --epochs 30 --batch-size 256 --image-size 16 --device cuda --lr-policy lambda --lr-decay-start 15 --eval-every 5
```

结果会写入：

- `outputs/<dataset>_<timestamp>/metrics.json`
- `outputs/<dataset>_<timestamp>/scores.csv`
- `outputs/<dataset>_<timestamp>/loss_curve.png`
- `outputs/<dataset>_<timestamp>/roc_curve.png`
- `outputs/<dataset>_<timestamp>/metrics_bar.png`
- `outputs/summary_metrics.csv`

## 数据处理逻辑

参考论文和代码的三个数据集处理方式为：

- 训练集只使用 normal 样本。
- 测试集包含 normal 和 abnormal。
- 数值特征归一化到 `[0,1]`。
- 使用正常训练样本估计特征相关性布局。
- 连续三条网络流量映射到 RGB 三通道，窗口内任意异常则该图像标为 abnormal。

`scripts/prepare_datasets.py` 默认不复制大量 PNG，而是在项目内生成 manifest，记录只读参考数据路径和样本数量。若以后只有 CSV，可用 `scripts/preprocess_csv_to_rgb.py` 在项目内生成同样目录结构的数据。
