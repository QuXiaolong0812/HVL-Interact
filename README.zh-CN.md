# HVL-Interact 使用说明

模型名称 **HVL-Interact** 对应论文中的层次化视觉—语言对齐与双人交互识别：
HVL 取自 Hierarchical Vision–Language，Interact 突出 Interaction Action Recognition。

本项目实现论文 *Hierarchical Vision–Language Alignment for Dual-Person Interaction Action
Recognition* 的数学方法，包括双人骨骼令牌、骨骼渲染、冻结 CLIP、三级文本描述、类别原型对齐和标签无关融合。

主入口为 `hvl-interact`，也可使用 `python -m hvl_interact`。建议在 Linux、Python 3.10–3.12 环境运行。
完整命令见 [英文说明](README.md)。

## 安装与文件组织

首先安装与驱动匹配的 PyTorch，再执行 `python -m pip install -e '.[dev]'`。
原始骨骼文件放在 `data/raw/nturgb+d_skeletons/`，官方缺失骨骼名单放在
`data/raw/NTU_RGBD120_samples_with_missing_skeletons.txt`。
NTU60 与 NTU120 新增部分的骨骼文件均应提供，程序递归搜索 `.skeleton`。

CLIP 使用 Hugging Face Transformers 格式的 `openai/clip-vit-base-patch16`，目录为
`weights/clip-vit-base-patch16/`。下载完整权重、配置、图像处理与分词器文件。程序仅从本地加载。
训练产生的项目权重在 `runs/实验名/`，单独整理的权重可放在 `weights/checkpoints/`。
这些大文件不进入 Git 源码管理。

依次执行 `prepare`、`split`、`doctor`、`train`、`evaluate`。`prepare` 会输出样本索引、质量排除记录和文件哈希。
`split` 会检查全部类别覆盖情况，并拒绝重复、遗漏或跨越官方协议边界的样本。

## 实验设置

全监督使用全部 26 类，配置分别为 `configs/supervised_xsub.yaml` 与
`configs/supervised_xset.yaml`。X-Sub 使用官方 53 个训练主体，X-Set 使用偶数 setup 训练、奇数 setup 测试。
验证集只从官方训练部分建立。同一 setup、performer、replication 的所有相机样本保持在同一分区。

零样本使用 `configs/zeroshot.yaml`，运行时必须明确提供 `--protocol` 和划分文件。
生成划分时必须通过 `--unseen` 指定六个不同的官方交互类别 ID，程序不会自动抽取。
未见类的官方训练样本不会进入训练或验证。零样本模型独立初始化，禁止直接沿用已在全部 26 类上训练的骨骼权重。
只用已见类验证集选择最佳 checkpoint，再在官方测试部分的六个未见类上进行语义匹配。

`train.batch_size` 是每进程批大小。两卡、每卡 16、累积 1 次时，全局批大小为 32。
单卡可把累积次数设为 2。显存不足时先减小每卡批大小，并相应增加累积次数。
配置中的绘图帧数、骨骼层数和增强幅度均可追踪，不会由目标准确率反推。

## 恢复与结果

使用 `--resume runs/实验名/last.pt` 进行 epoch 边界恢复。配置、数据、描述库、CLIP 文件、进程数和软硬件环境应与原任务一致。
`--stop-after-epoch` 用于分段运行，不改变原定总 epoch 数的余弦学习率计划。

预测文件逐条记录样本 ID、官方真值 ID、预测 ID、候选类别与分数。
`metrics.json` 同时给出样本准确率、宏平均准确率、各类样本数与输入哈希。
`confusion.csv` 保存整数计数。项目权重不重复保存冻结 CLIP，加载时会校验外部编码器指纹。

`predict` 只需要含有 `coordinates` 和 `valid` 的单个 NPZ，不需要真实标签。
其中形状分别为 `[2,T,25,3]` 和 `[2,T,25]`。预测使用 checkpoint 内保存的预处理设置。

## 方法与实验的对应关系

论文没有完全指定的细节，如骨骼主干结构、身体组边界、渲染数量、增强幅度和三级权重，已给出明确实现并在
[方法映射文档](docs/method_mapping.md) 中说明。它们属于当前实现的选择，不代表已恢复历史实验配置。
项目不会写入预设准确率或参数量。完整 NTU 训练结果需要在准备好数据和预训练编码器后实际运行得到。

测试覆盖几何关系、缺失值、三级对齐、标签无关推理、冻结编码器、断点恢复及双进程评估。
运行 `pytest -q`、`ruff check src tests` 和 `python -m build` 可检查代码与分发包。

