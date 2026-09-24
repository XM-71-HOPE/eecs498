# EECS 498-007 / 598-005 Deep Learning for Computer Vision — 材料包

UMich Justin Johnson 的计算机视觉课（Winter 2022）。**原官网已经下线**：
`https://web.eecs.umich.edu/~justincj/` 整站 404，所有课件/作业链接全部失效。
本目录里的东西是从 Wayback Machine 快照和公开仓库里捞回来、整理好的，离线可用。

整理时间：2026-09-20

---

## 目录结构

```
eecs498/
├── assignments/            作业（主打）
│   ├── A1/                 起点：PyTorch 101 + kNN（骨架为重建，见下文说明）
│   ├── A2/ ... A6/         官方 starter 原样解压（未做题）
│   └── official-zips/      官方 A2~A6.zip 原件
├── reference-solutions/    A1~A6 的 2019 版已解答 notebook（来自公开仓库，仅作参考）
├── notes/                  全课程手写笔记 PDF（95 MB）+ HTML 版
└── archive-pages/          官方页面快照：6 份作业说明 + schedule + syllabus + project
```

---

## 六次作业

| 作业 | 主题 | 关键文件 |
|---|---|---|
| A1 | PyTorch 基础 + kNN 分类器 | `pytorch101.py/.ipynb`、`knn.py/.ipynb` |
| A2 | 线性分类器（SVM / Softmax）+ 两层网络 | `linear_classifier.*`、`two_layer_net.*`、`challenge_problem.ipynb` |
| A3 | 全连接网络 + 卷积网络（手写反向传播、BN、Dropout） | `fully_connected_networks.*`、`convolutional_networks.*` |
| A4 | 目标检测：单阶段 FCOS + 两阶段 Faster R-CNN | `one_stage_detector.*`、`two_stage_detector.*`、`common.py` |
| A5 | 注意力模型：RNN/LSTM 图像描述 + Transformer | `rnn_lstm_captioning.*`、`Transformers.*` |
| A6 | 生成模型：VAE + GAN + 网络可视化 + 风格迁移 | `variational_autoencoders.*`、`generative_adversarial_networks.*`、`network_visualization.*`、`style_transfer.*` |

每个作业目录里都有一个 `eecs598/` 包（课程工具：数据下载、可视化、提交脚本），
原样保留，不要删。

### 关于 A1 的说明

官方的 `A1.zip` **没有被任何快照存档**（A2~A6 都有，唯独缺它）。
现在的 `assignments/A1/` 是我从一份结构一致的公开作业里，把每道题答案区
（`# Replace "pass" statement with your code` 到 `# END OF YOUR CODE` 之间）
清空并还原成 `pass` 得到的纯净骨架，函数名、docstring、TODO 提示与官方一致；
`eecs598/` 包取自同一来源。notebook 输出也已清空。
如果你更想要一份带解答的对照，去 `reference-solutions/A1/`。

---

## 怎么跑

### 1. 环境

用 uv 建环境，依赖钉在 `requirements*.txt` 里。**分两条命令：先公共依赖，再 torch。**

```bash
uv venv .venv
source .venv/bin/activate

# 公共依赖（两台机器一样）
uv pip install -r requirements.txt

# 然后选一条（索引地址写在文件里了，不用带参数）：
uv pip install -r requirements-cpu.txt    # 笔记本（没有 N 卡）
uv pip install -r requirements-cuda.txt   # 台式机（RTX 5060 Ti）
```

**为什么要分两条：** PyTorch 自己的索引里也有 tqdm、numpy 这些包，但版本是旧的
（tqdm 只到 4.66.5）。uv 默认只认"第一个含该包的索引"，而且 `--extra-index-url`
会遮蔽默认索引，混在一条命令里会报 unsatisfiable。所以两条命令彻底分开，各自用自己的索引。

版本钉死了：torch 2.14.0 / torchvision 0.29.0，两台机器一致，只是构建不同（cpu / cu130）。
CUDA 版还会拉一批 `nvidia-*-cu13` 和 triton，下载量大概 2~3GB，预留点时间。

装完验证：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

笔记本上 `cuda.is_available()` 是 `False`，台式机上应该是 `True`。

下载慢就挂代理：`export HTTPS_PROXY=http://127.0.0.1:7897`。

> 注意：uv 建的 venv 默认不带 pip。要么一律用 `uv pip install`，要么 `uv venv --seed`。
> 在 venv 里敲裸 `pip` 会落到系统的 `/usr/bin/pip`，撞上 PEP 668 报錯。

### 2. notebook 已经改好了

原版 notebook 是给 Google Colab 写的，开头几句 `drive.mount` 在本地会直接报错。
**本包里的 15 个 notebook 已经改成本地 / Colab 双通道**：开头那段会自动判断运行环境，
本地跑就用 notebook 所在目录作为 `GOOGLE_DRIVE_PATH`，在 Colab 上则照旧挂 Drive。
所以你不用手改任何代码，直接依次 Shift+Enter 就行。

### 3. 工作循环

每次作业的流程是一样的：

1. 在 `.py` 里实现函数（例如 A1 的 `knn.py`、A3 的 `convolutional_networks.py`）。
   notebook 会 `autoreload`，改完 `.py` 不用重启内核。
2. 回 notebook 里跑对应的 cell。每个实现后面都跟着自检 cell，里面写了**预期数值**（索引、误差、准确率的量级）。
   对上了就说明做对了。
3. 卡住就去看讲义里对应的那一讲，或者 `reference-solutions/` 里别人的实现（建议自己写完再看）。
4. 最后的 `make_aX_submission(...)` 那个 cell 是交作业用的，校外没有 Autograder，**跳过**。

### 4. 关于 GPU

两台机器的分工：

| 机器 | 显卡 | torch | 适合做 |
|---|---|---|---|
| 笔记本 | 无 | 2.14.0+cpu | A1、A2，以及读代码、调试 |
| 台式机 | RTX 5060 Ti 16GB | 2.14.0+cu130 | A3 ~ A6 的训练 |

- **A1、A2 在笔记本上跑完全没问题**，CPU 够用。
- **A3 往后搬到台式机**。有显卡之后，CIFAR-10 训练、目标检测、风格迁移都不再是负担，
  也不用再去蹭 Colab 的免费 T4。
- 笔记本上硬要跑 A3 之后的：notebook 里有些地方写死了 `device='cuda'`，
  纯 CPU 机器上得把 `'cuda'` 改成 `'cpu'`，而且会明显慢。
- 台式机驱动 610.43.02，显卡是 Blackwell（compute capability 12.0），
  必须用 CUDA 12.8 以上的构建，所以选 cu130（cu132 里也有同一版本，cu134 那个索引反而是空的）。

### 5. 数据

数据集由 `eecs598/data.py` / `a4_helper.py` 自动下载（CIFAR-10、VOC2007 等），第一次跑会慢。
下载地址是国外的，慢的话同样挂代理：`export https_proxy=http://127.0.0.1:7897`。

---

## 视频

- B 站中英字幕（2019 版全 22 讲，ShowMeAI 整理）：`https://www.bilibili.com/video/BV13P4y1t7gM`
- 另一份 2019 中英字幕（23 个视频）：`https://www.bilibili.com/video/BV1jA4m1P7GN`
- 2020 版中英字幕：`https://www.bilibili.com/video/BV1hhybYnE6n`
- 官方 YouTube 播放列表（Fall 2019）：`https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r`

注：2019 与 2022 的讲次顺序略有差别，但作业主题对应关系基本一致。

---

## 课件（slides）

官网的 slide PDF 也没有活的链接了，但 Wayback 里存着。模板（把 `NN` 换成 01~25）：

```
https://web.archive.org/web/2024id_/https://web.eecs.umich.edu/~justincj/slides/eecs498/WI2022/598_WI2022_lectureNN.pdf
```

每讲对应主题见 `archive-pages/schedule.html`（本地打开即可，里面是完整的讲义清单和推荐阅读）。
需要的话我可以把 25 份 slide 全部下载到本地（总共几百 MB，单文件最大 ~80 MB）。

---

## 原站快照

- 课程主页（WI2022）：`https://web.archive.org/web/20250623085956/https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/`
- 作业说明页的离线副本在 `archive-pages/`（改写自存档 HTML，链接仍指向 Wayback）。

---

## 参考仓库

- `frankly6/eecs498-19fa-notes-and-assignments` — 本包 notes 和 reference-solutions 的来源
- `grygry12345/DLCV-EECS498` — A1 骨架重建的来源
- `Anwarvic/DL-for-Computer-Vision-2020-Michigan-Course` — FA2020 版作业解答
- `iMeleon/EECS-498-007-598-005-solutions` — 2019/2020 版作业解答
- `Michael-Jetson/ML_DL_CV_with_pytorch` — 二三十万字中文笔记
