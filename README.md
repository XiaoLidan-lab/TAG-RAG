# 知识图谱问答实验代码

基于图神经网络的实体检索，结合 Llama 2 与 QLoRA 进行问答，包含 WebQSP 和 MetaQA 实验入口。

当前版本是研究代码归档。源码已做静态检查，尚未完成在全新环境中的端到端复现；已知运行问题见 [检查报告](docs/PROJECT_AUDIT.md)。

## 文件结构

| 文件或目录 | 用途 |
| --- | --- |
| `fine_tuning.py` | 训练 GNN 检索器 |
| `train_llm.py` / `inference.py` | WebQSP 路径输入实验入口，存在字段不一致问题 |
| `train_llm_metaqa.py` / `inference_metaqa.py` | MetaQA 实体输入实验入口 |
| `models.py` / `losses.py` | 检索模型与损失函数 |
| `llm.py` | Llama 2 加载、QLoRA 和生成 |
| `path_generate.py` | 图路径生成 |
| `my_datasets/` | 数据集读取与嵌入预处理 |
| `ultra/` | 图网络及 RSPMM C++/CUDA 扩展源码 |
| `config/` | Hydra 实验配置 |
| `docs/requirements.reference.txt` | 历史训练环境依赖快照，未经重新安装验证 |

## 运行环境与依赖

历史日志记录了 Linux、Python 3.9.21、PyTorch 2.1.0+cu118 和 NVIDIA GPU。代码使用 CUDA 张量迁移及 bitsandbytes 4-bit 量化，目前不能承诺在 macOS 或纯 CPU 环境运行。

主要依赖包括 PyTorch、torch-geometric、torch-scatter、Hydra、OmegaConf、transformers、sentence-transformers、datasets、peft、accelerate、bitsandbytes、sentencepiece、NumPy、pandas、SciPy、NetworkX、CVXPY、tqdm 和 wandb。RSPMM 会在导入时编译，需要可用的 C++ 编译器；CUDA 构建还需要匹配的 CUDA 编译工具链。

`docs/requirements.reference.txt` 来自历史 W&B 记录，不是已经验证可安装的依赖锁文件。其中 NumPy 1.22.4 与 SciPy 1.13.1 的版本组合需要重新核对，PyTorch 和 torch-scatter 的 CUDA 构建也需要匹配。不要把该快照视为一条命令即可完成安装的保证。

外部模型为 `sentence-transformers/all-mpnet-base-v2` 与 `meta-llama/Llama-2-7b-chat-hf`；运行前需自行准备模型访问权限或本地路径。

## 数据与模型文件

源码包不包含 `data/`、`outputs/` 和 `wandb/`。原始工作目录中的这些文件仍然保留。仓库尚未提供经过验证的数据下载、分片和预处理完整流程，也未提供模型权重下载地址。

数据加载器使用以下布局：

```text
data/<WebQSP 或 MetaQA>/<分片名>/
  raw/
    kg.txt
    train.json 或 valid.json 或 test.json
    sample.json                 # WebQSP 路径实验还需要此文件
  processed/
    kg_data.pt
    qa_data.pt
    ent2id.json
    rel2id.json
    ent_emb.pt
```

WebQSP 图三元组按 `&&` 分隔，MetaQA 按 `|` 分隔。问答 JSON 使用 `id`、`question`、`answer`、`q_entity`、`a_entity` 等字段。WebQSP 的 `sample.json` 还提供 `graph`。

检索器加载目录需要 `config.json` 和 `model.pth`。LLM checkpoint 路径由 `llm.checkpoint` 指定。配置中的日期目录是历史实验产物路径，源码包中不存在；应替换为配套模型的位置。

## 实验入口

完成依赖、数据、模型准备，并处理检查报告中的阻塞项后，从项目根目录执行对应入口：

```bash
# 默认检索器配置为 WebQSP
WANDB_MODE=disabled python fine_tuning.py

# 默认 LLM 训练配置为 MetaQA
python train_llm_metaqa.py graph_retriever.model_path=/path/to/metaqa/pretrained

# 使用同一数据集训练的 LLM checkpoint
python inference_metaqa.py graph_retriever.model_path=/path/to/metaqa/pretrained llm.checkpoint=/path/to/metaqa/llm_model_best.pth
```

这些命令说明参数形式，本次没有执行训练或推理。WebQSP 的 LLM 训练和推理需先统一 `inference_path` 与 `ent` 字段，并切换为 WebQSP 配置。`metaqa.py` 是独立转换脚本，当前输入路径未提供且有 400 条截断，不能作为完整数据准备入口直接运行。

## 发布说明

`.gitignore` 排除了本地数据、模型、日志、缓存和凭据文件。`github-source.zip` 是仅包含源码、配置和文档的归档；上传仓库时应解压后提交其中的文件。

项目暂未添加许可证。公开发布前，需要确认自有代码授权，以及 `ultra/` 等第三方代码来源和所需许可证、版权声明；数据和模型也应遵守各自的使用条款。
