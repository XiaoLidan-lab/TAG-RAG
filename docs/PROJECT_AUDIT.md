# 项目完整性与 GitHub 发布检查

检查日期：2026-10-15。检查对象为本地目录；本次仅新增发布文档、忽略规则及源码压缩包，未修改算法、配置、数据或权重。

## 结论

核心源码基本齐全，可整理为源码归档；当前不能认定为可直接复现的完整发布版本。

- 共 23 个 Python 源文件，均通过 AST 语法解析。
- 静态检查未发现显式引用但缺失的项目内 Python 模块。`ultra` 没有顶层 `__init__.py`，Python 3 可将其作为命名空间包；不能仅据此判断缺文件。
- RSPMM 引用的 C++、CUDA 及本地头文件在目录中存在。
- WebQSP 的 15 个训练分片、9 个测试分片和验证集，以及 MetaQA 的训练、验证、测试目录中，均发现 `kg_data.pt`、`qa_data.pt`、`ent2id.json`、`rel2id.json`、`ent_emb.pt` 和 `raw/kg.txt`。这里只验证存在性，未加载 tensor 或核验内容。
- 当前配置引用的三个历史模型路径在原目录中存在。源码包排除实验输出后，需要另行提供这些产物。
- 原目录未初始化 Git，原先没有 README、依赖清单、忽略规则和 LICENSE。

## 运行阻塞项

1. **WebQSP 的 LLM 输入字段不一致。** `train_llm.py:361`、`:379` 和 `inference.py:145` 构造 `inference_path`，但 `llm.py:96`、`:150` 读取 `samples["ent"]`；`utils.collate_fn` 仅聚合原字段，不负责转换，相关入口会遇到缺少 `ent` 的问题。需按实验意图统一字段，并核对训练与推理提示格式。
2. **默认推理配置混用不同数据集的模型。** `config/inference.yaml:11` 选择 MetaQA，`:35` 使用 MetaQA 检索器，但 `:28` 指向 WebQSP LLM checkpoint。已核对该 checkpoint 旁的 `.hydra/config.yaml`，确为 WebQSP 训练配置。文件存在不代表配套正确。
3. **训练配置与 WebQSP 入口不配套。** `config/train_llm.yaml` 默认 MetaQA；`train_llm.py:342` 需要 `data.raw_graph`，而 MetaQA 加载器不提供该属性。MetaQA 应使用对应入口；WebQSP 应准备独立且匹配的配置。
4. **缺少已验证的环境安装方案。** 当前 Python 环境未安装 torch、PyG、torch-scatter、Hydra、transformers 等核心库，不能执行导入、扩展编译或 GPU 实验。本次从历史日志恢复了 `requirements.reference.txt`；其中 NumPy 1.22.4 与 SciPy 1.13.1 的组合需要处理，CUDA 扩展版本也需与 PyTorch 匹配。

## 复现与发布缺口

- **数据准备流程不完整。** 有转换后的数据与加载器，但未发现 WebQSP 原始数据到当前图分片的完整生成脚本、下载说明或来源记录。`metaqa.py:33` 引用的 `data/MetaQA/qa_dev.txt` 不存在，且 `:26` 在 400 条处截断，模块导入也会直接执行转换。
- **部分缓存重建不稳健。** 两个数据加载器的 `process()` 仅在映射 JSON 缺失时初始化 `ent2id`、`rel2id` 和图局部变量；映射仍在、但图或问答缓存缺失时，后续分支可能使用未赋值变量。`self.ent_emb` 也只在生成实体缓存分支中赋值。不能承诺任意删除部分缓存后可重建。
- **CPU/macOS 路径未完成。** `utils.get_device()` 虽有 CPU 分支，`utils.cuda()` 仍调用 `.cuda()`；LLM 使用 bitsandbytes 4-bit。原环境为 Linux GPU，当前机器没有完成运行验证。
- **算法边界条件需后续验证。** `path_generate.py` 固定取 top-3，少于三个候选实体可能报错；当前 A* 的队列优先级使用累计边权，启发项未进入队列优先级，需核对与实验描述是否一致。
- **第三方授权缺失。** 未发现 LICENSE 或第三方版权声明，`ultra/` 来源需要核实。不能擅自给整个项目指定 MIT 等许可证。

## 体积与隐私检查

按普通文件逻辑大小统计，原目录约 77.66 GB（十进制）：`data/` 约 38.55 GB，`outputs/` 约 39.06 GB，`wandb/` 约 47.31 MB。193 个文件超过 100 MiB，最大单个 checkpoint 约 4.27 GB。源码仓库应排除这些实验产物，另行提供经过整理的数据和权重交付方式。

对拟发布的 Python 源码与 YAML 配置进行了常见 GitHub/Hugging Face token、私钥、密钥赋值和邮箱模式扫描，未命中。这不是完整安全审计。W&B 元数据含邮箱、机器名和本机路径，已整体排除；日志与 checkpoint 未做逐内容安全扫描，未纳入源码包。

## 本次验证范围

完成：目录清点、Python 语法检查、模块与配置引用静态检查、数据关键文件存在性、文件体积统计、源码常见敏感模式扫描、压缩包内容及完整性核验。

未完成：安装依赖、运行模型、CUDA 扩展编译、张量内容校验、全量数据正确性、训练/推理端到端复现、第三方授权核验。历史训练日志只说明曾经运行过，不能证明当前源码就是当时的完整可运行版本。
