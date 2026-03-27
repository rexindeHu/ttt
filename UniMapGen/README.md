# UniMapGen（Qwen3-VL-4B）单文档使用说明

本文档是本仓库唯一文档入口，已合并原先所有 Markdown 内容，并完成文件名简化。
当前仓库定位为：服务器端推理与评估前置验证，不包含训练逻辑。

如果你的目标是“先完成离散 token 前处理，再启动全参训练”，本仓库现在支持该流程，并且前处理阶段不需要模型权重。

## 1. 项目目标与边界

目标：

- 适配 Qwen3-VL-4B。
- 剥离本地训练相关代码。
- 保留离散 token 的数据处理、推理、评估、可视化链路。
- 作为后续 swift 全参数训练前的验证基线。

当前保留能力：

- 离散 token 编码与解码。
- JSONL 数据读取、样本过滤与消息组装。
- 模型推理与预测解析。
- 指标计算（含论文风格指标）与可视化输出。

当前不包含：

- 全参数训练入口。
- LoRA 训练入口。
- 训练专用 dataset/collator。

## 2. 文件名简化结果

本次已将核心文件重命名为更短、更清晰的名称。

- `scripts/eval_stagea_discrete.py` -> `scripts/eval.py`
- `scripts/launch_stagea_discrete_eval.sh` -> `scripts/run_eval.sh`
- `scripts/prepare_tokens.py`（新增）
- `scripts/run_prepare.sh`（新增）
- `stagea_discrete_eval.sbatch` -> `eval.sbatch`
- `unimapgen/discrete_map_token_format.py` -> `unimapgen/token_format.py`
- `unimapgen/paper_metrics.py` -> `unimapgen/metrics.py`
- `unimapgen/data/discrete_stagea_dataset.py` -> `unimapgen/data/dataset.py`
- `unimapgen/models/qwen3_vl_discrete.py` -> `unimapgen/models/qwen3.py`
- `configs/eval.env.example` -> `configs/eval.env`
- `configs/portable_paths.env.example` -> `configs/paths.env`

## 3. 目录结构与文件放置位置

推荐结构：

```text
tzy/
  Qwen3-VL-4B-Instruct/
    config.json
    tokenizer.json
    model.safetensors.index.json
    ...
  UniMapGen/
    README.md
    eval.sbatch
    configs/
      eval.env
      paths.env
    dataset/
      train.jsonl
      val.jsonl               # 可选
      images/
        *.png / *.jpg
    scripts/
      eval.py
      run_eval.sh
    unimapgen/
      token_format.py
      metrics.py
      data/
        dataset.py
      models/
        qwen3.py
    outputs/
    logs/
```

路径约定：

- 模型目录默认：`../Qwen3-VL-4B-Instruct`
- 数据根目录默认：`./dataset`
- 输出目录默认：`./outputs/eval`

## 4. 各文件功能说明

### 4.1 启动层

- `scripts/run_eval.sh`
  - 统一入口脚本。
  - 负责读取环境变量、检查路径、拼接命令并调用 Python 评估程序。

- `scripts/run_prepare.sh`
  - 离散 token 前处理入口。
  - 只读取数据，不加载模型权重。
  - 生成可直接用于训练的数据集产物。

- `eval.sbatch`
  - Slurm 提交脚本。
  - 在服务器集群上直接提交评估任务。

### 4.2 主流程层

- `scripts/eval.py`
  - 端到端执行评估流程：
    1) 解析参数
    2) 载入样本
    3) 调模型生成
    4) 解析预测线条
    5) 计算指标
    6) 输出 JSON 与可视化结果

- `scripts/prepare_tokens.py`
  - 将 assistant 的 `lines` 标注转换为离散 token 文本。
  - 输出训练可消费的 JSONL（system/user/assistant 三段消息）。
  - 同步输出前处理统计摘要。

### 4.3 核心库层

- `unimapgen/token_format.py`
  - 离散 token 格式定义（类别 token、坐标 token、序列结构）。
  - 负责 lines <-> token 文本的双向转换。

- `unimapgen/data/dataset.py`
  - 读取 JSONL。
  - 提取 user/assistant 消息。
  - 构建推理对话模板。
  - 清洗预测/标注线条数据。

- `unimapgen/models/qwen3.py`
  - 加载 Qwen3-VL-4B processor 与模型。
  - 提供统一推理模型加载函数。

- `unimapgen/metrics.py`
  - 计算 IoU、AP、Chamfer 等评估指标。

### 4.4 配置与数据目录

- `configs/paths.env`
  - 路径相关变量样例。

- `configs/eval.env`
  - 评估参数变量样例。

- `dataset/`
  - 存放 `train.jsonl`、`val.jsonl`（可选）和 `images/`。

- `outputs/`
  - 存放评估结果与可视化图。

- `logs/`
  - 存放运行日志。

## 5. 代码执行链路解析

实际调用链路如下：

1. `scripts/run_eval.sh`
2. `scripts/eval.py`
3. `unimapgen/models/qwen3.py`
4. `unimapgen/data/dataset.py`
5. `unimapgen/token_format.py`
6. `unimapgen/metrics.py`

核心流程说明：

- `eval.py` 先读取样本和图像。
- 使用 `dataset.py` 构建多模态对话 prompt。
- `qwen3.py` 加载模型并执行 `generate`。
- `token_format.py` 将模型文本输出解析为结构化线条。
- `metrics.py` 计算样本级和汇总级指标。
- 最后写出 `predictions.jsonl`、`predictions.json`、`summary.json` 和可视化图片。

## 6. 从零到一使用流程

### 步骤 1：准备数据（前处理阶段不需要模型）

在 `dataset/` 下放置：

- `train.jsonl`
- `val.jsonl`（可选）
- `images/`

每条 JSONL 样本至少包含：

- `id`
- `images`（相对 `dataset/` 的图像路径）
- `messages`（至少包含 user 和 assistant，assistant.content 中有 lines）

### 步骤 2：执行离散 token 前处理（无模型权重）

```bash
bash scripts/run_prepare.sh
```

默认产物：

- `outputs/prepare/train_tokens.jsonl`
- `outputs/prepare/prepare_summary.json`

说明：

- `train_tokens.jsonl` 中 assistant 已变成离散 token 文本。
- 默认不写入 `system` 消息（可通过 `INCLUDE_SYSTEM=1` 开启）。
- 默认使用 `TOKEN_SEP=none`，即 token 连续拼接，不用空格分隔，减少无效空白 token 开销。
- 不再输出 `target` 字段，避免数据冗余。
- 该文件用于后续全参训练的数据输入。

### 步骤 3：启动全参训练

本仓库不再内置训练代码；建议把 `outputs/prepare/train_tokens.jsonl` 直接喂给 swift 训练任务。

示例（占位，按你的服务器 swift 命令替换）：

```bash
swift sft \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --dataset /path/to/UniMapGen/outputs/prepare/train_tokens.jsonl \
  --output_dir /path/to/train_output
```

### 步骤 4：训练后评估（需要模型权重）

训练完成后再执行：

```bash
bash scripts/run_eval.sh
```

### 步骤 5：可选集群提交运行

```bash
sbatch eval.sbatch
```

### 步骤 6：配置参数（可选）

可参考：

- `configs/paths.env`
- `configs/eval.env`

关键参数：

- `MODEL_DIR`
- `CHECKPOINT_OR_MODEL`
- `PROCESSOR_PATH`
- `DATASET_ROOT`
- `DATASET_JSONL`
- `OUTPUT_DIR`
- `DEVICE`
- `MAX_SAMPLES`
- `MAX_NEW_TOKENS`
- `SKIP_VIZ`

前处理关键参数补充：

- `CATEGORIES=auto`（默认）: 自动从数据集中提取类别，如 `lane_line`。
- 若手动指定类别，请保证与标注类别一致，否则可能导致输出为空。
- `TOKEN_SEP=none`（默认）: token 紧凑拼接；如需可读性可设为 `space`。
- `INCLUDE_SYSTEM=0`（默认）: 不输出 system 提示词；设为 `1` 可恢复。

关于“去掉括号和逗号是否会歧义”：

- 不会。离散表示依赖的是保留 token 序列（如 `<line><cat_xxx><pts><12><34>...`），
  不是依赖 JSON 里的 `[]`、`,` 字符。
- 解析时使用 `<...>` token 规则做结构恢复，语义边界由控制 token（`<line> <pts> <eol> <eos>`）保证。

评估默认行为：若存在 `dataset/val.jsonl`，优先评估它，否则使用 `dataset/train.jsonl`。

## 7. 输出产物说明

默认输出目录：`outputs/eval`

主要产物：

- `predictions.jsonl`: 逐样本预测记录（便于流式处理）
- `predictions.json`: 完整预测列表
- `summary.json`: 汇总指标
- `viz/*.png`: GT 与预测对比图（若未关闭可视化）

## 8. 与 swift 全参数训练衔接建议

建议在进入 swift 训练前，先固定以下基线：

1. 数据快照版本（JSONL + images）。
2. token 规则（`shared_numbers`、`coord_num_bins=896`）。
3. 一次完整评估结果（`summary.json` + 可视化样本）。
4. 模型与 processor 路径规范。

这样可保证后续训练前后可做稳定回归对比。

## 9. 快速排障

- 模型找不到：检查 `MODEL_DIR` 是否存在且包含 `config.json`。
- 图片找不到：检查 JSONL 里的 `images` 路径是否相对 `dataset/`。
- 输出为空：将 `MAX_SAMPLES=1`，先验证单样本端到端。
- 显存不足：减小 `MAX_NEW_TOKENS`，或切换更大显存设备。

## 10. 当前最简启动命令

```bash
cd UniMapGen
bash scripts/run_prepare.sh
```
