# Qwen-VL LoRA 微调示例

本仓库提供一个面向 Qwen2-VL 系列多模态模型的 LoRA 微调脚本，兼顾本地实验和 Azure Machine Learning（下文简称 **Azure ML**）上的批量训练需求。核心脚本 `qwen_vl_lora/src/train_lora.py` 基于 Hugging Face Transformers、PEFT 与 Accelerate，支持 4bit 量化加载、可选的验证集评估、W&B 记录等能力。

- 🔧 **快速上手**：命令行参数化设计，方便在不同环境中重复实验。
- 🖼️ **多模态数据**：支持图文混合输入，自动处理 system/user/assistant 多轮对话格式。
- 💾 **轻量化训练**：默认启用 LoRA + 4bit 量化，显存占用友好。
- ☁️ **可扩展至 Azure ML**：配套 `aml/` 目录中的 notebook 与环境定义，便于在云端算力上运行。

## 目录结构

```
.
├── aml/                         # Azure ML 相关 notebook 与示意图
├── dataset/                     # 示例数据或数据占位（需自行准备 JSONL）
├── qwen_vl_lora/
│   ├── env/conda.yaml           # Azure ML 使用的 Conda 环境定义
│   └── src/train_lora.py        # LoRA 训练入口
├── vm/
│   ├── images/                  # Azure VM 指南使用的截图
│   └── workshop_finetune_qwen3_vl.ipynb  # Azure VM 一体化调试与训练手册
├── requirements.txt             # 完整依赖列表（与 pyproject 同步）
├── pyproject.toml               # 项目元数据，支持 uv/pip 安装
└── qwen_vl_lora.ipynb           # 本地实验 notebook 示例
```

## 环境准备

1. **Python 版本**：建议 Python 3.10。
2. **创建虚拟环境**（任选其一）：
   - 使用 `uv`：
     ```bash
     uv venv
     uv pip install -r requirements.txt
     ```
   - 使用 `pip`：
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```
   - 在 Azure ML 中，可直接引用 `qwen_vl_lora/env/conda.yaml` 构建环境。
3. **可选依赖**：在 GPU 资源允许的情况下，`bitsandbytes` 会自动启用 4bit 量化；若在 macOS 上训练，可移除该依赖或改用 `load_in_8bit=False`。

## 数据准备

训练脚本期望读取一个或多个 JSONL 文件，每行是一条图文样本，字段说明如下：

- `image`：图片相对路径或绝对路径，可放在 `dataset_dir` 目录下。
- `question`：用户提问或指令文本，可为空。
- `answer`：期望模型输出的文本。
- `system`：可选，提供额外的系统提示。

示例（保存为 `train.jsonl`）：

```json
{"image": "images/cat.jpg", "question": "这只猫在做什么？", "answer": "它正趴在床上休息。"}
{"image": "images/dog.jpg", "system": "你是一名风景摄影解说员。", "question": "描述画面。", "answer": "画面里是一只在草地上奔跑的狗。"}
```

如果需要验证集，可在相同目录中提供 `validation.jsonl`，并在运行参数中指定 `--validation-file validation.jsonl`。

## 本地运行

以下命令展示了在本地进行 LoRA 微调的基本流程，输出将保存在 `./outputs/run-001`：

```bash
python qwen_vl_lora/src/train_lora.py \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --dataset-dir ./dataset \
  --train-file train.jsonl \
  --output-dir ./outputs/run-001 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --report-to none
```

常用参数说明：

- `--target-modules`：LoRA 注入的模块列表，默认覆盖注意力与 MLP 关键层。
- `--bf16`：默认开启（若 GPU 支持），可通过 `--bf16`/`--no-bf16` 控制。
- `--report-to`：设置为 `wandb` 即可启用 Weights & Biases 记录。
- `--save-strategy` / `--eval-strategy`：控制保存与评估频率，默认以 epoch 为粒度。

训练结束后，模型权重与处理器将保存在 `output_dir` 下，分别对应 `adapter_config.json`、`adapter_model.bin` 以及 `processor/` 目录，可直接用于推理或继续训练。

## 在 Azure VM 上运行

对于倾向使用自建 Azure GPU 虚拟机的场景，仓库提供了 `vm/workshop_finetune_qwen3_vl.ipynb`。该 notebook 覆盖从创建 Standard_NVads_A10_v5（双 A10）GPU VM、配置网络/磁盘与禁用 Secure Boot，到安装 NVIDIA GRID 驱动、CUDA 11.8 工具链、Docker 以及 LLaMA-Factory 的完整流程。

运行步骤建议：

1. 按 notebook 提供的截图和说明，在 Azure 门户中完成 VM、虚拟网络与磁盘配置，并通过 Cloud Shell 或本地 Azure CLI 执行必要的准备命令。
2. 使用 SSH 登录新建 VM，依次执行 notebook 中的系统更新、驱动安装和 CUDA 环境配置指令，验证 `nvidia-smi` 与 `nvcc --version` 正常。
3. 安装项目依赖（notebook 默认使用 LLaMA-Factory，可根据需求切换到本仓库的 `train_lora.py`）并将数据集上传到 VM。
4. 在 Jupyter Lab 中打开 notebook，按照分节内容完成数据准备、LoRA 配置与训练监控；若希望复用脚本，可直接运行 `python qwen_vl_lora/src/train_lora.py ...`。
5. 训练完成后，将生成的 LoRA 适配器与处理器打包至存储账号或下载到本地，便于后续部署。

## 在 Azure ML 上运行

1. 参考 `aml/workshop_qwen_vl_aml.ipynb` 中的步骤，创建或重用 Azure ML 工作区与计算资源。
2. 上传数据集至 Azure ML（例如使用 `Dataset.File.from_files` 或将 JSONL/图片打包为数据资产）。
3. 构建环境时指向 `qwen_vl_lora/env/conda.yaml`，或在 `requirements-notebook.txt` 基础上自定义镜像。
4. 提交命令作业时，将数据与输出目录挂载到容器中，命令与本地基本一致，例如：
   ```bash
   python qwen_vl_lora/src/train_lora.py --dataset-dir /mnt/data --train-file train.jsonl --output-dir ./outputs
   ```
5. 作业完成后，LoRA 适配器与处理器会保存在 Azure ML 的输出目录，可作为模型资产注册。

## 许可证

本仓库未显式提供许可证，默认遵循作者保留所有权利。如需在商业环境中使用，请先与仓库作者确认。
