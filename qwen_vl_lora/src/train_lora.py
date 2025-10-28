import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """解析命令行参数，便于在 Azure ML 作业中灵活传参。"""
    parser = argparse.ArgumentParser(description="LoRA finetuning for Qwen-VL demo 03")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--train-file", type=str, default="train.jsonl")
    parser.add_argument("--validation-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, default="epoch")
    parser.add_argument("--eval-strategy", type=str, default="epoch")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--report-to", type=str, default="none")
    return parser.parse_args()

def load_records(dataset_dir: str, file_name: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，并返回样本列表。"""
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info("Loaded %d samples from %s", len(records), file_path)
    return records

def resolve_image_path(dataset_dir: str, image_path: str) -> str:
    """统一处理相对路径，便于在 Azure ML 计算节点上访问图片。"""
    return image_path if os.path.isabs(image_path) else os.path.join(dataset_dir, image_path)

@dataclass
class QwenRecord:
    """用 dataclass 存储单条样本，提升可读性。"""
    image: str
    question: str
    answer: str
    system: Optional[str] = None

class QwenVLDataset(Dataset):
    """将原始 JSON 样本转换为模型可直接使用的提示格式。"""
    def __init__(self, records: List[Dict[str, Any]], dataset_dir: str, processor: AutoProcessor):
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.records: List[QwenRecord] = [
            QwenRecord(
                image=resolve_image_path(dataset_dir, item["image"]),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                system=item.get("system")
            )
            for item in records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        if not os.path.exists(record.image):
            raise FileNotFoundError(f"Image not found: {record.image}")
        image = Image.open(record.image).convert("RGB")
        messages: List[Dict[str, Any]] = []
        if record.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": record.system}]})
        user_content: List[Dict[str, Any]] = [{"type": "image", "image": image}]
        if record.question:
            user_content.append({"type": "text", "text": record.question})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": record.answer}]})
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"prompt": prompt, "image": image}

class QwenDataCollator:
    """自定义 collator，将批次中的文本和图片一起编码为张量。"""
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [feature["prompt"] for feature in features]
        images = [feature["image"] for feature in features]
        batch = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

def create_model(args: argparse.Namespace) -> AutoModelForVision2Seq:
    """加载基础模型并应用 LoRA 配置，同时启用 4bit 量化以节省显存。"""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=args.trust_remote_code,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    target_modules = [module.strip() for module in args.target_modules.split(",") if module]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    train_records = load_records(args.dataset_dir, args.train_file)
    train_dataset = QwenVLDataset(train_records, args.dataset_dir, processor)
    eval_dataset = None
    if args.validation_file:
        eval_records = load_records(args.dataset_dir, args.validation_file)
        eval_dataset = QwenVLDataset(eval_records, args.dataset_dir, processor)
    model = create_model(args)
    collator = QwenDataCollator(processor)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="no" if eval_dataset is None else args.eval_strategy,
        bf16=args.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to=[args.report_to] if args.report_to and args.report_to != "none" else [],
        run_name=os.getenv("AML_RUN_ID", "qwen-vl-lora"),
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator
    )
    trainer.train()
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(os.path.join(args.output_dir, "processor"))
    logger.info("Training completed. Artifacts saved to %s", args.output_dir)

if __name__ == "__main__":
    main()
