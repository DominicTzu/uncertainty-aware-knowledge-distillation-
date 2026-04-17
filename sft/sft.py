import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================
# Config
# =========================
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_JSONL = "/root/data/ultrachat_kd/ultrachat_train_kd.jsonl"
EVAL_JSONL = "/root/data/ultrachat_kd/ultrachat_eval_kd.jsonl"
OUT_DIR = "/root/exp/qwen25_3b_sft_ultrachat"

MAX_LEN = 768

# 调试时可改小；正式跑设为 None
DEBUG_TRAIN_N = 1000 # debug:1000, full:None
DEBUG_EVAL_N = 200  # debug:200, full:None

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training
LR = 2e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 16
NUM_EPOCHS = 1
LOGGING_STEPS = 10
EVAL_STEPS = 200 # debug:50，full:200
SAVE_STEPS = 200 # debug:50，full:200
SAVE_TOTAL_LIMIT = 2

# =========================
# Utils
# =========================
def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def maybe_take(data: List[Dict], n: int):
    if n is None or n >= len(data):
        return data
    return data[:n]

# =========================
# Dataset
# =========================
class ResponseOnlySFTDataset(Dataset):
    """
    每条样本输入:
      {
        "prompt": "...assistant\\n",
        "target": "..."
      }

    训练目标:
      full_text = prompt + target
      labels中 prompt 部分全设为 -100
      仅 target 部分参与 CE loss
    """
    def __init__(self, records: List[Dict], tokenizer, max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        ex = self.records[idx]
        prompt = ex["prompt"]
        target = ex["target"]

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        full_ids = self.tokenizer(
            prompt + target,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # 关键：如果full被截断，labels也要对齐截断
        full_ids = full_ids[:self.max_len]
        attention_mask = [1] * len(full_ids)

        # prompt部分不算loss
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

@dataclass
class SFTCollator:
    tokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        pad_id = self.tokenizer.pad_token_id

        for x in features:
            cur_len = len(x["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch

# =========================
# Main
# =========================
def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print("Loading data...")
    train_data = read_jsonl(TRAIN_JSONL)
    eval_data = read_jsonl(EVAL_JSONL)

    train_data = maybe_take(train_data, DEBUG_TRAIN_N)
    eval_data = maybe_take(eval_data, DEBUG_EVAL_N)

    print(f"train size = {len(train_data)}")
    print(f"eval size  = {len(eval_data)}")

    train_dataset = ResponseOnlySFTDataset(train_data, tokenizer, MAX_LEN)
    eval_dataset = ResponseOnlySFTDataset(eval_data, tokenizer, MAX_LEN)
    collator = SFTCollator(tokenizer=tokenizer)

    sample = train_dataset[0]
    ids = sample["input_ids"]
    labs = sample["labels"]

    active_ids = [tid for tid, lab in zip(ids, labs) if lab != -100]

    print("===== prompt+target decoded =====")
    print(tokenizer.decode(ids))

    print("\n===== loss-active decoded =====")
    print(tokenizer.decode(active_ids))

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        bf16=True,
        report_to="none",

        dataloader_num_workers=2,
        remove_unused_columns=False,

        load_best_model_at_end=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Start training...")
    trainer.train()

    print("Saving final adapter...")
    final_dir = os.path.join(OUT_DIR, "lora_final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Done.")
    print("Final saved to:", final_dir)

if __name__ == "__main__":
    main()