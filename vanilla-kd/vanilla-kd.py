import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer import logger

# =========================
# Config
# =========================
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TEACHER_NAME = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_JSONL = "/root/data/ultrachat_kd/ultrachat_train_kd.jsonl"
EVAL_JSONL = "/root/data/ultrachat_kd/ultrachat_eval_kd.jsonl"
OUT_DIR = "/root/exp/qwen25_3b_vanilla_kd_ultrachat"

MAX_LEN = 768

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
EVAL_STEPS = 50 # debug: 50, full: 200
SAVE_STEPS = 50 # debug: 50, full: 200
SAVE_TOTAL_LIMIT = 2

# KD
KD_LAMBDA = 0.5
KD_TEMPERATURE = 2.0

# Optional: teacher 低显存加载
USE_TEACHER_4BIT = False

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

        full_ids = full_ids[:self.max_len]
        attention_mask = [1] * len(full_ids)

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
    tokenizer: object

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for x in features:
            cur_len = len(x["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# =========================
# KD Trainer
# =========================
class VanillaKDTrainer(Trainer):
    def __init__(self, teacher_model=None, kd_lambda=0.5, kd_temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # student
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        student_logits = student_outputs.logits  # [B, L, V]

        # teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            teacher_logits = teacher_outputs.logits  # [B, L, V]

        # causal LM shift
        shift_student_logits = student_logits[:, :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # only target positions
        active_mask = shift_labels.ne(-100)  # [B, L-1]

        # CE
        ce_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # KD on active positions only
        T = self.kd_temperature

        active_student = shift_student_logits[active_mask]   # [N, V]
        active_teacher = shift_teacher_logits[active_mask]   # [N, V]

        if active_student.numel() == 0:
            kd_loss = torch.tensor(0.0, device=ce_loss.device)
        else:
            log_p_s = F.log_softmax(active_student / T, dim=-1)
            p_t = F.softmax(active_teacher / T, dim=-1)

            kd_loss = F.kl_div(
                log_p_s,
                p_t,
                reduction="batchmean",
            ) * (T ** 2)

        loss = (1.0 - self.kd_lambda) * ce_loss + self.kd_lambda * kd_loss

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            logger.info(
                f"step={self.state.global_step} "
                f"loss={loss.item():.4f} "
                f"ce={ce_loss.item():.4f} "
                f"kd={kd_loss.item():.4f}"
            )

        return (loss, student_outputs) if return_outputs else loss

# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_NAME,
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

    print("Loading student...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    student_model.config.use_cache = False

    print("Applying LoRA to student...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    student_model = get_peft_model(student_model, peft_config)
    student_model.print_trainable_parameters()

    print("Loading teacher...")
    # 先给最稳的非4bit版本；显存不够再换4bit
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    teacher_model.config.use_cache = False
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        #overwrite_output_dir=True,

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

        bf16=False,
        fp16=True,
        report_to="none",

        dataloader_num_workers=2,
        remove_unused_columns=False,

        load_best_model_at_end=False,
        seed=SEED,
    )

    trainer = VanillaKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        kd_lambda=KD_LAMBDA,
        kd_temperature=KD_TEMPERATURE,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        #tokenizer=tokenizer,
    )

    print("Start vanilla KD training...")
    trainer.train()

    print("Saving final adapter...")
    final_dir = os.path.join(OUT_DIR, "lora_final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Done.")
    print("Final saved to:", final_dir)

if __name__ == "__main__":
    main()