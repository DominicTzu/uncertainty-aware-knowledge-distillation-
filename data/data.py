import os
import json
import random
import hashlib
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# =========================
# Config
# =========================
OUT_DIR = "/root/data/ultrachat_kd"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 0
random.seed(SEED)

# Dataset
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
TRAIN_SPLIT = "train_sft"
TEST_SPLIT = "test_sft"

# Tokenizer:
# 用 student tokenizer 做长度过滤最合理，因为最终训练主要受 student 侧约束
TOKENIZER_NAME = "Qwen/Qwen2.5-3B-Instruct"

# 保持原数据量不变
MAX_TRAIN_SAMPLES_BEFORE_FILTER = 100000
MAX_EVAL_SAMPLES_BEFORE_FILTER = 10000

# Length filtering
MIN_RESPONSE_TOKENS = 16
MAX_RESPONSE_TOKENS = 256
MAX_PROMPT_TOKENS = 512
MAX_TOTAL_TOKENS = 768

# 输出保留多少
MAX_TRAIN_KEEP = 50000
MAX_EVAL_KEEP = 2000

# 是否加“简洁回答”的轻量提示
USE_CONCISE_PROMPT = True

# 温和版，不要写得太死
CONCISE_INSTRUCTION = (
    "Please answer clearly and concisely. "
    "Keep the response brief and focused, and avoid unnecessary details unless needed."
)

# =========================
# Utils
# =========================
def write_jsonl(path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sha1_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()

def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    x = x.replace("\u0000", " ")
    x = x.strip()
    return x

def first_user_assistant_pair(messages: List[Dict]) -> Optional[Dict[str, str]]:
    """
    只取首轮 user -> assistant
    要求:
      - messages长度>=2
      - 第1条是user
      - 第2条是assistant
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    m0, m1 = messages[0], messages[1]
    if not isinstance(m0, dict) or not isinstance(m1, dict):
        return None

    if m0.get("role") != "user":
        return None
    if m1.get("role") != "assistant":
        return None

    user_text = normalize_text(m0.get("content", ""))
    assistant_text = normalize_text(m1.get("content", ""))

    if not user_text or not assistant_text:
        return None

    return {"user": user_text, "assistant": assistant_text}

def build_prompt(user_text: str) -> str:
    """
    构造后续训练用prompt。
    这里不直接拼target，只构造输入侧prompt。
    采用一个温和的system-like instruction，避免模型输出太长。
    """
    if USE_CONCISE_PROMPT:
        prompt = (
            f"<|im_start|>system\n"
            f"{CONCISE_INSTRUCTION}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_text}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        prompt = (
            f"<|im_start|>user\n"
            f"{user_text}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return prompt

def tokenize_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def process_split(
    ds,
    tokenizer,
    sample_cap: Optional[int],
    keep_cap: Optional[int],
    split_name: str
) -> List[Dict]:
    """
    流程:
      1. 随机打乱
      2. 先抽 sample_cap
      3. 提取首轮
      4. 构造强化后的prompt
      5. 基于 tokenizer 做长度过滤
      6. 输出统一jsonl格式
    """
    n_total = len(ds)
    idxs = list(range(n_total))
    random.shuffle(idxs)

    if sample_cap is not None:
        idxs = idxs[:sample_cap]

    kept = []
    stats = {
        "raw_seen": 0,
        "bad_pair": 0,
        "too_short_resp": 0,
        "too_long_resp": 0,
        "too_long_prompt": 0,
        "too_long_total": 0,
        "kept": 0,
    }

    pbar = tqdm(idxs, desc=f"filtering {split_name}", total=len(idxs))

    for i in pbar:
        ex = ds[i]
        stats["raw_seen"] += 1

        pair = first_user_assistant_pair(ex.get("messages", []))
        if pair is None:
            stats["bad_pair"] += 1
            continue

        user_text = pair["user"]
        assistant_text = pair["assistant"]

        prompt = build_prompt(user_text)
        target = assistant_text

        # 只 tokenize 两次，不再单独 tokenize(prompt + target)
        prompt_ids = tokenize_ids(tokenizer, prompt)
        target_ids = tokenize_ids(tokenizer, target)

        prompt_tokens = len(prompt_ids)
        target_tokens = len(target_ids)
        total_tokens = prompt_tokens + target_tokens

        if target_tokens < MIN_RESPONSE_TOKENS:
            stats["too_short_resp"] += 1
            continue
        if target_tokens > MAX_RESPONSE_TOKENS:
            stats["too_long_resp"] += 1
            continue
        if prompt_tokens > MAX_PROMPT_TOKENS:
            stats["too_long_prompt"] += 1
            continue
        if total_tokens > MAX_TOTAL_TOKENS:
            stats["too_long_total"] += 1
            continue

        rec = {
            "id": sha1_text(f"{split_name}::{user_text}::{assistant_text}"),
            "dataset": DATASET_NAME,
            "split": split_name,

            # 后续训练主要用这两个
            "prompt": prompt,
            "target": target,

            # 保留原始文本，方便排查
            "prompt_raw": user_text,
            "target_raw": assistant_text,

            # 元信息
            "prompt_tokens": prompt_tokens,
            "target_tokens": target_tokens,
            "total_tokens": total_tokens,
        }
        kept.append(rec)
        stats["kept"] += 1

        if stats["raw_seen"] % 1000 == 0:
            pbar.set_postfix({
                "seen": stats["raw_seen"],
                "kept": stats["kept"],
                "keep_rate": f"{stats['kept'] / max(stats['raw_seen'], 1):.3f}",
            })

        if keep_cap is not None and len(kept) >= keep_cap:
            break

    print(f"\n[{split_name}] stats")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if kept:
        avg_prompt = sum(x["prompt_tokens"] for x in kept) / len(kept)
        avg_target = sum(x["target_tokens"] for x in kept) / len(kept)
        avg_total = sum(x["total_tokens"] for x in kept) / len(kept)
        print(f"  avg_prompt_tokens: {avg_prompt:.2f}")
        print(f"  avg_target_tokens: {avg_target:.2f}")
        print(f"  avg_total_tokens: {avg_total:.2f}")

    return kept

# =========================
# Main
# =========================
def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True,
        use_fast=True,
    )

    print("Loading dataset...")
    ds_train = load_dataset(DATASET_NAME, split=TRAIN_SPLIT)
    ds_eval = load_dataset(DATASET_NAME, split=TEST_SPLIT)

    train_records = process_split(
        ds=ds_train,
        tokenizer=tokenizer,
        sample_cap=MAX_TRAIN_SAMPLES_BEFORE_FILTER,
        keep_cap=MAX_TRAIN_KEEP,
        split_name=TRAIN_SPLIT,
    )

    eval_records = process_split(
        ds=ds_eval,
        tokenizer=tokenizer,
        sample_cap=MAX_EVAL_SAMPLES_BEFORE_FILTER,
        keep_cap=MAX_EVAL_KEEP,
        split_name=TEST_SPLIT,
    )

    train_path = os.path.join(OUT_DIR, "ultrachat_train_kd.jsonl")
    eval_path = os.path.join(OUT_DIR, "ultrachat_eval_kd.jsonl")

    write_jsonl(train_path, train_records)
    write_jsonl(eval_path, eval_records)

    print("\nSaved files:")
    print(train_path, len(train_records))
    print(eval_path, len(eval_records))

    # 抽样看几条
    preview_path = os.path.join(OUT_DIR, "preview_samples.jsonl")
    preview = random.sample(train_records, min(20, len(train_records)))
    write_jsonl(preview_path, preview)
    print(preview_path, len(preview))

if __name__ == "__main__":
    main()