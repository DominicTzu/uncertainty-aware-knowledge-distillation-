import os
import json
import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm.auto import tqdm

# =========================
# Config
# =========================
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 如果要做 entropy 分桶，需要 teacher
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
USE_TEACHER_FOR_BUCKETS = True

# 评测目标模型
MODEL_PATH = "/root/exp/qwen25_3b_ukd_entropy_ultrachat/lora_final"
RUN_NAME = "ukd_entropy"
USE_PEFT = True

# 数据
EVAL_JSONL = "/root/data/ultrachat_kd/ultrachat_eval_kd.jsonl"

# 输出
SUMMARY_JSONL = "/root/exp/kd_project_eval/eval_summary.jsonl"
DETAIL_DIR = "/root/exp/kd_project_eval/details"
os.makedirs(DETAIL_DIR, exist_ok=True)

# 样本数
MAX_EVAL_SAMPLES = 1000  # 正式可设 None

# 生成参数
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
NUM_BEAMS = 1

# teacher-forcing eval 最大长度
MAX_LEN = 768

EPS = 1e-12

# =========================
# Utils
# =========================
def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: str, record: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def maybe_take(data: List[Dict], n: int):
    if n is None or n >= len(data):
        return data
    return data[:n]

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def safe_exp(x: float) -> float:
    # 防止极端情况下exp溢出
    x = min(x, 50.0)
    return math.exp(x)

def normalize_text(x: str) -> str:
    x = x.strip().lower()
    x = " ".join(x.split())
    return x

def whitespace_tokens(x: str) -> List[str]:
    return normalize_text(x).split()

def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred == ref else 0.0

def normalized_exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0

def token_overlap_f1(pred: str, ref: str) -> float:
    pred_toks = whitespace_tokens(pred)
    ref_toks = whitespace_tokens(ref)

    if len(pred_toks) == 0 and len(ref_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0

    ref_counts = {}
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    for t in pred_toks:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)

def build_full_ids_and_labels(tokenizer, prompt: str, target: str, max_len: int):
    """
    full_text = prompt + target
    labels: prompt部分为-100, target部分保留token id
    """
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    full_ids = tokenizer(
        prompt + target,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    full_ids = full_ids[:max_len]
    prompt_len = min(len(prompt_ids), len(full_ids))

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[:len(full_ids)]

    attention_mask = [1] * len(full_ids)

    return (
        torch.tensor([full_ids], dtype=torch.long),
        torch.tensor([attention_mask], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )

# =========================
# Load models
# =========================
def load_student_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    if USE_PEFT:
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
    else:
        model = base_model

    return model, tokenizer

def load_teacher():
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher

# =========================
# Generation eval
# =========================
@torch.no_grad()
def generate_one(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        num_beams=NUM_BEAMS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return pred

# =========================
# Teacher-forcing metrics
# =========================
@torch.no_grad()
def compute_response_metrics_for_one(
    model,
    tokenizer,
    prompt: str,
    target: str,
    max_len: int,
):
    """
    计算 student 在 gold target 上的 response-only:
      - mean NLL
      - PPL
      - mean top1 confidence
      - active token count
    """
    input_ids, attention_mask, labels = build_full_ids_and_labels(
        tokenizer, prompt, target, max_len
    )
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    labels = labels.to(model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # [1, L, V]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    active_mask = shift_labels.ne(-100)  # [1, L-1]
    if active_mask.sum().item() == 0:
        return {
            "nll": 0.0,
            "ppl": 1.0,
            "mean_confidence": 0.0,
            "num_active_tokens": 0,
        }

    active_logits = shift_logits[active_mask]  # [N, V]
    active_labels = shift_labels[active_mask]  # [N]

    log_probs = F.log_softmax(active_logits, dim=-1)  # [N, V]
    probs = log_probs.exp()

    gold_logp = log_probs.gather(
        dim=-1,
        index=active_labels.unsqueeze(-1)
    ).squeeze(-1)  # [N]

    nll = -gold_logp.mean().item()
    ppl = safe_exp(nll)

    top1_conf = probs.max(dim=-1).values.mean().item()

    return {
        "nll": nll,
        "ppl": ppl,
        "mean_confidence": top1_conf,
        "num_active_tokens": int(active_labels.numel()),
    }

@torch.no_grad()
def compute_teacher_entropy_for_one(
    teacher_model,
    tokenizer,
    prompt: str,
    target: str,
    max_len: int,
):
    """
    用teacher在 gold target 上计算 response-only 平均 entropy
    """
    input_ids, attention_mask, labels = build_full_ids_and_labels(
        tokenizer, prompt, target, max_len
    )
    input_ids = input_ids.to(teacher_model.device)
    attention_mask = attention_mask.to(teacher_model.device)
    labels = labels.to(teacher_model.device)

    outputs = teacher_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    active_mask = shift_labels.ne(-100)
    if active_mask.sum().item() == 0:
        return {
            "teacher_mean_entropy": 0.0,
            "teacher_num_active_tokens": 0,
        }

    active_logits = shift_logits[active_mask]  # [N, V]
    log_probs = F.log_softmax(active_logits, dim=-1)
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)  # [N]
    return {
        "teacher_mean_entropy": entropy.mean().item(),
        "teacher_num_active_tokens": int(entropy.numel()),
    }

# =========================
# Bucket stats
# =========================
def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
    return xs[idx]

def summarize_bucket(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "count": 0,
            "normalized_exact_match": 0.0,
            "token_overlap_f1": 0.0,
            "response_nll": 0.0,
            "response_ppl": 1.0,
            "teacher_mean_entropy": 0.0,
        }

    return {
        "count": len(rows),
        "normalized_exact_match": mean([r["normalized_exact_match"] for r in rows]),
        "token_overlap_f1": mean([r["token_overlap_f1"] for r in rows]),
        "response_nll": mean([r["response_nll"] for r in rows]),
        "response_ppl": mean([r["response_ppl"] for r in rows]),
        "teacher_mean_entropy": mean([r["teacher_mean_entropy"] for r in rows]),
    }

def build_entropy_buckets(details: List[Dict]) -> Dict:
    valid = [d for d in details if "teacher_mean_entropy" in d]
    if not valid:
        return {}

    ents = [d["teacher_mean_entropy"] for d in valid]
    q33 = percentile(ents, 1/3)
    q66 = percentile(ents, 2/3)

    low = []
    mid = []
    high = []

    for d in valid:
        x = d["teacher_mean_entropy"]
        if x <= q33:
            low.append(d)
        elif x <= q66:
            mid.append(d)
        else:
            high.append(d)

    return {
        "entropy_bucket_thresholds": {
            "q33": q33,
            "q66": q66,
        },
        "low_entropy_bucket": summarize_bucket(low),
        "mid_entropy_bucket": summarize_bucket(mid),
        "high_entropy_bucket": summarize_bucket(high),
    }

# =========================
# Main
# =========================
def main():
    print("Loading student...")
    model, tokenizer = load_student_and_tokenizer()

    teacher_model = None
    if USE_TEACHER_FOR_BUCKETS:
        print("Loading teacher...")
        teacher_model = load_teacher()

    print("Loading eval data...")
    data = read_jsonl(EVAL_JSONL)
    data = maybe_take(data, MAX_EVAL_SAMPLES)
    print(f"eval size = {len(data)}")

    details = []

    em_list = []
    nem_list = []
    f1_list = []
    pred_len_list = []
    ref_len_list = []

    response_nll_list = []
    response_ppl_list = []
    student_conf_list = []
    teacher_ent_list = []

    pbar = tqdm(data, total=len(data), desc=f"Evaluating {RUN_NAME}")

    for i, ex in enumerate(pbar):
        prompt = ex["prompt"]
        target = ex["target"]

        # generation eval
        pred = generate_one(model, tokenizer, prompt)
        em = exact_match(pred, target)
        nem = normalized_exact_match(pred, target)
        f1 = token_overlap_f1(pred, target)

        pred_len = len(tokenizer(pred, add_special_tokens=False)["input_ids"])
        ref_len = len(tokenizer(target, add_special_tokens=False)["input_ids"])

        # teacher-forcing eval on gold target
        tf_metrics = compute_response_metrics_for_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            max_len=MAX_LEN,
        )

        detail = {
            "run_name": RUN_NAME,
            "idx": i,
            "id": ex.get("id"),
            "prompt": prompt,
            "target": target,
            "prediction": pred,

            "exact_match": em,
            "normalized_exact_match": nem,
            "token_overlap_f1": f1,

            "pred_len": pred_len,
            "ref_len": ref_len,

            "response_nll": tf_metrics["nll"],
            "response_ppl": tf_metrics["ppl"],
            "student_mean_confidence": tf_metrics["mean_confidence"],
            "num_active_tokens": tf_metrics["num_active_tokens"],
        }

        if teacher_model is not None:
            teacher_metrics = compute_teacher_entropy_for_one(
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                prompt=prompt,
                target=target,
                max_len=MAX_LEN,
            )
            detail.update(teacher_metrics)

        details.append(detail)

        em_list.append(em)
        nem_list.append(nem)
        f1_list.append(f1)
        pred_len_list.append(pred_len)
        ref_len_list.append(ref_len)

        response_nll_list.append(tf_metrics["nll"])
        response_ppl_list.append(tf_metrics["ppl"])
        student_conf_list.append(tf_metrics["mean_confidence"])

        if teacher_model is not None:
            teacher_ent_list.append(detail["teacher_mean_entropy"])

        postfix = {
            "nem": f"{mean(nem_list):.4f}",
            "f1": f"{mean(f1_list):.4f}",
            "nll": f"{mean(response_nll_list):.4f}",
            "ppl": f"{mean(response_ppl_list):.4f}",
        }
        if teacher_model is not None and teacher_ent_list:
            postfix["t_ent"] = f"{mean(teacher_ent_list):.4f}"

        pbar.set_postfix(postfix)

        if (i + 1) % 20 == 0:
            msg = (
                f"[{i + 1}/{len(data)}] "
                f"nem={mean(nem_list):.4f} "
                f"f1={mean(f1_list):.4f} "
                f"nll={mean(response_nll_list):.4f} "
                f"ppl={mean(response_ppl_list):.4f}"
            )
            if teacher_model is not None and teacher_ent_list:
                msg += f" teacher_entropy={mean(teacher_ent_list):.4f}"
            print(msg)

    # overall summary
    summary = {
        "run_name": RUN_NAME,
        "model_path": MODEL_PATH,
        "base_model": BASE_MODEL,
        "eval_jsonl": EVAL_JSONL,
        "num_examples": len(details),

        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "num_beams": NUM_BEAMS,

        # generation metrics
        "exact_match": mean(em_list),
        "normalized_exact_match": mean(nem_list),
        "token_overlap_f1": mean(f1_list),
        "avg_pred_len": mean(pred_len_list),
        "avg_ref_len": mean(ref_len_list),

        # teacher-forcing metrics
        "response_nll": mean(response_nll_list),
        "response_ppl": mean(response_ppl_list),
        "student_mean_confidence": mean(student_conf_list),
    }

    if teacher_model is not None:
        summary["teacher_mean_entropy"] = mean(teacher_ent_list)
        summary.update(build_entropy_buckets(details))

    # save
    detail_path = os.path.join(DETAIL_DIR, f"{RUN_NAME}_details.jsonl")
    write_jsonl(detail_path, details)
    append_jsonl(SUMMARY_JSONL, summary)

    print("\nSaved detail file:")
    print(detail_path)

    print("\nAppended summary to:")
    print(SUMMARY_JSONL)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()