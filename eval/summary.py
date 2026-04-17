import json
import pandas as pd

SUMMARY_JSONL = "/root/exp/kd_project_eval/eval_summary.jsonl"

rows = []
with open(SUMMARY_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

flat_rows = []
for r in rows:
    flat = {
        "run_name": r.get("run_name"),
        "num_examples": r.get("num_examples"),
        "norm_em": r.get("normalized_exact_match"),
        "f1": r.get("token_overlap_f1"),
        "resp_nll": r.get("response_nll"),
        "resp_ppl": r.get("response_ppl"),
        "student_conf": r.get("student_mean_confidence"),
        "teacher_entropy": r.get("teacher_mean_entropy", None),

        "low_f1": r.get("low_entropy_bucket", {}).get("token_overlap_f1"),
        "mid_f1": r.get("mid_entropy_bucket", {}).get("token_overlap_f1"),
        "high_f1": r.get("high_entropy_bucket", {}).get("token_overlap_f1"),

        "low_nll": r.get("low_entropy_bucket", {}).get("response_nll"),
        "mid_nll": r.get("mid_entropy_bucket", {}).get("response_nll"),
        "high_nll": r.get("high_entropy_bucket", {}).get("response_nll"),
    }
    flat_rows.append(flat)

df = pd.DataFrame(flat_rows)
print(df.to_string(index=False))