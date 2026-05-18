#!/usr/bin/env python3
"""Low-resource analysis: performance vs data size"""
import json
import numpy as np
from pathlib import Path

# 从现有full ARIS结果中分析不同数据量的表现
with open("reports/controlled_runs/kgstruct_4369533/final/merged_ok.jsonl") as f:
    full_data = [json.loads(line) for line in f]

with open("reports/controlled_runs/kgstruct_4369533/bnn_out.jsonl") as f:
    bnn_data = [json.loads(line) for line in f if line.strip() and "bnn_output" in line]

results = {}
for size in [50, 100, 200, 500, 1000, 2000]:
    subset = full_data[:size]
    
    # 计算metrics
    uncertainties = [r.get("bnn_raw", {}).get("bnn_output", {}).get("total_uncertainty", 0) for r in subset]
    epistemic = [r.get("bnn_raw", {}).get("bnn_output", {}).get("epistemic_uncertainty", 0) for r in subset]
    
    results[size] = {
        "n_records": size,
        "mean_uncertainty": float(np.mean(uncertainties)),
        "std_uncertainty": float(np.std(uncertainties)),
        "mean_epistemic": float(np.mean(epistemic)),
        "convergence_pct": float(np.mean(uncertainties) / np.mean([r.get("bnn_raw", {}).get("bnn_output", {}).get("total_uncertainty", 0) for r in full_data]) * 100)
    }
    
    print(f"Size {size:4d}: unc={np.mean(uncertainties):5.2f}±{np.std(uncertainties):4.2f}, convergence={results[size]['convergence_pct']:.1f}%")

# 保存结果
Path("reports/lowres_analysis").mkdir(exist_ok=True)
with open("reports/lowres_analysis/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Low-resource analysis complete")
print("Results saved to: reports/lowres_analysis/results.json")
