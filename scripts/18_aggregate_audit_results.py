"""汇总 100 次实验结果 - 正确计算 Mean±SD"""
import json
import numpy as np
from pathlib import Path

print("="*70)
print("Aggregating 100 Audit Validation Experiments")
print("Data: Local credentialed-access derived clinical records")
print("="*70)

stats_files = sorted(Path(".").glob("stats_seed*.json"))

if len(stats_files) == 0:
    print("❌ No stats files found!")
    exit(1)

print(f"\nFound {len(stats_files)} result files")

all_stats = [json.load(open(f)) for f in stats_files]

# 提取指标
accuracies = [s["accuracy"] for s in all_stats]
precisions = [s["precision"] for s in all_stats]
recalls = [s["recall"] for s in all_stats]
f1s = [s["f1"] for s in all_stats]

# 提取每次运行的混淆矩阵
TPs = [s["confusion"]["TP"] for s in all_stats]
TNs = [s["confusion"]["TN"] for s in all_stats]
FPs = [s["confusion"]["FP"] for s in all_stats]
FNs = [s["confusion"]["FN"] for s in all_stats]

def ci_95(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    margin = 1.96 * std / np.sqrt(len(data))
    return mean - margin, mean + margin

acc_ci = ci_95(accuracies)
prec_ci = ci_95(precisions)
rec_ci = ci_95(recalls)

print("\n" + "="*70)
print("RESULTS (100 experiments, 2000 samples each)")
print("="*70)

print(f"\n📊 Accuracy:")
print(f"  Mean ± SD:  {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies, ddof=1)*100:.2f}%")
print(f"  95% CI:     [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%]")
print(f"  Range:      [{np.min(accuracies)*100:.2f}%, {np.max(accuracies)*100:.2f}%]")

print(f"\n📊 Precision:")
print(f"  Mean ± SD:  {np.mean(precisions)*100:.2f}% ± {np.std(precisions, ddof=1)*100:.2f}%")
print(f"  95% CI:     [{prec_ci[0]*100:.2f}%, {prec_ci[1]*100:.2f}%]")

print(f"\n📊 Recall:")
print(f"  Mean ± SD:  {np.mean(recalls)*100:.2f}% ± {np.std(recalls, ddof=1)*100:.2f}%")
print(f"  95% CI:     [{rec_ci[0]*100:.2f}%, {rec_ci[1]*100:.2f}%]")

print(f"\n📊 F1-Score:")
print(f"  Mean ± SD:  {np.mean(f1s):.3f} ± {np.std(f1s, ddof=1):.3f}")

print(f"\n📋 Confusion Matrix (per run, out of 2000):")
print(f"  TP: {np.mean(TPs):.1f} ± {np.std(TPs, ddof=1):.1f}")
print(f"  TN: {np.mean(TNs):.1f} ± {np.std(TNs, ddof=1):.1f}")
print(f"  FP: {np.mean(FPs):.1f} ± {np.std(FPs, ddof=1):.1f}")
print(f"  FN: {np.mean(FNs):.1f} ± {np.std(FNs, ddof=1):.1f}")

# 保存汇总
summary = {
    "n_experiments": len(all_stats),
    "samples_per_run": 2000,
    "total_samples": len(all_stats) * 2000,
    "accuracy": {
        "mean": float(np.mean(accuracies)), 
        "std": float(np.std(accuracies, ddof=1)),
        "ci_95": [float(acc_ci[0]), float(acc_ci[1])]
    },
    "precision": {
        "mean": float(np.mean(precisions)), 
        "std": float(np.std(precisions, ddof=1)),
        "ci_95": [float(prec_ci[0]), float(prec_ci[1])]
    },
    "recall": {
        "mean": float(np.mean(recalls)), 
        "std": float(np.std(recalls, ddof=1)),
        "ci_95": [float(rec_ci[0]), float(rec_ci[1])]
    },
    "f1": {
        "mean": float(np.mean(f1s)), 
        "std": float(np.std(f1s, ddof=1))
    },
    "confusion_per_run": {
        "TP": {"mean": float(np.mean(TPs)), "std": float(np.std(TPs, ddof=1))},
        "TN": {"mean": float(np.mean(TNs)), "std": float(np.std(TNs, ddof=1))},
        "FP": {"mean": float(np.mean(FPs)), "std": float(np.std(FPs, ddof=1))},
        "FN": {"mean": float(np.mean(FNs)), "std": float(np.std(FNs, ddof=1))},
    },
    "confusion_total": {
        "TP": int(np.sum(TPs)),
        "TN": int(np.sum(TNs)),
        "FP": int(np.sum(FPs)),
        "FN": int(np.sum(FNs)),
    }
}

with open("AGGREGATE_SUMMARY.json", 'w') as f:
    json.dump(summary, f, indent=2)

# CSV 输出
with open("AGGREGATE_RESULTS.csv", 'w') as f:
    f.write("metric,mean,std,ci_95_lower,ci_95_upper\n")
    f.write(f"accuracy,{np.mean(accuracies):.4f},{np.std(accuracies, ddof=1):.4f},{acc_ci[0]:.4f},{acc_ci[1]:.4f}\n")
    f.write(f"precision,{np.mean(precisions):.4f},{np.std(precisions, ddof=1):.4f},{prec_ci[0]:.4f},{prec_ci[1]:.4f}\n")
    f.write(f"recall,{np.mean(recalls):.4f},{np.std(recalls, ddof=1):.4f},{rec_ci[0]:.4f},{rec_ci[1]:.4f}\n")
    f.write(f"f1,{np.mean(f1s):.4f},{np.std(f1s, ddof=1):.4f},---,---\n")
    f.write(f"TP,{np.mean(TPs):.2f},{np.std(TPs, ddof=1):.2f},---,---\n")
    f.write(f"TN,{np.mean(TNs):.2f},{np.std(TNs, ddof=1):.2f},---,---\n")
    f.write(f"FP,{np.mean(FPs):.2f},{np.std(FPs, ddof=1):.2f},---,---\n")
    f.write(f"FN,{np.mean(FNs):.2f},{np.std(FNs, ddof=1):.2f},---,---\n")

print(f"\n✓ Saved: AGGREGATE_SUMMARY.json")
print(f"✓ Saved: AGGREGATE_RESULTS.csv")

print("\n" + "="*70)
print(f"PAPER RESULT:")
print(f"The audit gate achieved {np.mean(accuracies)*100:.1f}% ± {np.std(accuracies, ddof=1)*100:.1f}% accuracy")
print(f"(95% CI: [{acc_ci[0]*100:.1f}%, {acc_ci[1]*100:.1f}%])")
print(f"across 100 independent runs (2000 samples each).")
print("="*70)
