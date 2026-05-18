"""
分析 audit 的判别能力
"""
import json
from collections import defaultdict

# 读取结果
with open("audit_test_results.jsonl", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

# 统计
stats = defaultdict(int)
confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

for r in results:
    expected = r["expected"]
    actual = r["actual"]
    
    # 混淆矩阵
    if expected == "block" and actual == "block":
        confusion["TP"] += 1  # True Positive (正确拦截垃圾)
    elif expected == "pass" and actual == "pass":
        confusion["TN"] += 1  # True Negative (正确放行质量)
    elif expected == "pass" and actual == "block":
        confusion["FP"] += 1  # False Positive (误杀质量)
    elif expected == "block" and actual == "pass":
        confusion["FN"] += 1  # False Negative (漏过垃圾)
    
    stats[f"{expected}_total"] += 1
    if r["correct"]:
        stats[f"{expected}_correct"] += 1

# 计算指标
total = len(results)
accuracy = (confusion["TP"] + confusion["TN"]) / total
precision = confusion["TP"] / (confusion["TP"] + confusion["FP"]) if (confusion["TP"] + confusion["FP"]) > 0 else 0
recall = confusion["TP"] / (confusion["TP"] + confusion["FN"]) if (confusion["TP"] + confusion["FN"]) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("="*60)
print("Audit 判别能力分析")
print("="*60)

print(f"\n📊 总体统计:")
print(f"  测试样本: {total}")
print(f"  准确率: {accuracy*100:.1f}% ({confusion['TP']+confusion['TN']}/{total})")
print(f"  精确率: {precision*100:.1f}% (拦截的有多少是真垃圾)")
print(f"  召回率: {recall*100:.1f}% (垃圾有多少被拦截)")
print(f"  F1 分数: {f1:.3f}")

print(f"\n📋 混淆矩阵:")
print(f"                预测: Block    预测: Pass")
print(f"  实际: Garbage    {confusion['TP']:>4}          {confusion['FN']:>4}")
print(f"  实际: Quality    {confusion['FP']:>4}          {confusion['TN']:>4}")

print(f"\n🎯 分类表现:")
print(f"  垃圾信息 (n={stats['block_total']}):")
print(f"    正确拦截: {stats['block_correct']} ({stats['block_correct']/stats['block_total']*100:.1f}%)")
print(f"    漏过: {stats['block_total']-stats['block_correct']}")

print(f"\n  质量信息 (n={stats['pass_total']}):")
print(f"    正确放行: {stats['pass_correct']} ({stats['pass_correct']/stats['pass_total']*100:.1f}%)")
print(f"    误拦截: {stats['pass_total']-stats['pass_correct']}")

# 显示错误案例
print(f"\n❌ 错误案例分析:")
errors = [r for r in results if not r["correct"]]

if errors:
    print(f"  总错误: {len(errors)}")
    
    # False Positives (误杀质量信息)
    fps = [r for r in errors if r["expected"] == "pass"]
    if fps:
        print(f"\n  误拦截质量信息 (FP): {len(fps)} 条")
        for r in fps[:3]:
            print(f"    - {r['id']}: {r['n_hits']} hits")
            if r['reasons']:
                print(f"      原因: {r['reasons'][0]['rule']}")
    
    # False Negatives (漏过垃圾)
    fns = [r for r in errors if r["expected"] == "block"]
    if fns:
        print(f"\n  漏过垃圾信息 (FN): {len(fns)} 条")
        for r in fns[:3]:
            print(f"    - {r['id']}: {r['n_hits']} hits (不够拦截)")

print("\n" + "="*60)
print("论文结论")
print("="*60)

if accuracy >= 0.7:
    print(f"✅ Audit 具有良好的判别能力 ({accuracy*100:.0f}% 准确率)")
    print(f"   可以有效区分垃圾信息和质量信息")
    print(f"   适合作为 BNN gate 的预筛选层")
elif accuracy >= 0.5:
    print(f"⚠️  Audit 有一定判别能力 ({accuracy*100:.0f}% 准确率)")
    print(f"   但需要进一步优化规则")
else:
    print(f"❌ Audit 判别能力不足 ({accuracy*100:.0f}% 准确率)")
    print(f"   需要重新设计规则")

print(f"\n推荐改进方向:")
if precision < 0.8:
    print(f"  - 降低误拦截 (FP={confusion['FP']}): 放宽某些规则")
if recall < 0.8:
    print(f"  - 提高拦截率 (FN={confusion['FN']}): 加强垃圾检测规则")
