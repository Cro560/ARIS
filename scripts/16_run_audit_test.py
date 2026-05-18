"""
对测试数据运行 audit 规则检查
"""
import json
import sys
sys.path.insert(0, "../scripts")

# 导入 audit 函数（假设在你的代码库中）
# 如果没有，我们简化实现一个基本的 audit 检查

def simple_audit(content):
    """
    简化的 audit 规则（基于你现有的规则）
    返回: (blocked, n_hits, reasons)
    """
    reasons = []
    n_hits = 0
    
    # 规则1: 长度检查（太短）
    if len(content) < 100:
        reasons.append({"rule": "min_length", "threshold": 100, "actual": len(content)})
        n_hits += 1
    
    # 规则2: 必需标签（药物类）
    required_drug_labels = ["Drug", "Dosage", "Strength", "Frequency", "Route"]
    required_allergy_labels = ["Allergen", "Symptom", "Onset", "Treatment"]
    
    has_drug_info = any(label.lower() in content.lower() for label in 
                        ["mg", "ml", "tablet", "capsule", "daily", "bid", "tid", "po", "iv", "im"])
    has_allergy_info = any(label.lower() in content.lower() for label in
                           ["allergy", "allergen", "reaction", "urticaria", "anaphylaxis", "epipen"])
    
    # 至少需要一套完整信息
    if not has_drug_info and not has_allergy_info:
        reasons.append({"rule": "missing_clinical_labels", "required": "drug_or_allergy"})
        n_hits += 1
    
    # 规则3: 关键细节缺失
    has_vitals = any(term in content.lower() for term in ["bp", "hr", "rr", "temp", "o2sat", "vitals"])
    has_dosage = any(term in content.lower() for term in ["mg", "ml", "mcg", "units"])
    has_timeline = any(term in content.lower() for term in ["day", "hour", "minute", "onset", "duration"])
    
    detail_score = sum([has_vitals, has_dosage, has_timeline])
    if detail_score < 2:
        reasons.append({"rule": "insufficient_detail", "score": detail_score, "min": 2})
        n_hits += 1
    
    # 规则4: 模糊或无意义内容
    vague_patterns = ["some", "maybe", "possibly", "unclear", "unknown", "patient came", 
                     "need help", "check patient", "follow up needed"]
    vague_count = sum(1 for pattern in vague_patterns if pattern in content.lower())
    
    if vague_count >= 2:
        reasons.append({"rule": "vague_language", "count": vague_count})
        n_hits += 1
    
    # 判断是否 block
    blocked = n_hits >= 2  # 2 个或以上规则违反就拦截
    
    return blocked, n_hits, reasons

# 读取测试数据
with open("test_cases_100.jsonl", encoding="utf-8") as f:
    test_cases = [json.loads(line) for line in f]

# 运行 audit
results = []
for r in test_cases:
    blocked, n_hits, reasons = simple_audit(r["content"])
    
    results.append({
        "id": r["id"],
        "type": r["type"],
        "expected": r["expected_audit_result"],
        "actual": "block" if blocked else "pass",
        "n_hits": n_hits,
        "reasons": reasons,
        "correct": (r["expected_audit_result"] == "block" and blocked) or 
                  (r["expected_audit_result"] == "pass" and not blocked)
    })

# 保存结果
with open("audit_test_results.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✓ Audit 测试完成")
print(f"  结果保存: audit_test_results.jsonl")
