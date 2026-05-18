#!/usr/bin/env python3
"""
Synthetic audit-stress validation for the deterministic audit gate.

This script evaluates whether the learned audit gate separates structurally
sufficient candidate records from low-quality stress cases. High-quality
controls are reconstructed from locally generated entity extractions, while
low-quality cases are synthetic incomplete clinical fragments.

The repository does not redistribute MIMIC-IV records, derived patient-level
tables, entity spans, or record-level routing manifests. Set the following
environment variables before running in an authorized local environment:

  ARIS_REPO
  ARIS_DERIVED_DATA
  ARIS_AUDIT_STRESS_INPUT
  ARIS_RULES_DIR
"""
import json
import pickle
import random
import sys
import os
from pathlib import Path
import pandas as pd

sys.path.insert(0, '${ARIS_REPO}')

if len(sys.argv) < 2:
    sys.exit(1)

SEED = int(sys.argv[1])
random.seed(SEED)

print(f"{'='*70}")
print(f"Audit Validation - Seed {SEED}")
print(f"Sample: 2000 (1000 low + 1000 high)")
print(f"Strategy: Real high-quality + Synthetic low-quality")
print(f"{'='*70}")

try:
    import ahocorasick
    HAS_AC = True
except:
    HAS_AC = False

PARQUET_PATH = os.environ.get("ARIS_AUDIT_STRESS_INPUT", "${ARIS_DERIVED_DATA}/discharge_ner_sections_clean.parquet")
RULES_DIR = Path(os.environ.get("ARIS_RULES_DIR", "${ARIS_REPO}/audit_rules/releases/_CURRENT"))

# ============================================
# 加载系统
# ============================================
print(f"\nLoading data...")
df = pd.read_parquet(PARQUET_PATH)
print(f"✓ Loaded {len(df):,} NER records from {df['note_id'].nunique():,} docs")

def load_audit_system():
    with open(RULES_DIR / "rules.json", 'r') as f:
        rules = json.load(f)
    with open(RULES_DIR / "ac_index.pkl", 'rb') as f:
        ac_data = pickle.load(f)
        ac_obj = ac_data.get('ac_obj')
    print(f"✓ Loaded audit rules")
    return rules, ac_obj

def audit_text(text, ac_obj, rules):
    if ac_obj is None or not HAS_AC:
        return False, 0, [], {}
    
    matches = []
    try:
        for end_idx, (label, token) in ac_obj.iter(text.lower()):
            matches.append({'label': label, 'token': token})
    except:
        return False, 0, [], {}
    
    hits_by_label = {}
    for m in matches:
        label = m['label']
        hits_by_label[label] = hits_by_label.get(label, 0) + 1
    
    n_hits = len(matches)
    MIN_HITS = 2
    blocked = n_hits < MIN_HITS
    
    reasons = []
    if blocked:
        reasons.append({'rule': 'min_hits', 'threshold': MIN_HITS, 'actual': n_hits})
    
    return blocked, n_hits, reasons, hits_by_label

# ============================================
# 生成测试数据
# ============================================
def generate_test_data(df, n_low=1000, n_high=1000):
    records = []
    
    # === high-quality controls：from local derived entity extractions ===
    doc_stats = df.groupby('note_id').agg({
        'entity_text': 'count',
        'entity_label': lambda x: x.nunique(),
    }).rename(columns={'entity_text': 'n_entities', 'entity_label': 'n_types'})
    
    # 选择实体多、类型多样的文档
    high_quality_docs = doc_stats[
        (doc_stats['n_entities'] >= 8) & 
        (doc_stats['n_types'] >= 3)
    ].index.tolist()
    
    print(f"\n  High-quality pool: {len(high_quality_docs):,} docs")
    
    sampled_high = random.sample(high_quality_docs, min(n_high, len(high_quality_docs)))
    
    for i, doc_id in enumerate(sampled_high):
        doc_entities = df[df['note_id'] == doc_id].sort_values('start')
        entities = doc_entities['entity_text'].tolist()
        labels = doc_entities['entity_label'].tolist()
        
        # 构建临床文本
        text_parts = []
        for ent, lbl in zip(entities[:15], labels[:15]):
            if lbl == 'Drug':
                text_parts.append(f"administered {ent}")
            elif lbl == 'Dosage':
                text_parts.append(f"{ent}")
            elif lbl == 'Route':
                text_parts.append(f"via {ent}")
            elif lbl == 'Reason':
                text_parts.append(f"for {ent}")
            else:
                text_parts.append(ent)
        
        content = ". ".join(text_parts[:12]) + "."
        
        records.append({
            "id": f"HIGH-{i+1:04d}",
            "type": "test_quality",
            "content": content,
            "expected": "pass",
            "seed": SEED,
        })
    
    # === low-quality stress cases：合成（确保足够） ===
    print(f"  Low-quality: Synthesizing {n_low} samples")
    
    # 低质量模板（扩展）
    low_quality_templates = [
        "Patient came in today.",
        "Feeling unwell.",
        "Some symptoms present.",
        "Check patient.",
        "Follow up needed.",
        "Rash on arm.",
        "Headache.",
        "Stomach pain.",
        "Cough and fever.",
        "Dizzy.",
        "Patient reports issue.",
        "Needs review.",
        "Uncomfortable.",
        "Allergic reaction.",
        "Interaction possible.",
        "Symptoms worsened.",
        "Mild issue.",
        "Patient okay.",
        "Continue monitoring.",
        "No major concerns.",
        "Pain noted.",
        "Swelling seen.",
        "Tests ordered.",
        "Waiting results.",
        "Discussed with patient.",
        "Left clinic.",
        "Stable condition.",
        "Observation needed.",
        "Patient discharged.",
        "Return if worse.",
    ]
    
    for i in range(n_low):
        # 组合1-3个短句
        n_parts = random.randint(1, 3)
        parts = random.sample(low_quality_templates, n_parts)
        content = " ".join(parts)
        
        records.append({
            "id": f"LOW-{i+1:04d}",
            "type": "test_garbage",
            "content": content,
            "expected": "block",
            "seed": SEED,
        })
    
    random.shuffle(records)
    return records

# ============================================
# 主流程
# ============================================
rules, ac_obj = load_audit_system()

print(f"\nGenerating test data...")
test_data = generate_test_data(df, n_low=1000, n_high=1000)
print(f"✓ Generated {len(test_data)} samples")

test_file = f"test_cases_seed{SEED:03d}.jsonl"
with open(test_file, 'w', encoding='utf-8') as f:
    for r in test_data:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\nRunning audit...")
results = []

for record in test_data:
    blocked, n_hits, reasons, hits_by_label = audit_text(
        record['content'], ac_obj, rules
    )
    
    correct = (
        (record['expected'] == 'block' and blocked) or
        (record['expected'] == 'pass' and not blocked)
    )
    
    results.append({
        'id': record['id'],
        'type': record['type'],
        'expected': record['expected'],
        'actual': 'block' if blocked else 'pass',
        'n_hits': n_hits,
        'correct': correct,
        'seed': SEED,
    })

result_file = f"audit_results_seed{SEED:03d}.jsonl"
with open(result_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# 统计
confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
for r in results:
    if r['expected'] == 'block' and r['actual'] == 'block':
        confusion["TP"] += 1
    elif r['expected'] == 'pass' and r['actual'] == 'pass':
        confusion["TN"] += 1
    elif r['expected'] == 'pass' and r['actual'] == 'block':
        confusion["FP"] += 1
    elif r['expected'] == 'block' and r['actual'] == 'pass':
        confusion["FN"] += 1

total = len(results)
accuracy = (confusion["TP"] + confusion["TN"]) / total
precision = confusion["TP"] / (confusion["TP"] + confusion["FP"]) if (confusion["TP"] + confusion["FP"]) > 0 else 0
recall = confusion["TP"] / (confusion["TP"] + confusion["FN"]) if (confusion["TP"] + confusion["FN"]) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

stats = {
    "seed": SEED,
    "total": total,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion": confusion,
}

with open(f"stats_seed{SEED:03d}.json", 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*70}")
print(f"Seed {SEED} Results")
print(f"{'='*70}")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1:        {f1:.3f}")
print(f"TP={confusion['TP']}, TN={confusion['TN']}, FP={confusion['FP']}, FN={confusion['FN']}")
print(f"{'='*70}")
