import json
import random

random.seed(42)

# ============================================
# 垃圾信息模板（应该被 audit 拦截）
# ============================================
garbage_templates = [
    # 缺少关键信息
    "Patient came in today.",
    "Feeling unwell. Need help.",
    "Some symptoms present.",
    "Check patient.",
    "Follow up needed.",
    
    # 不完整描述
    "Rash on arm. Red spots.",
    "Headache and tired.",
    "Stomach pain for 2 days.",
    "Cough and fever.",
    "Dizzy when standing.",
    
    # 没有具体细节
    "Patient reports allergic reaction.",
    "Possible drug interaction.",
    "Symptoms worsened recently.",
    "Needs medication review.",
    "Complained of discomfort.",
    
    # 过于简短
    "Allergic.",
    "Rash.",
    "Pain.",
    "Swelling noted.",
    "Patient uncomfortable.",
    
    # 无诊断价值
    "Patient waiting for results.",
    "Tests ordered.",
    "Will monitor situation.",
    "Discussed with patient.",
    "Patient left clinic.",
    
    # 错误/矛盾信息
    "Patient allergic to peanuts but ate peanuts.",
    "No symptoms but in distress.",
    "Mild reaction requiring immediate surgery.",
    "Asymptomatic emergency.",
    "Healthy patient in ICU.",
    
    # 缺少时间/剂量等关键细节
    "Gave medication.",
    "Patient took some pills.",
    "Applied treatment.",
    "Administered drug.",
    "Patient received therapy.",
]

# ============================================
# 质量信息模板（应该通过 audit）
# ============================================
quality_templates = [
    # 完整的过敏记录
    """Patient: 34F, History: Known peanut allergy (anaphylaxis 2020)
Presentation: Accidental peanut exposure at restaurant. Onset within 5 minutes.
Symptoms: Urticaria on face/neck, lip swelling, throat tightness, difficulty breathing
Vitals: BP 110/70, HR 105, RR 24, O2Sat 94% room air
Allergen: Peanut (confirmed via ingredient list)
Treatment: EpiPen 0.3mg IM (right thigh), Benadryl 50mg PO
Response: Symptoms improved after 10 minutes, breathing normalized
Plan: Observe 4 hours, prescribe EpiPen x2, allergy clinic referral""",

    # 详细的药物反应
    """Patient: 56M, Day 3 of Amoxicillin 500mg PO TID for sinusitis
Onset: 2 hours post-dose, generalized pruritic maculopapular rash
Distribution: Trunk and extremities, no mucosal involvement
Associated: Mild nausea, no fever, no respiratory symptoms
Drug: Amoxicillin (first exposure, no prior beta-lactam use)
Labs: Normal CBC, CMP within limits
Action: Discontinued amoxicillin, started cetirizine 10mg daily
Substitute: Prescribed azithromycin 500mg daily x5 for sinusitis
Documentation: Allergy alert added to chart""",

    # 多过敏原记录
    """Patient: 28F, Atopic history (eczema, allergic rhinitis)
Allergens: Shellfish (shrimp, crab - urticaria), Tree nuts (walnuts - angioedema), 
          Latex (contact dermatitis), Pollen (seasonal rhinitis)
Testing: Skin prick positive for above allergens (2024-01)
Avoidance: Patient educated on cross-reactivity (latex-food syndrome)
Medications: Carries EpiPen 0.3mg, uses loratadine 10mg PRN
Recent: No reactions in past 6 months with strict avoidance
Plan: Annual allergy review, consider immunotherapy for pollen""",

    # 实验室确认的过敏
    """Patient: 42M, Suspected penicillin allergy (childhood rash)
Testing: Penicillin skin testing performed (2024-01-10)
Results: Positive wheal 8mm to penicillin G, negative control 0mm
IgE: Penicillin-specific IgE 5.2 kU/L (positive >0.35)
Conclusion: Type I hypersensitivity to penicillin confirmed
Alternatives: Documented safe use of azithromycin, fluoroquinolones
Cross-reactivity: Avoid all beta-lactams, cephalosporins cautious
Updated: Allergy chart updated with test results and alternatives""",

    # 药物剂量调整记录
    """Patient: 67F, CKD Stage 3 (eGFR 45), HTN, DM2
Current: Metformin 1000mg BID, Lisinopril 20mg daily
Issue: Starting antibiotic for UTI, need dose adjustment
Drug: Ciprofloxacin selected (culture pending)
Calculation: CrCl 42 ml/min (Cockcroft-Gault)
Dosing: Reduced to 250mg PO BID (vs standard 500mg BID)
Duration: 7 days, monitor renal function
Interaction: No significant interaction with current meds
Follow-up: Repeat culture 48hr, renal panel 1 week""",
]

# 扩展到 50 条
def generate_garbage(n=50):
    """生成垃圾信息"""
    records = []
    for i in range(n):
        # 随机选择模板或组合多个短句
        if random.random() < 0.7:
            content = random.choice(garbage_templates)
        else:
            # 组合 2-3 个短句
            parts = random.sample(garbage_templates, random.randint(2, 3))
            content = " ".join(parts)
        
        records.append({
            "id": f"GARBAGE-{i+1:03d}",
            "type": "test_garbage",
            "content": content,
            "expected_audit_result": "block",  # 期望被拦截
            "timestamp": "2026-01-15T00:00:00Z"
        })
    
    return records

def generate_quality(n=50):
    """生成质量信息"""
    records = []
    for i in range(n):
        # 使用质量模板，随机变化细节
        template = random.choice(quality_templates)
        
        # 随机化一些参数
        template = template.replace("34F", f"{random.randint(25,75)}{random.choice(['M','F'])}")
        template = template.replace("56M", f"{random.randint(25,75)}{random.choice(['M','F'])}")
        
        records.append({
            "id": f"QUALITY-{i+1:03d}",
            "type": "test_quality",
            "content": template,
            "expected_audit_result": "pass",  # 期望通过
            "timestamp": "2026-01-15T00:00:00Z"
        })
    
    return records

# 生成数据
garbage = generate_garbage(50)
quality = generate_quality(50)

# 合并并打乱
all_records = garbage + quality
random.shuffle(all_records)

# 保存
with open("test_cases_100.jsonl", "w", encoding="utf-8") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✓ 生成 {len(all_records)} 条测试数据")
print(f"  垃圾信息: {len(garbage)} 条 (期望 audit hit)")
print(f"  质量信息: {len(quality)} 条 (期望 audit pass)")
print(f"  保存到: test_cases_100.jsonl")
