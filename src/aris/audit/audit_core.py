"""
从 12_audit_notes_with_ac.py 提取的核心 audit 逻辑
用于独立的验证实验
"""
import json
import pickle
import sys
from pathlib import Path

# 尝试导入 pyahocorasick
try:
    import ahocorasick
    HAS_AC = True
except ImportError:
    HAS_AC = False
    print("[WARN] pyahocorasick not available, using fallback", file=sys.stderr)

def load_rules(rules_dir):
    """加载规则和 AC index"""
    rules_dir = Path(rules_dir)
    
    # 加载 rules.json
    rules_path = rules_dir / "rules.json"
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing rules.json: {rules_path}")
    
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    
    # 加载 AC index
    ac_index_path = rules_dir / "ac_index.pkl"
    ac_obj = None
    
    if ac_index_path.exists():
        with open(ac_index_path, 'rb') as f:
            ac_data = pickle.load(f)
            ac_obj = ac_data.get('ac_obj')
            engine = ac_data.get('engine', 'unknown')
            
            if ac_obj is None and HAS_AC:
                print(f"[WARN] AC index exists but ac_obj is None", file=sys.stderr)
    
    return rules, ac_obj

def audit_text_with_ac(text, ac_obj, rules, profile='strict'):
    """
    使用 AC automaton 对文本进行 audit
    
    返回: (blocked, n_hits, reasons, hits_by_label)
    """
    if ac_obj is None:
        # Fallback: no AC available
        return False, 0, [], {}
    
    # 查找所有匹配
    matches = []
    try:
        for end_idx, (label, token) in ac_obj.iter(text.lower()):
            start_idx = end_idx - len(token) + 1
            matches.append({
                'label': label,
                'token': token,
                'start': start_idx,
                'end': end_idx + 1
            })
    except AttributeError:
        # ac_obj 没有 iter 方法
        return False, 0, [], {}
    
    # 按 label 统计 hits
    hits_by_label = {}
    for m in matches:
        label = m['label']
        hits_by_label[label] = hits_by_label.get(label, 0) + 1
    
    # 计算总 hits
    n_hits = len(matches)
    
    # 获取 thresholds 和 profile 配置
    thresholds = rules.get('thresholds', {})
    profiles = rules.get('profiles', {})
    
    profile_config = profiles.get(profile, {})
    min_hits = profile_config.get('min_hits', 2)
    
    # 判断是否 block
    blocked = n_hits < min_hits
    
    # 生成 reasons
    reasons = []
    if blocked:
        reasons.append({
            'rule': 'min_hits',
            'profile': profile,
            'threshold': min_hits,
            'actual': n_hits
        })
    
    return blocked, n_hits, reasons, hits_by_label

def audit_record(record, ac_obj, rules, profile='strict'):
    """
    对单条记录进行 audit
    
    输入: {"id": ..., "content": ...}
    输出: {"id": ..., "blocked": ..., "n_hits": ..., "reasons": ..., ...}
    """
    text = record.get('content', record.get('notes', record.get('text', '')))
    
    blocked, n_hits, reasons, hits_by_label = audit_text_with_ac(
        text, ac_obj, rules, profile
    )
    
    return {
        'id': record.get('id', record.get('doc_id', 'unknown')),
        'blocked': blocked,
        'n_hits': n_hits,
        'reasons': reasons,
        'hits_by_label': hits_by_label,
        'profile': profile,
    }

# 简化的测试函数
if __name__ == '__main__':
    print("Testing audit_core_lib...")
    
    # 测试加载规则
    try:
        rules, ac_obj = load_rules('audit_rules/releases/_CURRENT')
        print(f"✓ Rules loaded")
        print(f"  AC object type: {type(ac_obj)}")
        print(f"  Rules keys: {list(rules.keys())}")
    except Exception as e:
        print(f"✗ Failed to load rules: {e}")
        sys.exit(1)
    
    # 测试 audit
    test_text = "Patient prescribed metformin 500mg twice daily for diabetes"
    blocked, n_hits, reasons, hits = audit_text_with_ac(test_text, ac_obj, rules)
    
    print(f"\nTest audit:")
    print(f"  Text: {test_text}")
    print(f"  Blocked: {blocked}")
    print(f"  Hits: {n_hits}")
    print(f"  By label: {hits}")
    print(f"  Reasons: {reasons}")
