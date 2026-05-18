from __future__ import annotations
import json, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

# -----------------------------
# IO
# -----------------------------
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_automaton(pairs: List[Tuple[str,str]]):
    import ahocorasick
    A = ahocorasick.Automaton()
    for tok, lab in pairs:
        A.add_word(tok, (lab, tok))
    A.make_automaton()
    return A

def match_with_index(text: str, engine: str, pairs, token2label) -> Dict[str,int]:
    hits: Dict[str,int] = {}
    low = (text or "").lower()

    if engine == "pyahocorasick":
        A = _build_automaton(pairs)
        for _, (lab, tok) in A.iter(low):
            hits[lab] = hits.get(lab, 0) + 1
    else:
        for tok, lab in token2label.items():
            if tok in low:
                hits[lab] = hits.get(lab, 0) + low.count(tok)
    return hits

# -----------------------------
# Config / results
# -----------------------------
@dataclass
class StrictProfile:
    profile: str
    min_total_hits: int
    required_labels: List[str]
    min_required_labels: int

@dataclass
class ACGateResult:
    blocked: bool
    n_hits_total: int
    hits_by_label: Dict[str,int]
    req_hit: int
    reasons: List[dict]

# -----------------------------
# Load from rules release
# -----------------------------
def load_strict_profile(rules_dir: str, profile: str = "strict") -> StrictProfile:
    rules = Path(rules_dir)
    thr_path = rules / "thresholds.json"
    prof_path = rules / "profiles.json"
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing thresholds.json in {rules_dir}")
    if not prof_path.exists():
        raise FileNotFoundError(f"Missing profiles.json in {rules_dir}")

    thr = _load_json(str(thr_path))
    profiles = _load_json(str(prof_path))
    if profile not in profiles:
        raise KeyError(f"Profile '{profile}' not found; available={list(profiles.keys())}")

    prof_cfg = profiles[profile]
    return StrictProfile(
        profile=profile,
        min_total_hits=int(thr.get("min_total_hits", 1)),
        required_labels=list(prof_cfg.get("required_labels", [])),
        min_required_labels=int(prof_cfg.get("min_required_labels", 0)),
    )

def load_ac_index(rules_dir: str) -> Dict[str, Any]:
    rules = Path(rules_dir)
    ac_pkl = rules / "ac_index.pkl"
    if not ac_pkl.exists():
        raise FileNotFoundError(f"Missing ac_index.pkl in {rules_dir} (run scripts/11_build_ac_index.py)")
    payload = pickle.load(open(ac_pkl, "rb"))

    engine = payload.get("engine", "fallback_set")
    if engine == "pyahocorasick":
        try:
            import ahocorasick  # noqa: F401
        except Exception:
            engine = "fallback_set"

    return {
        "engine": engine,
        "pairs": payload.get("pairs", []),
        "token2label": payload.get("token2label", {}),
        "n_tokens": payload.get("n_tokens", len(payload.get("token2label", {}))),
    }

# -----------------------------
# Evaluate gate
# -----------------------------
def eval_gate(text: str, strict: StrictProfile, ac_index: Dict[str,Any]) -> ACGateResult:
    hits_by_label = match_with_index(text, ac_index["engine"], ac_index["pairs"], ac_index["token2label"])
    n_hits_total = int(sum(hits_by_label.values()))
    req_hit = sum(1 for lab in strict.required_labels if hits_by_label.get(lab, 0) > 0)

    reasons: List[dict] = []
    blocked = False

    if n_hits_total < strict.min_total_hits:
        blocked = True
        reasons.append({"type":"min_total_hits", "need":strict.min_total_hits, "got":n_hits_total})

    if strict.min_required_labels > 0 and req_hit < strict.min_required_labels:
        blocked = True
        reasons.append({
            "type":"min_required_labels",
            "need":strict.min_required_labels,
            "got":req_hit,
            "required_labels": strict.required_labels
        })

    return ACGateResult(
        blocked=blocked,
        n_hits_total=n_hits_total,
        hits_by_label=hits_by_label,
        req_hit=req_hit,
        reasons=reasons,
    )
