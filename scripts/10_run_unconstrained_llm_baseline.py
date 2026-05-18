import os, json, argparse, time, hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# IO (streaming)
# -----------------------------
def read_jsonl_iter(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl_append(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -----------------------------
# KG loader (npz + pkl) with mmap + cap
# -----------------------------
def load_pickle(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

class KG:
    """
    Baseline KG (memory-aware):
    - np.load(..., mmap_mode="r") to avoid loading whole array at once
    - max_edges cap to prevent CPU RAM explosion when building adjacency
    """
    def __init__(self, npz_path: str, pkl_path: str, split: str="train", max_edges: Optional[int]=None):
        self.npz_path = npz_path
        self.pkl_path = pkl_path
        self.split = split
        self.max_edges = max_edges

        z = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        if split not in z.files:
            raise ValueError(f"split '{split}' not in {z.files}")
        triples = z[split]
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError(f"Bad triples shape: {triples.shape}, expected (N,3)")

        # cap edges BEFORE building adjacency
        if max_edges is not None and triples.shape[0] > max_edges:
            triples = triples[:max_edges]

        m = load_pickle(pkl_path)

        self.id2ent = m.get("id2ent") or m.get("id2entity") or m.get("id2node") or {}
        self.id2rel = m.get("id2rel") or m.get("id2relation") or {}
        self.ent2id = m.get("ent2id") or m.get("entity2id") or m.get("node2id") or {}
        self.rel2id = m.get("rel2id") or m.get("relation2id") or {}

        self.triples = np.asarray(triples, dtype=np.int64)

        # adjacency: ent_id -> list of (rel_id, other_ent_id, direction)
        self.adj: Dict[int, List[Tuple[int,int,int]]] = {}
        for h, r, t in self.triples:
            h = int(h); r = int(r); t = int(t)
            self.adj.setdefault(h, []).append((r, t, 0))
            self.adj.setdefault(t, []).append((r, h, 1))

    def ent_name(self, ent_id: int) -> str:
        v = self.id2ent.get(int(ent_id)) if isinstance(self.id2ent, dict) else None
        return str(v) if v is not None else f"ENT_{ent_id}"

    def rel_name(self, rel_id: int) -> str:
        v = self.id2rel.get(int(rel_id)) if isinstance(self.id2rel, dict) else None
        return str(v) if v is not None else f"REL_{rel_id}"

    def resolve_entities(self, entities: List[str]) -> List[int]:
        ids=[]
        for e in entities:
            if e is None:
                continue
            key=str(e).strip()
            if not key:
                continue
            if key in self.ent2id:
                ids.append(int(self.ent2id[key]))
                continue
            lk = key.lower()
            if lk in self.ent2id:
                ids.append(int(self.ent2id[lk]))
                continue
        # de-dup preserve order
        out=[]
        seen=set()
        for x in ids:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    def retrieve(self, ent_ids: List[int], max_triples: int=24) -> List[Tuple[str,str,str,str]]:
        out=[]
        seen=set()
        for eid in ent_ids:
            for r, other, direction in self.adj.get(int(eid), []):
                h = self.ent_name(eid if direction==0 else other)
                t = self.ent_name(other if direction==0 else eid)
                rel = self.rel_name(r)
                key=(h,rel,t,direction)
                if key in seen:
                    continue
                seen.add(key)
                out.append((h, rel, t, "head" if direction==0 else "tail"))
                if len(out) >= max_triples:
                    return out
        return out

# -----------------------------
# helpers
# -----------------------------
def extract_entities_from_row(row: Dict[str, Any]) -> List[str]:
    ents=[]
    if isinstance(row.get("entities"), list):
        ents += [str(x) for x in row["entities"] if x is not None]
    if isinstance(row.get("entity_strings"), list):
        ents += [str(x) for x in row["entity_strings"] if x is not None]
    for k in ["diagnoses", "medications", "problems"]:
        v=row.get(k)
        if isinstance(v, list):
            ents += [str(x) for x in v if x is not None]
        elif isinstance(v, str):
            ents.append(v)
    out=[]
    seen=set()
    for e in ents:
        e=str(e).strip()
        if not e:
            continue
        if e not in seen:
            seen.add(e); out.append(e)
    return out

def get_text(row: Dict[str, Any]) -> str:
    # IMPORTANT: your pipeline uses "content" a lot
    for k in ["content", "text","query","input","raw","description"]:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""

def build_prompt(doc_id: str, user_text: str, kg_context: str) -> str:
    return (
        "You are a clinical documentation assistant.\n"
        "Use the provided Knowledge Graph (KG) context if it is helpful.\n"
        "Write a concise clinical note summary from the patient description.\n"
        "If medications are mentioned, include them with dose/frequency/route if available.\n"
        "Output ONLY the note text.\n\n"
        f"[DOC_ID] {doc_id}\n"
        f"[PATIENT_DESCRIPTION]\n{user_text}\n\n"
        f"[KG_CONTEXT]\n{kg_context}\n\n"
        "[NOTE]\n"
    )

def format_kg_context(triples: List[Tuple[str,str,str,str]]) -> str:
    if not triples:
        return "(empty)"
    lines=[]
    for h, r, t, d in triples:
        lines.append(f"- ({d}) {h} --{r}--> {t}")
    return "\n".join(lines)

# -----------------------------
# batched generation (token-level slicing)
# -----------------------------
@torch.no_grad()
def generate_batch(
    model,
    tok,
    prompts: List[str],
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    do_sample = temperature > 0

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(model.device)

    # per-row true input lengths (avoid padding confusion)
    if attn is not None:
        in_lens = attn.sum(dim=1).tolist()
    else:
        # fallback: assume no padding
        in_lens = [input_ids.shape[1]] * input_ids.shape[0]

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=False,
    )

    # out: [B, S]
    if isinstance(out, (list, tuple)):
        out_ids = out[0]
    else:
        out_ids = out

    texts=[]
    for i in range(out_ids.shape[0]):
        start = int(in_lens[i])
        gen_ids = out_ids[i, start:]
        txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
        texts.append(txt)
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    # old flag kept for compatibility
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--device", default=None, help="preferred: cuda or cpu; overrides --cuda")
    ap.add_argument("--stable_fp32", action="store_true")

    ap.add_argument("--max_input_tokens", type=int, default=1536)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)

    ap.add_argument("--debug_one", action="store_true")

    # KG args (optional)
    ap.add_argument("--kg_npz", default=None)
    ap.add_argument("--kg_pkl", default=None)
    ap.add_argument("--kg_split", default="train")
    ap.add_argument("--kg_max_edges", type=int, default=800000, help="cap KG edges to avoid OOM")
    ap.add_argument("--kg_max_triples", type=int, default=24)
    args = ap.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # v100 prefers fp16; bf16 can be problematic
    dtype = torch.float32 if args.stable_fp32 or device == "cpu" else torch.float16

    kg = None
    if args.kg_npz and args.kg_pkl:
        kg = KG(args.kg_npz, args.kg_pkl, split=args.kg_split, max_edges=args.kg_max_edges)

    print("== BASELINE (LLM + KG, FREE) ==")
    print("model_dir:", os.path.abspath(args.model_dir))
    print("in_jsonl :", args.in_jsonl)
    print("out_jsonl:", args.out_jsonl)
    print("device   :", device, "dtype:", dtype)
    print("batch_size:", args.batch_size, "max_input_tokens:", args.max_input_tokens, "max_new_tokens:", args.max_new_tokens)
    print("kg_npz   :", args.kg_npz)
    print("kg_pkl   :", args.kg_pkl)
    print("kg_max_edges:", args.kg_max_edges)
    print("debug_one:", args.debug_one)

    t0=time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # IMPORTANT: avoid device_map="auto" to prevent CPU offload + RAM spikes
    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
        )
    model.eval()

    # overwrite output file
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    if os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    buf_meta=[]
    buf_prompts=[]
    total=0
    ok=0

    for row in read_jsonl_iter(args.in_jsonl):
        total += 1

        doc_id = str(row.get("doc_id") or row.get("id") or "")
        user_text = get_text(row)
        entities = extract_entities_from_row(row)

        ent_ids=[]
        triples=[]
        if kg is not None:
            ent_ids = kg.resolve_entities(entities)
            triples = kg.retrieve(ent_ids, max_triples=args.kg_max_triples)

        kg_context = format_kg_context(triples)
        prompt = build_prompt(doc_id, user_text, kg_context)
        ph = sha256_text(prompt)[:16]

        out_meta = {
            "doc_id": doc_id or row.get("doc_id") or row.get("id"),
            "ok": False,
            "notes": None,
            "model_dir": os.path.abspath(args.model_dir),
            "prompt_hash": ph,
            "kg_used": kg is not None,
            "kg_entities_in": entities,
            "kg_entity_ids": ent_ids,
            "kg_triples_n": len(triples),
            "gen_cfg": {
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "batch_size": args.batch_size,
            }
        }

        buf_meta.append(out_meta)
        buf_prompts.append(prompt)

        if args.debug_one and len(buf_prompts) >= 1:
            # force flush
            pass

        if len(buf_prompts) >= args.batch_size or args.debug_one:
            try:
                texts = generate_batch(
                    model, tok, buf_prompts,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                for m, t in zip(buf_meta, texts):
                    m["notes"] = t
                    m["ok"] = True
                    ok += 1
            except Exception as e:
                for m in buf_meta:
                    m["error"] = repr(e)
            write_jsonl_append(args.out_jsonl, buf_meta)
            buf_meta=[]
            buf_prompts=[]

            if args.debug_one:
                break

    # flush tail
    if buf_prompts:
        try:
            texts = generate_batch(
                model, tok, buf_prompts,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            for m, t in zip(buf_meta, texts):
                m["notes"] = t
                m["ok"] = True
                ok += 1
        except Exception as e:
            for m in buf_meta:
                m["error"] = repr(e)
        write_jsonl_append(args.out_jsonl, buf_meta)

    print("WROTE_JSONL:", args.out_jsonl)
    print(f"OK={ok}/{total}")
    print("elapsed_sec:", round(time.time()-t0, 3))

if __name__ == "__main__":
    main()
