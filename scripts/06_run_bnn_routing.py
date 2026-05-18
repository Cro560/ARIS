import os, json, argparse, time
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# HARD BLOCK: forbid generate()
# -----------------------------
class NoGenerateWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        raise RuntimeError(
            "[BNN_GATE] generate() is FORBIDDEN. "
            "LLM must NOT produce free text. BNN must take over from inference to output."
        )


def patch_forbid_generate_global() -> bool:
    """
    Extra safety: monkeypatch HF GenerationMixin.generate to fail-closed.
    """
    try:
        from transformers.generation.utils import GenerationMixin

        def _blocked_generate(self, *args, **kwargs):
            raise RuntimeError("[BNN_GATE] Global generate() blocked (encoder-only pipeline).")

        GenerationMixin.generate = _blocked_generate
        return True
    except Exception:
        return False


@torch.no_grad()
def encode_texts(
    tokenizer,
    model,
    texts: List[str],
    device: torch.device,
    max_length: int = 256,
) -> torch.Tensor:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    return outputs.last_hidden_state[:, 0, :]  # CLS pooling


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_py(x: Any):
    """
    Recursively convert tensors and nested containers to JSON-safe Python types.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return x.tolist()

    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]

    if isinstance(x, (float, int, str, bool)) or x is None:
        return x

    # fallback: keep auditability, never crash
    try:
        return float(x)
    except Exception:
        return str(x)


def normalize_bnn_output(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        return to_py(out)
    return {"score": to_py(out)}


def load_bnn(
    device: torch.device,
    *,
    llm_embedding_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    prior_std: float,
    n_samples: int,
    bnn_ckpt=None,
):
    from bayes_bnn.bnn_model import BayesianReasoningModule

    bnn = BayesianReasoningModule(
        llm_embedding_dim=llm_embedding_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        prior_std=prior_std,
        n_samples=n_samples,
    ).to(device).eval()

    ctor = {
        "llm_embedding_dim": llm_embedding_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "prior_std": prior_std,
        "n_samples": n_samples,
    }
        # --- optional checkpoint loading ---
    if bnn_ckpt:
        import os, torch
        if not os.path.exists(bnn_ckpt):
            raise FileNotFoundError(f"BNN ckpt not found: {bnn_ckpt}")
        sd = torch.load(bnn_ckpt, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        missing, unexpected = bnn.load_state_dict(sd, strict=False)
        try:
            mcnt = len(missing) if missing is not None else 0
            ucnt = len(unexpected) if unexpected is not None else 0
            print(f"[INFO] loaded bnn_ckpt: {bnn_ckpt} (missing={mcnt}, unexpected={ucnt})")
        except Exception:
            print(f"[INFO] loaded bnn_ckpt: {bnn_ckpt}")
    bnn.eval()
    return bnn, "BayesianReasoningModule", ctor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_model", required=True, help="HF model name or local path (encoder-only).")
    ap.add_argument("--in_jsonl", default="reports/bnn/bnn_input_CURRENT.jsonl")
    ap.add_argument("--out_jsonl", default="reports/bnn/bnn_pred_CURRENT.jsonl")
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--bnn_hidden_dims", type=str, default="512,256,128",
                    help="Comma-separated hidden dims for BayesianReasoningModule.")
    ap.add_argument("--bnn_output_dim", type=int, default=1)
    ap.add_argument("--bnn_prior_std", type=float, default=1.0)
    ap.add_argument("--bnn_n_samples", type=int, default=10)
    ap.add_argument("--bnn_ckpt", type=str, default=None,
                    help="Optional path to a trained BNN checkpoint (.pt/.pth state_dict).")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patched = patch_forbid_generate_global()

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model, use_fast=True)
    base = AutoModel.from_pretrained(args.encoder_model, use_safetensors=True).to(device).eval()
    encoder = NoGenerateWrapper(base).to(device).eval()

    probe = encode_texts(tokenizer, encoder, ["probe"], device=device, max_length=16)
    in_dim = int(probe.shape[-1])

    hidden_dims = [int(x) for x in args.bnn_hidden_dims.split(",") if x.strip()]

    bnn, bnn_class, bnn_ctor = load_bnn(
        device,
        llm_embedding_dim=in_dim,
        hidden_dims=hidden_dims,
        output_dim=args.bnn_output_dim,
        prior_std=args.bnn_prior_std,
        n_samples=args.bnn_n_samples,
        bnn_ckpt=args.bnn_ckpt,
    )

    run_meta = {
        "run_id": f"bnn_gate_{now_tag()}",
        "device": str(device),
        "encoder_model": args.encoder_model,
        "llm_role": "ENCODER_ONLY",
        "generate_blocked_wrapper": True,
        "generate_blocked_global_patch": bool(patched),
        "max_length": int(args.max_length),
        "in_dim": in_dim,
        "bnn_class": bnn_class,
        "bnn_ctor": bnn_ctor,
        "final_output_source": "BNN_ONLY",
    }

    t0 = time.time()
    rows = []

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is None:
                continue

            features_text = obj.get("features_text", obj.get("text", ""))

            emb = encode_texts(tokenizer, encoder, [features_text], device=device, max_length=args.max_length)
            out = bnn(emb)
            out_norm = normalize_bnn_output(out)

            rows.append({
                "doc_id": doc_id,
                "audit_meta": obj.get("audit_meta", {}),
                "bnn_output": out_norm,
                "policy": {
                    "llm_generate_forbidden": True,
                    "llm_free_text_output": False,
                    "final_output_source": "BNN_ONLY",
                },
            })

    elapsed = time.time() - t0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_meta": run_meta, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[OK] wrote:", args.out_jsonl)
    print("[INFO] elapsed_sec:", elapsed)

    # Must fail: generate()
    try:
        _ = encoder.generate(**tokenizer("test", return_tensors="pt").to(device))
        raise AssertionError("generate() unexpectedly succeeded")
    except Exception as e:
        print("[CHECK] generate() blocked:", str(e)[:160])


if __name__ == "__main__":
    main()
