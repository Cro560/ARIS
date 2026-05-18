#!/usr/bin/env python3
import os, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

MODEL_DIR = os.environ.get("MODEL_DIR")
INPUT = os.environ.get("INPUT")
OUT_DIR = os.environ.get("OUT_DIR")

MAX_LEN = 512
STRIDE = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
id2label = model.config.id2label

def predict_spans(text):
    enc = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=MAX_LEN, stride=STRIDE, return_overflowing_tokens=True
    )
    spans=[]
    for ids, attn, offsets in zip(
        enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]
    ):
        ids_t = torch.tensor([ids], device=device)
        attn_t = torch.tensor([attn], device=device)
        with torch.no_grad():
            logits = model(ids_t, attention_mask=attn_t).logits[0].cpu().numpy()
        preds = np.argmax(logits, axis=-1)

        cur=None
        for p,(s,e) in zip(preds, offsets):
            if (s,e)==(0,0): continue
            lab=id2label[int(p)]
            if lab=="O":
                if cur: spans.append(cur); cur=None
                continue
            pref, ent = lab.split("-",1)
            if pref=="B" or cur is None or cur["label"]!=ent:
                if cur: spans.append(cur)
                cur={"label":ent,"start":s,"end":e,"text":text[s:e]}
            else:
                cur["end"]=e
                cur["text"]=text[cur["start"]:cur["end"]]
        if cur: spans.append(cur)

    spans=sorted(spans,key=lambda x:(x["start"], -x["end"]))
    merged=[]
    for s in spans:
        if not merged:
            merged.append(s); continue
        last=merged[-1]
        if s["label"]==last["label"] and s["start"]<=last["end"]:
            last["end"]=max(last["end"],s["end"])
            last["text"]=text[last["start"]:last["end"]]
        else:
            merged.append(s)
    return merged

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(INPUT,"r",encoding="utf-8") as fin:
        for line in tqdm(fin):
            rec=json.loads(line)
            nid=rec["id"]; text=rec["text"]
            spans = predict_spans(text)
            out = {"id":nid, "entities":spans}
            with open(os.path.join(OUT_DIR,f"{nid}.json"),"w",encoding="utf-8") as f:
                json.dump(out,f,ensure_ascii=False,indent=2)
    print("Saved structured EHR to", OUT_DIR)

if __name__=="__main__":
    main()
