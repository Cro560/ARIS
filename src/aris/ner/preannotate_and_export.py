#!/usr/bin/env python3
import re, json, os
def tqdm(x, **kwargs):
    return x

INPUT = "${ARIS_WORKDIR}/ehr_pipeline/notes.jsonl"
OUTPUT = "${ARIS_WORKDIR}/ehr_pipeline/pre_annotated.jsonl"

# 你可以不断扩展这些 patterns
SIMPLE_MEDNAME = re.compile(
    r"\b(?:warfarin|aspirin|digoxin|rosuvastatin|metoprolol|lisinopril|"
    r"citalopram|levofloxacin|vancomycin|cefazolin|allopurinol|colchicine|"
    r"apixaban|furosemide|mirtazapine|heparin|ticagrelor|imdur|isosorbide)\b",
    flags=re.IGNORECASE
)

MED_DOSE = re.compile(
    r"([A-Za-z][A-Za-z0-9\-\(\)\/\s]{1,50}?)\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|tab|tabs|tablet|PAK|pack)?)",
    flags=re.IGNORECASE
)

LAB_PATTERN = re.compile(
    r"\b(WBC|HGB|Hgb|Hct|Creatinine|Cr|K|Na|Glucose|Platelet|PLT|Tbili|ALT|AST|"
    r"ALK PHOS|ALP|Albumin|BNP|Troponin|Lactate)\b[^\n]{0,40}?\-?\s*:?(\d+\.?\d*)",
    flags=re.IGNORECASE
)

DIAG_KEYWORDS = [
    "gout","pneumonia","acute kidney injury","AKI","heart failure","HFrEF","NSTEMI",
    "cardiogenic shock","dehydration","weakness","encephalopathy","cholelithiasis",
    "atrial fibrillation","coronary artery disease"
]

def extract_spans(text):
    spans=[]
    for m in SIMPLE_MEDNAME.finditer(text):
        spans.append({"label":"MED","start":m.start(),"end":m.end(),"text":m.group(0)})

    for m in MED_DOSE.finditer(text):
        s = m.start(1); e = m.end(2)
        spans.append({"label":"MED","start":s,"end":e,"text":text[s:e]})

    for m in LAB_PATTERN.finditer(text):
        spans.append({"label":"LAB","start":m.start(),"end":m.end(),"text":text[m.start():m.end()]})

    for kw in DIAG_KEYWORDS:
        for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
            spans.append({"label":"DIAG","start":m.start(),"end":m.end(),"text":m.group(0)})

    # merge overlap
    spans = sorted(spans, key=lambda x:(x["start"], -x["end"]))
    merged=[]
    for s in spans:
        if not merged:
            merged.append(s); continue
        last=merged[-1]
        if s["start"] <= last["end"] and s["label"]==last["label"]:
            last["end"]=max(last["end"], s["end"])
            last["text"]=text[last["start"]:last["end"]]
        else:
            merged.append(s)
    return merged

def main():
    with open(INPUT,"r",encoding="utf-8") as fin, open(OUTPUT,"w",encoding="utf-8") as fout:
        for line in tqdm(fin):
            rec=json.loads(line)
            nid=rec["id"]; text=rec["text"]
            spans=extract_spans(text)
            fout.write(json.dumps({"id":nid,"text":text,"spans":spans}, ensure_ascii=False)+"\n")
    print("Wrote pre-annotated ->", OUTPUT)

if __name__=="__main__":
    main()
