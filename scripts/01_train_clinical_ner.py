#!/usr/bin/env python3
import os, json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import evaluate

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

TRAIN_FILE = os.environ.get("TRAIN_FILE")
VAL_FILE   = os.environ.get("VAL_FILE")
TEST_FILE  = os.environ.get("TEST_FILE")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")

MAX_LEN = 512

LABEL_LIST = ["O","B-MED","I-MED","B-LAB","I-LAB","B-DIAG","I-DIAG"]
label2id = {l:i for i,l in enumerate(LABEL_LIST)}
id2label = {i:l for l,i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
seqeval = evaluate.load("seqeval")

def load_jsonl(p):
    out=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f: out.append(json.loads(line))
    return out

def spans_to_word_labels(text, spans):
    words=[]; labels=[]
    idx=0
    for w in text.split():
        s=text.find(w, idx); e=s+len(w); idx=e
        words.append(w)
        lab="O"
        for sp in spans:
            if sp["start"] < e and sp["end"] > s:
                base=sp["label"].upper()
                if base not in ["MED","LAB","DIAG"]: continue
                lab="B-"+base
                break
        labels.append(lab)
    return words, labels

def prepare(records):
    items=[]
    for r in records:
        words, labels = spans_to_word_labels(r["text"], r.get("spans",[]))
        items.append({"words":words, "labels":labels})
    return Dataset.from_list(items)

def tokenize_align(batch):
    tok = tokenizer(batch["words"], is_split_into_words=True,
                    truncation=True, max_length=MAX_LEN)
    out_labels=[]
    for i, labs in enumerate(batch["labels"]):
        word_ids = tok.word_ids(batch_index=i)
        prev=None; ids=[]
        for widx in word_ids:
            if widx is None:
                ids.append(-100)
            elif widx != prev:
                l=labs[widx]
                ids.append(label2id.get(l,0))
            else:
                l=labs[widx]
                if l.startswith("B-"): l="I-"+l[2:]
                ids.append(label2id.get(l,0))
            prev=widx
        out_labels.append(ids)
    tok["labels"]=out_labels
    return tok

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    true_preds=[]; true_labels=[]
    for pr, lb in zip(preds, labels):
        tp=[]; tl=[]
        for p_i, l_i in zip(pr, lb):
            if l_i==-100: continue
            tp.append(id2label[p_i])
            tl.append(id2label[l_i])
        true_preds.append(tp); true_labels.append(tl)
    res = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision":res["overall_precision"],
        "recall":res["overall_recall"],
        "f1":res["overall_f1"]
    }

def main():
    train_ds = prepare(load_jsonl(TRAIN_FILE)).map(tokenize_align, batched=True, remove_columns=["words","labels"])
    val_ds   = prepare(load_jsonl(VAL_FILE)).map(tokenize_align, batched=True, remove_columns=["words","labels"])
    test_ds  = prepare(load_jsonl(TEST_FILE)).map(tokenize_align, batched=True, remove_columns=["words","labels"])

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",           # 这里改名
        eval_steps=300,
        save_strategy="steps",           # 建议显式写一下，和 eval 对齐
        save_steps=300,
        logging_steps=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        learning_rate=3e-5,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(3)]
    )

    trainer.train()
    print("Test:", trainer.evaluate(test_ds))
    trainer.save_model(OUTPUT_DIR)

if __name__=="__main__":
    main()
