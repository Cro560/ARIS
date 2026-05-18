import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# IMPORTANT: use the row-level parquet that contains sample_weight
DATA_PATH = "weighted_ner_data_with_weights.parquet"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUT_DIR = "clinicalbert_weighted_cls"

class NerRowDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        text = f"{r['section_norm']}: {r['entity_text']}"
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        y = self.label2id[str(r["entity_label"])]
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        # keep weight for sampler
        item["sample_weight"] = torch.tensor(float(r["sample_weight"]), dtype=torch.float)
        return item

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WeightedTrainer(Trainer):
    """
    Use WeightedRandomSampler via overriding get_train_dataloader.
    This is mild reweighting: it changes sampling frequency, not the loss scale.
    """
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        assert train_dataset is not None

        # extract weights from dataset df in the same order
        w = train_dataset.df["sample_weight"].astype("float64").values
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
        # safety clip
        w = np.clip(w, 0.25, 4.0)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(w, dtype=torch.double),
            num_samples=len(w),
            replacement=True,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True,
        )

def main():
    seed_everything(42)

    df = pd.read_parquet(DATA_PATH)
    # keep necessary columns
    need = ["section_norm", "entity_text", "entity_label", "sample_weight"]
    df = df[need].dropna()
    df["entity_label"] = df["entity_label"].astype(str)
    df["sample_weight"] = df["sample_weight"].astype("float64")

    labels = sorted(df["entity_label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # split (same as before)
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * 0.98)
    train_df = df.iloc[idx[:cut]]
    eval_df  = df.iloc[idx[cut:]]

    train_ds = NerRowDataset(train_df, tok, label2id)
    eval_ds  = NerRowDataset(eval_df, tok, label2id)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Saved:", OUT_DIR)

if __name__ == "__main__":
    main()
