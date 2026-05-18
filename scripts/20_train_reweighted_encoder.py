import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

DATA_PATH = "weighted_ner_data_with_weight.parquet"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUT_DIR = "clinicalbert_reweighted_cls"

class NerRowDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self): return len(self.df)

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
        return item

class WeightedTrainer(Trainer):
    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

def main():
    df = pd.read_parquet(DATA_PATH)
    need = ["section_norm","entity_text","entity_label","weight"]
    df = df[need].dropna()
    df["entity_label"] = df["entity_label"].astype(str)
    df["weight"] = df["weight"].astype("float32")

    labels = sorted(df["entity_label"].unique().tolist())
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    n = len(df)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    cut = int(n * 0.98)

    train_df = df.iloc[idx[:cut]].copy()
    eval_df  = df.iloc[idx[cut:]].copy()

    # 关键：证明 reweighting “真的被用上”
    print("[REWEIGHT CHECK] train weight describe:")
    print(train_df["weight"].describe().to_string())
    print("[REWEIGHT CHECK] top weights:")
    print(train_df["weight"].value_counts().head(10).to_string())

    sampler = WeightedRandomSampler(
        weights=torch.tensor(train_df["weight"].values, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True
    )

    train_ds = NerRowDataset(train_df.drop(columns=["weight"]), tok, label2id)
    eval_ds  = NerRowDataset(eval_df.drop(columns=["weight"]), tok, label2id)

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
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        train_sampler=sampler,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Saved:", OUT_DIR)

if __name__ == "__main__":
    main()
