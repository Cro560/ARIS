#!/usr/bin/env python3
import gzip, csv, sys, os

if len(sys.argv) != 4:
    print("Usage: python3 fast_filter.py <input.csv.gz> <keyfile.csv.gz> <output.csv.gz>")
    sys.exit(1)

INPUT = sys.argv[1]
KEYS = sys.argv[2]
OUTPUT = sys.argv[3]

# Load keys (very fast even for millions)
keys = set()
with gzip.open(KEYS, 'rt') as f:
    r = csv.DictReader(f)
    first_col = r.fieldnames[0]
    for row in r:
        keys.add(row[first_col])

print(f"[INFO] Loaded {len(keys):,} keys from {KEYS}")

# Filter input in streaming mode (no memory usage)
with gzip.open(INPUT, 'rt') as fin, gzip.open(OUTPUT, 'wt') as fout:
    r = csv.DictReader(fin)
    w = csv.DictWriter(fout, fieldnames=r.fieldnames)
    w.writeheader()

    id_col = r.fieldnames[0]  # assumes first column is subject_id or hadm_id
    count = 0
    kept = 0

    for row in r:
        count += 1
        if row[id_col] in keys:
            w.writerow(row)
            kept += 1
        if count % 1_000_000 == 0:
            print(f"[INFO] Processed {count:,} rows... kept {kept:,}")

print(f"[DONE] Finished. Kept {kept:,} rows out of {count:,}. Output: {OUTPUT}")
