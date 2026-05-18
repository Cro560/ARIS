#!/usr/bin/env python3
import gzip, csv, os, sys
from collections import defaultdict

DATA_DIR = "${ARIS_WORKDIR}/agg_out"
OUTFILE = os.path.join(DATA_DIR, "patient_features.csv")

# Load subject_id universe from any file that contains subject_id
subject_ids = set()

def load_subject_ids(path):
    try:
        with gzip.open(path, 'rt') as f:
            hdr = next(f).strip().split(',')
            if 'subject_id' not in hdr:
                return
            idx = hdr.index('subject_id')
            for row in f:
                parts = row.rstrip('\n').split(',')
                if len(parts) > idx:
                    subject_ids.add(parts[idx])
    except:
        pass

# collect subject_ids
for fn in [
    "filtered_labevents.csv.gz",
    "filtered_microbiologyevents.csv.gz",
    "filtered_poe.csv.gz",
    "filtered_poe_detail.csv.gz",
]:
    load_subject_ids(os.path.join(DATA_DIR, fn))

if not subject_ids:
    print("ERROR: no subject_ids found in any file", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] Loaded {len(subject_ids)} subject_ids")

# ------------------------------
# Feature containers
# ------------------------------
lab_count = defaultdict(int)
lab_unique_items = defaultdict(set)

micro_count = defaultdict(int)
micro_orgs = defaultdict(set)

poe_count = defaultdict(int)
poe_orders = defaultdict(set)

poed_count = defaultdict(int)
poed_items = defaultdict(set)

# ------------------------------
# Process labevents
# ------------------------------
lab_path = os.path.join(DATA_DIR, "filtered_labevents.csv.gz")
if os.path.exists(lab_path):
    print("[STEP] Scanning labevents...")
    with gzip.open(lab_path, 'rt') as f:
        hdr = next(f).strip().split(',')
        if 'subject_id' in hdr:
            sid_i = hdr.index('subject_id')
            item_i = hdr.index('itemid') if 'itemid' in hdr else None
            for row in f:
                parts = row.rstrip('\n').split(',')
                if len(parts) <= sid_i: continue
                sid = parts[sid_i]
                if sid not in subject_ids: continue
                lab_count[sid] += 1
                if item_i is not None and len(parts) > item_i:
                    lab_unique_items[sid].add(parts[item_i])
else:
    print("[WARN] labevents file missing")

# ------------------------------
# Process microbiology
# ------------------------------
micro_path = os.path.join(DATA_DIR, "filtered_microbiologyevents.csv.gz")
if os.path.exists(micro_path):
    print("[STEP] Scanning microbiology...")
    with gzip.open(micro_path, 'rt') as f:
        hdr = next(f).strip().split(',')
        if 'subject_id' in hdr:
            sid_i = hdr.index('subject_id')
            org_i = hdr.index('org_name') if 'org_name' in hdr else None
            for row in f:
                parts = row.rstrip('\n').split(',')
                if len(parts) <= sid_i: continue
                sid = parts[sid_i]
                if sid not in subject_ids: continue
                micro_count[sid] += 1
                if org_i is not None and len(parts) > org_i:
                    micro_orgs[sid].add(parts[org_i])
else:
    print("[WARN] microbiologyevents missing")

# ------------------------------
# Process POE
# ------------------------------
poe_path = os.path.join(DATA_DIR, "filtered_poe.csv.gz")
if os.path.exists(poe_path):
    print("[STEP] Scanning POE...")
    with gzip.open(poe_path, 'rt') as f:
        hdr = next(f).strip().split(',')
        if 'subject_id' in hdr:
            sid_i = hdr.index('subject_id')
            code_i = hdr.index('order_code') if 'order_code' in hdr else None
            for row in f:
                parts = row.rstrip('\n').split(',')
                if len(parts) <= sid_i: continue
                sid = parts[sid_i]
                if sid not in subject_ids: continue
                poe_count[sid] += 1
                if code_i is not None and len(parts) > code_i:
                    poe_orders[sid].add(parts[code_i])
else:
    print("[WARN] poe missing")

# ------------------------------
# Process POE_DETAIL
# ------------------------------
poed_path = os.path.join(DATA_DIR, "filtered_poe_detail.csv.gz")
if os.path.exists(poed_path):
    print("[STEP] Scanning POE_DETAIL...")
    with gzip.open(poed_path, 'rt') as f:
        hdr = next(f).strip().split(',')
        if 'subject_id' in hdr:
            sid_i = hdr.index('subject_id')
            item_i = hdr.index('order_name') if 'order_name' in hdr else None
            for row in f:
                parts = row.rstrip('\n').split(',')
                if len(parts) <= sid_i: continue
                sid = parts[sid_i]
                if sid not in subject_ids: continue
                poed_count[sid] += 1
                if item_i is not None and len(parts) > item_i:
                    poed_items[sid].add(parts[item_i])
else:
    print("[WARN] poe_detail missing")

# ------------------------------
# Write output
# ------------------------------
print("[STEP] Writing patient_features.csv")
with open(OUTFILE, 'w') as out:
    out.write("subject_id,lab_count,lab_unique_items,micro_count,micro_orgs,poe_count,poe_unique_orders,poe_detail_count,poe_detail_unique_items\n")
    for sid in sorted(subject_ids, key=lambda x: int(x)):
        out.write(",".join([
            sid,
            str(lab_count[sid]),
            str(len(lab_unique_items[sid])),
            str(micro_count[sid]),
            str(len(micro_orgs[sid])),
            str(poe_count[sid]),
            str(len(poe_orders[sid])),
            str(poed_count[sid]),
            str(len(poed_items[sid])),
        ]) + "\n")

print(f"[DONE] Created: {OUTFILE}")
