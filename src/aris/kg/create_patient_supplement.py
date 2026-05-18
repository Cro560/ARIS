#!/usr/bin/env python3
import os, gzip, csv, math, json
from collections import defaultdict, Counter
from datetime import datetime

OUTDIR = os.path.abspath(os.environ.get("OUTDIR", "${ARIS_WORKDIR}/agg_out"))
LAB_IN = os.path.join(OUTDIR, "filtered_labevents.csv.gz")
MICRO_IN = os.path.join(OUTDIR, "filtered_microbiologyevents.csv.gz")
POE_IN = os.path.join(OUTDIR, "filtered_poe.csv.gz")
POE_DETAIL_IN = os.path.join(OUTDIR, "filtered_poe_detail.csv.gz")

# Output files
SUPP_OUT = os.path.join(OUTDIR, "patient_supplement.csv")
LAB_PAIR_OUT = os.path.join(OUTDIR, "patient_lab_item_stats.csv.gz")

# Lab numeric sanity thresholds (tunable)
ABS_MAX = 1e6
ABS_MIN = -1e5

def try_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    # quickly reject non-numeric markers that contain letters (but allow scientific notation)
    if any(ch.isalpha() for ch in s) and not ('e' in s.lower()):
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or v > ABS_MAX or v < ABS_MIN:
        return None
    return v

def parse_time(s):
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    # try iso and common formats
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except Exception:
            continue
    # fallback try fromisoformat
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

# Welford aggregator
class Welford:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None
    def add(self, x):
        if x is None:
            return
        self.n += 1
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    def std(self):
        if self.n <= 0:
            return None
        return math.sqrt(self.M2 / self.n)  # population std

# containers
# per subject overall labs
lab_w = defaultdict(Welford)          # subject -> Welford (numeric valuenum)
lab_total_count = Counter()           # subject -> total lab rows (including non-numeric)
lab_numeric_count = Counter()         # subject -> numeric valuenum rows
lab_item_counts = defaultdict(Counter) # subject -> Counter(itemid->count)
lab_first = {}                         # subject -> datetime
lab_last = {}

# per (subject,item) pair aggregator (for output)
from collections import defaultdict as _dd
pair_w = _dd(lambda: Welford())
pair_count_raw = Counter()
pair_count_numeric = Counter()
pair_first = {}
pair_last = {}

# microbiology
micro_count = Counter()
micro_terms = defaultdict(Counter)

# POE
poe_count = Counter()
poe_detail_count = Counter()
poe_cats = defaultdict(Counter)
poe_first = {}
poe_last = {}

# set of subjects seen
subjects = set()

# -------------------------------------------------------
# Process labs (stream)
# -------------------------------------------------------
if os.path.exists(LAB_IN):
    print("[STEP] processing labs:", LAB_IN)
    with gzip.open(LAB_IN, "rt") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        sid_col = next((c for c in headers if c.lower()=="subject_id"), None)
        item_col = next((c for c in headers if c.lower()=="itemid"), None)
        val_col = next((c for c in headers if c.lower()=="valuenum"), None)
        time_col = next((c for c in headers if "charttime" in c.lower() or "chart_time" in c.lower()), None)
        if sid_col is None or item_col is None:
            raise SystemExit("[ERROR] lab file missing required columns (subject_id,itemid)")
        total_rows = 0
        for r in reader:
            total_rows += 1
            sid = (r.get(sid_col) or "").strip()
            item = (r.get(item_col) or "").strip()
            if not sid or not item:
                continue
            subjects.add(sid)
            lab_total_count[sid] += 1
            lab_item_counts[sid][item] += 1
            key = (sid, item)
            pair_count_raw[key] += 1
            # numeric
            num = None
            if val_col and r.get(val_col) is not None:
                num = try_float(r.get(val_col))
            if num is not None:
                lab_numeric_count[sid] += 1
                lab_w[sid].add(num)
                pair_w[key].add(num)
                pair_count_numeric[key] += 1
            # times
            if time_col and r.get(time_col):
                dt = parse_time(r.get(time_col))
                if dt:
                    # per-subject
                    if sid not in lab_first or dt < lab_first[sid]:
                        lab_first[sid] = dt
                    if sid not in lab_last or dt > lab_last[sid]:
                        lab_last[sid] = dt
                    # per-pair
                    if key not in pair_first or dt < pair_first[key]:
                        pair_first[key] = dt
                    if key not in pair_last or dt > pair_last[key]:
                        pair_last[key] = dt
    print(f"[INFO] labs rows processed: {total_rows}, subjects seen (labs): {len(subjects)}")
else:
    print("[SKIP] lab input not found:", LAB_IN)

# -------------------------------------------------------
# Process microbiology
# -------------------------------------------------------
if os.path.exists(MICRO_IN):
    print("[STEP] processing microbiology:", MICRO_IN)
    with gzip.open(MICRO_IN, "rt") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        sid_col = next((c for c in headers if c.lower()=="subject_id"), None)
        # pick organism/specimen column heuristically
        cand = next((c for c in headers if "organ" in c.lower() or "org" in c.lower() or "spec" in c.lower()), headers[-1])
        if sid_col is None:
            raise SystemExit("[ERROR] microbiology missing subject_id")
        total = 0
        for r in reader:
            total += 1
            sid = (r.get(sid_col) or "").strip()
            if not sid:
                continue
            subjects.add(sid)
            micro_count[sid] += 1
            val = (r.get(cand) or "").strip()
            if val:
                micro_terms[sid][val] += 1
    print(f"[INFO] micro rows processed: {total}")
else:
    print("[SKIP] microbiology input not found:", MICRO_IN)

# -------------------------------------------------------
# Process POE and POE_DETAIL
# -------------------------------------------------------
def process_poe_file(path, is_detail=False):
    if not os.path.exists(path):
        print("[SKIP] poe file not found:", path)
        return 0
    total = 0
    with gzip.open(path, "rt") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        sid_col = next((c for c in headers if c.lower()=="subject_id"), None)
        cat_col = next((c for c in headers if "order_type" in c.lower() or "category" in c.lower() or "order_cat" in c.lower() or "ordertype" in c.lower()), None)
        time_col = next((c for c in headers if "time" in c.lower() or "date" in c.lower()), None)
        if sid_col is None:
            raise SystemExit("[ERROR] poe file missing subject_id: " + path)
        for r in reader:
            total += 1
            sid = (r.get(sid_col) or "").strip()
            if not sid:
                continue
            subjects.add(sid)
            if is_detail:
                poe_detail_count[sid] += 1
            else:
                poe_count[sid] += 1
            if cat_col and r.get(cat_col):
                poe_cats[sid][ (r.get(cat_col) or "").strip() ] += 1
            if time_col and r.get(time_col):
                dt = parse_time(r.get(time_col))
                if dt:
                    if sid not in poe_first or dt < poe_first[sid]:
                        poe_first[sid] = dt
                    if sid not in poe_last or dt > poe_last[sid]:
                        poe_last[sid] = dt
    return total

t1 = process_poe_file(POE_IN, is_detail=False)
t2 = process_poe_file(POE_DETAIL_IN, is_detail=True)
print(f"[INFO] poe rows processed: {t1}, poe_detail rows processed: {t2}")

# -------------------------------------------------------
# Write per-pair item stats (compressed) for possible later per-item normalization
# -------------------------------------------------------
print("[STEP] writing per-pair lab stats to", LAB_PAIR_OUT)
with gzip.open(LAB_PAIR_OUT, "wt") as fh:
    w = csv.writer(fh)
    w.writerow(["subject_id","itemid","count_raw","count_numeric","mean","std","min","max","first_time","last_time"])
    for (sid,item), welf in sorted(pair_w.items(), key=lambda x: (int(x[0][0]) if x[0][0].isdigit() else x[0][0], x[0][1])):
        cr = pair_count_raw[(sid,item)]
        cn = pair_count_numeric[(sid,item)]
        mean = welf.mean if welf.n>0 else ""
        std = welf.std() if welf.n>0 else ""
        mn = welf.min if welf.n>0 else ""
        mx = welf.max if welf.n>0 else ""
        ft = pair_first.get((sid,item))
        lt = pair_last.get((sid,item))
        fh.writerow([sid, item, cr, cn, mean, std, mn, mx, ft.isoformat() if ft else "", lt.isoformat() if lt else ""])
print("[DONE] per-pair lab stats written.")

# -------------------------------------------------------
# Assemble patient_supplement.csv
# -------------------------------------------------------
print("[STEP] writing supplement CSV to", SUPP_OUT)
headers = [
    "subject_id",
    # lab-level
    "lab_total_count","lab_numeric_count","lab_mean","lab_std","lab_min","lab_max","lab_distinct_item_count","lab_top_items_json","lab_first_time","lab_last_time",
    # micro
    "micro_count","micro_unique_count","micro_top_terms_json",
    # poe
    "poe_count","poe_detail_count","poe_unique_cat_count","poe_top_cats_json","poe_first_time","poe_last_time"
]

with open(SUPP_OUT, "w", newline='') as fh:
    w = csv.writer(fh)
    w.writerow(headers)
    for sid in sorted(subjects, key=lambda x: int(x) if x.isdigit() else x):
        lw = lab_w.get(sid)
        if lw and lw.n>0:
            lab_mean = lw.mean
            lab_std = lw.std()
            lab_min = lw.min
            lab_max = lw.max
            lab_num = lw.n
        else:
            lab_mean = lab_std = lab_min = lab_max = lab_num = ""
        lt_count = lab_total_count.get(sid, 0)
        distinct_items = len(lab_item_counts.get(sid, {}))
        top_items = lab_item_counts.get(sid, {}).most_common(3)
        # micro
        mc = micro_count.get(sid, 0)
        mu = len(micro_terms.get(sid, {}))
        top_micro = micro_terms.get(sid, {}).most_common(5)
        # poe
        pc = poe_count.get(sid, 0)
        pdet = poe_detail_count.get(sid, 0)
        puc = len(poe_cats.get(sid, {}))
        top_poe = poe_cats.get(sid, {}).most_common(5)
        # times
        lfirst = lab_first.get(sid)
        llast = lab_last.get(sid)
        pfirst = poe_first.get(sid)
        plast = poe_last.get(sid)

        w.writerow([
            sid,
            lt_count, lab_num, lab_mean, lab_std, lab_min, lab_max, distinct_items, json.dumps(top_items), (lfirst.isoformat() if lfirst else ""), (llast.isoformat() if llast else ""),
            mc, mu, json.dumps(top_micro),
            pc, pdet, puc, json.dumps(top_poe), (pfirst.isoformat() if pfirst else ""), (plast.isoformat() if plast else "")
        ])
print("[DONE] patient supplement CSV written:", SUPP_OUT)

# print brief diagnostics
total_subjects = len(subjects)
print(f"[SUMMARY] subjects={total_subjects}, lab_total_rows_by_subject_sum={sum(lab_total_count.values())}, lab_numeric_rows_sum={sum(lab_numeric_count.values())}")
