# ARIS: Auditable and Uncertainty-Controlled KG Admission

This repository contains the anonymous reproducibility package for ARIS, an uncertainty-controlled framework for auditable clinical knowledge graph (KG) expansion. ARIS treats LLM-derived outputs as candidate signals rather than immediately accepted knowledge. Candidate updates must pass structural audit checks, Bayesian uncertainty routing, and provenance requirements before they are eligible to influence the KG.

## What is included

This repository includes:

- source code for cohort filtering, NER, KG construction, audit routing, Bayesian uncertainty estimation, graph reweighting, and ablation analysis;
- configuration files describing the pipeline settings;
- aggregate result CSV files corresponding to the paper tables;
- Slurm entry points for reproducing the controlled pipeline and baselines;
- documentation for data access and reproducibility.

## What is not included

This repository does not redistribute MIMIC-IV records or derived patient-level data. In particular, it does not include:

- raw MIMIC-IV tables;
- filtered patient, admission, or note-level files;
- note text;
- extracted entity spans from real notes;
- record-level routing manifests;
- model checkpoints;
- credentials, cache directories, or environment folders.

Authorized users should obtain MIMIC-IV v2.2 access directly through PhysioNet and run the provided scripts in their own local environment.

## Pipeline overview

The reproducible pipeline follows these stages:

1. Filter MIMIC-IV records using allergy-related ICD-10 criteria.
2. Train and run a Bio_ClinicalBERT token-classification model for clinical entity extraction.
3. Construct the final clinical KG from normalized entities, note/admission/patient links, co-occurrence relations, and section-aware context edges.
4. Apply learned deterministic audit rules using Aho-Corasick matching.
5. Run Bayesian uncertainty routing on fixed encoder and KG features.
6. Merge audit, uncertainty, and provenance metadata.
7. Evaluate routing coverage, audit-stress validation, uncertainty decomposition, graph reweighting, and ablation baselines.

## Expected cohort statistics

After allergy-related filtering, the expected public-data cohort statistics are:

| Quantity | Value |
|---|---:|
| Unique patients | 1,256 |
| Admissions | 7,030 |
| Discharge summaries | 7,031 |
| Extracted entity mentions | 1,432,155 |
| Final KG nodes | 63,532 |
| Final KG triples | 4,294,094 |
| Test records for controlled-routing evaluation | 2,000 |

## Result files

Aggregate results are provided under `results/`:

| File | Description |
|---|---|
| `exp1_routing_provenance.csv` | Routing, audit, uncertainty, and provenance coverage |
| `exp2_audit_gate.csv` | Synthetic audit-stress validation |
| `exp3_uncertainty.csv` | Bayesian uncertainty decomposition and low-resource stability |
| `exp4_graph_reweighting.csv` | Graph-structure preservation after uncertainty reweighting |
| `exp5_ablation.csv` | System-level ablations and baseline comparisons |

These files contain table-level aggregate results only. They do not contain record-level clinical data.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
````

Set local paths:

```bash
export ARIS_REPO=/path/to/aris-anonymous-clean
export ARIS_WORKDIR=/path/to/workdir
export ARIS_DERIVED_DATA=/path/to/local/derived/data
export HF_HOME=/path/to/huggingface/cache
```

Run the main controlled pipeline:

```bash
bash scripts/03_run_data_aggregation.sh
python scripts/04_run_audit_gate.py
python scripts/05_make_bnn_input.py
python scripts/06_run_bnn_routing.py
python scripts/08_merge_routing_outputs.py
```

Run evaluation scripts:

```bash
python scripts/11_run_operating_point_sweep.py
python scripts/12_compare_classical_ner_kg.py
python scripts/13_analyze_uncertainty.py
python scripts/14_analyze_low_resource_uncertainty.py
python scripts/19_generate_paper_tables.py
```

See `docs/data_access.md` and `docs/reproducibility.md` for details.
