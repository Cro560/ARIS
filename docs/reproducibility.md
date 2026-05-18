# Reproducibility Guide

This document summarizes how to reproduce the ARIS pipeline in an authorized local environment.

## Environment variables

Set the following variables before running the pipeline:

```bash
export ARIS_REPO=/path/to/aris-anonymous-clean
export ARIS_WORKDIR=/path/to/workdir
export ARIS_DERIVED_DATA=/path/to/local/derived/data
export HF_HOME=/path/to/huggingface/cache
````

## Pipeline stages

### 1. Data filtering

```bash
python scripts/00_filter_mimic_allergy.py
```

This step filters MIMIC-IV records using allergy-related ICD-10 criteria.

### 2. NER training and inference

```bash
python scripts/01_train_clinical_ner.py
python scripts/02_run_ner_inference.py
```

The NER model extracts nine entity types: Drug, Dosage, Route, Frequency, Strength, Duration, Reaction, Severity, and Other.

### 3. KG construction

```bash
bash scripts/03_run_data_aggregation.sh
```

The final KG is constructed from normalized entities, note/admission/patient links, co-occurrence edges, section-aware context edges, and structured annotations.

### 4. Audit and routing

```bash
python scripts/04_run_audit_gate.py
python scripts/05_make_bnn_input.py
python scripts/06_run_bnn_routing.py
python scripts/07_postprocess_bnn_outputs.py
python scripts/08_merge_routing_outputs.py
```

This stage applies deterministic audit checks, Bayesian uncertainty routing, and provenance-preserving output merging.

### 5. Full controlled pipeline

```bash
python scripts/09_run_full_controlled_pipeline.py
```

Slurm entry points are also provided under `slurm/`.

### 6. Baselines and ablations

```bash
python scripts/10_run_unconstrained_llm_baseline.py
python scripts/11_run_operating_point_sweep.py
python scripts/12_compare_classical_ner_kg.py
```

### 7. Uncertainty and graph analyses

```bash
python scripts/13_analyze_uncertainty.py
python scripts/14_analyze_low_resource_uncertainty.py
python scripts/20_train_reweighted_encoder.py
python scripts/21_train_weighted_encoder.py
```

### 8. Audit-stress validation

```bash
python scripts/15_run_audit_stress_validation.py
python scripts/17_analyze_audit_performance.py
python scripts/18_aggregate_audit_results.py
python scripts/19_generate_paper_tables.py
```

## Expected aggregate outputs

The released aggregate results are stored under `results/`. Each CSV uses the schema:

```text
experiment,setting,metric,value,std,n,notes
```

These files reproduce the table-level claims in the paper without exposing record-level clinical data.

## Notes

Absolute values may vary if users retrain NER, change KG construction settings, alter BNN random seeds, or use different local preprocessing. The paper reports results from the fixed configuration summarized in `configs/pipeline.yaml`.
