# Data Access

ARIS is evaluated using MIMIC-IV v2.2, a credentialed-access clinical dataset. This repository does not redistribute raw MIMIC-IV files or derived patient-level data.

## Required data

Authorized users should obtain MIMIC-IV v2.2 access through PhysioNet and place the required tables under the local path specified by their environment variables and configuration files.

The pipeline expects local access to MIMIC-IV diagnosis, admission, patient, discharge-note, laboratory, microbiology, and prescription-related tables as required by the preprocessing scripts.

## Allergy-related cohort filtering

The cohort is constructed by filtering records with allergy-related ICD-10 criteria, including diagnosis codes corresponding to allergic reactions, anaphylaxis, allergy status, adverse effects, and drug/food/environment-related hypersensitivity.

The filtering and preprocessing logic is implemented in:

```text
scripts/00_filter_mimic_allergy.py
src/aris/data/
configs/pipeline.yaml
````

Expected aggregate statistics after filtering are:

| Quantity            | Value |
| ------------------- | ----: |
| Unique patients     | 1,256 |
| Admissions          | 7,030 |
| Discharge summaries | 7,031 |

## Release policy

This repository releases code, configuration files, Slurm entry points, documentation, and aggregate result tables.

It does not release:

* raw MIMIC-IV tables;
* filtered patient-level or note-level files;
* real note text;
* real entity spans;
* subject, admission, or note identifier mappings;
* full record-level routing manifests;
* model checkpoints or cache directories.

This design allows reproducibility for authorized users while respecting clinical data-use restrictions.
