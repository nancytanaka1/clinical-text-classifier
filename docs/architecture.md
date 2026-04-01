# Architecture

## Goal

This repo separates exploratory notebook work from production execution in Azure Databricks.

## Layers

- `notebooks/`: EDA, one-off analysis, and early experimentation
- `src/clinical_text_classifier/`: reusable application logic that can be tested locally and executed in Databricks jobs
- `databricks.yml`: Databricks Asset Bundle definition for deployable jobs
- `configs/*.yml`: environment overrides for dev, staging, and prod
- `.github/workflows/ci.yml`: CI pipeline for lint, test, build, and bundle validation

## Execution Model

1. Analysts prototype in notebooks.
2. Stable logic is promoted into the Python package.
3. CI builds a wheel from the package.
4. Databricks jobs execute wheel entry points for data prep, model training, and smoke checks.
5. Environment-specific overrides control schedules, cluster sizing, and deployment behavior.

## Why This Is More Production Ready

- business logic is versioned as Python modules, not hidden inside notebooks
- jobs are deployable and repeatable through the bundle
- local tests can validate data logic without Databricks
- CI catches packaging and bundle issues before deployment
