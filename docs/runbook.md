# Runbook

## Local Development

```bash
pip install -e .[dev]
pytest
clinical-prepare-data --config configs/config.yaml
clinical-train-baseline --config configs/config.yaml
```

## Databricks Deployment

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev prepare_data_job
databricks bundle run -t dev training_job
```

## Promotion Path

1. Merge to `dev` to validate packaging and deployment.
2. Promote to `main` after staging confidence is high.
3. Let prod schedules trigger governed jobs.

## Operational Notes

- keep notebooks focused on exploration, not production orchestration
- prefer wheel tasks for repeatable jobs
- pin production data access through governed Databricks locations and secrets
