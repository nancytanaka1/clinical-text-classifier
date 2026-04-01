# Databricks notebook source

# MAGIC %md
# MAGIC # Test Runner — Clinical Text Classifier

# COMMAND ----------

# MAGIC %pip install pytest pytest-cov --quiet

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/Repos/nguyenn.mail@gmail.com/clinical-text-classifier/tests/",
        "-v",
        "--tb=short",
        "--junitxml=/tmp/test-results.xml",
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode != 0:
    raise SystemExit(f"Tests failed with return code {result.returncode}")
else:
    print("All tests passed.")
