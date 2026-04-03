#!/usr/bin/env bash
set -euo pipefail
cd /root/trading-models
export PYTHONPATH=src
/usr/bin/python3 scripts/run_baseline.py
printf '\n[%s] nightly baseline refresh complete\n' "$(date -Is)" >> /root/.hermes/memories/MODEL_LOG.md
printf 'Latest results file: /root/trading-models/reports/RESULTS.md\n' >> /root/.hermes/memories/MODEL_LOG.md
