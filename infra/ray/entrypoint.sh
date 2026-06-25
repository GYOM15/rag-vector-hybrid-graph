#!/bin/bash
# Start a single-node Ray with its Prometheus metrics endpoint, then the Serve app.
set -euxo pipefail

# --metrics-export-port exposes Ray + Ray Serve metrics for Prometheus to scrape.
ray start --head --dashboard-host=0.0.0.0 --metrics-export-port=8080

# Blocks; serves the OpenAI API on :8000 and manages autoscaling replicas.
serve run serve_app:app --host 0.0.0.0
