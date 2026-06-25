#!/bin/bash
# Bootstraps the serving + observability stack on first boot.
# The Deep Learning AMI already has Docker + the NVIDIA container toolkit.
set -euxo pipefail

cd /home/ubuntu
git clone ${repo_url} app
cd app/infra

# MODEL is consumed by the compose file's $${MODEL} interpolation.
echo "MODEL=${model}" > .env

# Pick the stack: single-GPU vLLM, or Ray Serve autoscaling (multi-GPU).
COMPOSE_FILE=$([ "${serving_mode}" = "ray" ] && echo docker-compose.ray.yml || echo docker-compose.yml)
docker compose -f "$COMPOSE_FILE" up -d --build
