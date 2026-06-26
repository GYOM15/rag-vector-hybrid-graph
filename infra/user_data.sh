#!/bin/bash
# Bootstraps the serving + observability stack on first boot.
# The Deep Learning AMI already has Docker + the NVIDIA container toolkit.
set -euxo pipefail

cd /home/ubuntu
git clone -b ${repo_branch} ${repo_url} app
cd app/infra

# MODEL is consumed by docker-compose.yml's $${MODEL} interpolation.
echo "MODEL=${model}" > .env
docker compose up -d
