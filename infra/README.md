# Serving + observability on AWS (Terraform)

Deploys a single GPU instance that serves a model with **vLLM** (OpenAI-compatible API)
and exposes a live **Grafana** dashboard fed by **Prometheus**. The RAG pipeline and the
benchmarks run as *clients* against the endpoint — no pipeline code changes (the repo's
`openai` provider already speaks this API).

```
your machine ──HTTP──>  EC2 GPU instance (docker compose)
  serving_bench /          ├─ vllm           :8000  /v1  + /metrics
  answer_eval              ├─ dcgm-exporter  :9400  GPU metrics
                           ├─ prometheus     :9090  scrapes vllm + dcgm every 15s
  browser  ──HTTP──>       └─ grafana        :3000  dashboards (reads prometheus)
```

## How the observability works

- **Prometheus** is a metrics database with a **pull** model: every 15 s it HTTP-GETs each
  target's `/metrics` page and stores the numbers as time series. `prometheus/prometheus.yml`
  lists the targets — `vllm:8000` and `dcgm-exporter:9400` (services reach each other by name
  on the shared Docker network).
- **vLLM is already instrumented**: it serves Prometheus metrics at `/metrics` — request rate,
  latency histograms, generated tokens, and the running/waiting **batching queue**.
- **The GPU** is exposed by a small side-car, the **DCGM exporter**, which reads the hardware
  (utilisation, memory, power, temperature) and serves it as `/metrics`.
- **Grafana** reads Prometheus (auto-registered as a data source via
  `grafana/provisioning/datasources/`), runs PromQL queries, and draws the panels defined in
  `grafana/dashboards/vllm-serving.json`. Everything is provisioned from files — no manual clicks.

## Prerequisites

- AWS account + credentials configured locally (`aws configure`).
- A **GPU vCPU quota** for the G/P family (often **0 by default** → request an increase first;
  it can take hours to a day).
- An existing **EC2 key pair** (for SSH).
- The repo reachable by `git clone` from the instance (public, or adjust `repo_url`).

## Run it

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # then fill allowed_cidr, key_name, repo_url
terraform init
terraform apply                                 # prints public_ip, vllm_api, grafana_url

# Give vLLM a few minutes to download weights, then from your machine:
LLM_PROVIDER=openai OPENAI_BASE_URL=<vllm_api> OPENAI_MODEL=<model> OPENAI_API_KEY=EMPTY \
  python -m eval.answer_eval --max-queries 100
python -m eval.serving_bench --base-url <vllm_api> --model <model> --n-prompts 128

# Open <grafana_url> in a browser and watch the dashboard while the load runs.

terraform destroy                               # IMPORTANT: stop paying for the GPU
```

## Cost

A `g5.xlarge` is roughly ~$1/hour. The workflow is **apply → capture numbers + dashboard
screenshots → destroy** — nothing is meant to run idle.
