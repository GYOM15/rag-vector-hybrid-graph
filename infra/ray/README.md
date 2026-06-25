# Ray Serve autoscaling (multi-GPU)

The scale layer on top of the single-GPU vLLM stack. **Ray Serve** runs several
vLLM replicas behind one OpenAI-compatible endpoint and **autoscales** them with
load (1 → N replicas, one GPU each). This is the thing a single GPU — local or
Colab — cannot show.

```
client ──> :8000  Ray Serve (OpenAI API)
                   ├─ vLLM replica  (GPU 0)   scaled 1..MAX with load
                   ├─ vLLM replica  (GPU 1)
                   └─ ...
           :8080  Ray + Serve metrics ─> Prometheus ─> Grafana (replicas, ongoing reqs)
           :9400  DCGM (per-GPU metrics) ┘
```

## Files

- `serve_app.py` — the Ray Serve app: a vLLM-backed deployment with an autoscaling
  block (`min/max_replicas`, `target_ongoing_requests`). Built with `ray.serve.llm`.
- `Dockerfile` + `entrypoint.sh` — image = vLLM base + pinned Ray; starts Ray with a
  metrics port, then `serve run`.
- `../docker-compose.ray.yml` — ray-llm + DCGM + Prometheus (`prometheus.ray.yml`) + Grafana.
- `../grafana/dashboards/ray-autoscaling.json` — replicas, ongoing requests, request rate, GPU%.

## Deploy

Set the Ray variant in `terraform.tfvars`, on a multi-GPU instance:

```hcl
serving_mode  = "ray"
instance_type = "g5.12xlarge"   # 4x A10G
```

Then `terraform apply`. To see autoscaling: open Grafana, then push load with
`python -m eval.serving_bench --base-url <vllm_api> --model <model> --n-prompts 256`
at rising concurrency — replicas should climb on the dashboard, then scale back down.

## How autoscaling shows up

Ray Serve exposes metrics on `:8080` (enabled by `--metrics-export-port`). Prometheus
scrapes them; Grafana plots the **replica count** and **ongoing requests** — so you
literally watch replicas spin up as load rises and the GPUs light up (DCGM).

## ⚠️ Needs a real multi-GPU run to validate

This was written for syntax + structure but **not run** (no local GPU). Two things are
**version-sensitive** and may need a small tweak on first deploy:
- **`ray.serve.llm` API** — field names (`model_loading_config`, `deployment_config`,
  `engine_kwargs`) track the pinned Ray version in the `Dockerfile`; adjust if your Ray differs.
- **Ray metric names** — the dashboard's PromQL (`ray_serve_*`) varies by Ray version;
  confirm the exact names in Grafana → Explore and tweak the panels.
