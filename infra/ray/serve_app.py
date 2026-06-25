"""Ray Serve app: autoscaling vLLM replicas behind one OpenAI-compatible endpoint.

Ray Serve sits in front of the vLLM engine and scales replicas with load
(min..max replicas, one GPU each). Exposes the OpenAI API on :8000 — the same
contract the pipeline already speaks, so nothing downstream changes.

    serve run serve_app:app           # local
    (or via ../docker-compose.ray.yml on a multi-GPU node)

Version note: `ray.serve.llm` evolves quickly. Ray is pinned in the Dockerfile;
if a field name differs on your Ray version, adjust here (the shape is stable:
a model id, an autoscaling block, and vLLM engine kwargs).
"""

import os

from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={"model_id": os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")},
    # Autoscaling: idle at MIN, scale out to MAX when a replica's concurrent
    # request count exceeds the target. This is the behaviour the dashboard shows.
    deployment_config={
        "autoscaling_config": {
            "min_replicas": int(os.getenv("MIN_REPLICAS", "1")),
            "max_replicas": int(os.getenv("MAX_REPLICAS", "4")),
            "target_ongoing_requests": int(os.getenv("TARGET_ONGOING", "8")),
        }
    },
    # One GPU per replica; on a 4-GPU node that means up to 4 replicas.
    engine_kwargs={"max_model_len": 8192, "tensor_parallel_size": 1},
)

app = build_openai_app({"llm_configs": [llm_config]})
