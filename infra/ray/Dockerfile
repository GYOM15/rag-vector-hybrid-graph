# Ray Serve + vLLM image: autoscaling vLLM replicas on a multi-GPU node.
# Base already has CUDA + vLLM; we add Ray Serve (+ the llm extra). Ray is pinned
# because ray.serve.llm's API moves between releases.
FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir "ray[serve,llm]==2.40.0"

WORKDIR /app
COPY serve_app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# OpenAI API :8000, Ray metrics (for Prometheus) :8080, Ray dashboard :8265.
EXPOSE 8000 8080 8265
ENTRYPOINT ["./entrypoint.sh"]
