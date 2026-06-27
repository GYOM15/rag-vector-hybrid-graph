# Deploy guide — vLLM serving + observability on AWS

Step-by-step to stand up the single-GPU **vLLM + Prometheus/Grafana** stack with
Terraform, run the benchmarks against it, and tear it down. Each step shows the
**🖥️ Console** way *and* the **⌨️ CLI** way. Region: `ca-central-1` (g5 is available there).

> 💸 **Cost** — a `g5.xlarge` is ~$1/hr; the whole demo is a few dollars **if you destroy after**.
> Forgetting `terraform destroy` ≈ ~$25/day. Set the budget alarm (step 4).
> 🔑 **Rule** — quota / key pair / instance all live in **one region**: keep everything in `ca-central-1`.

---

## 0. Prerequisites
- AWS CLI v2 + Terraform installed: `aws --version` · `terraform version`.
- An AWS account.

## 1. Configure the AWS CLI (one time)
You need an **access key** (from IAM), then hand it to the CLI.
- 🖥️ **Console**: IAM → Users → your user → **Security credentials** → **Create access key** → use case **CLI** → copy the **Access Key ID** + **Secret** (shown once). Permissions: attach `AmazonEC2FullAccess` + `ServiceQuotasFullAccess` (or `AdministratorAccess`).
- ⌨️ **CLI**: `aws configure` → paste key, secret, region `ca-central-1`, format `json` (stored in `~/.aws/`).
- **Verify**: `aws sts get-caller-identity` → returns your account/ARN = OK.

## 2. Region
- 🖥️ **Console**: region selector (top-right) → **Canada (Central)**.
- ⌨️ **CLI**: `aws configure set region ca-central-1` (and `region = "ca-central-1"` in tfvars, step 6).
- **See it**: `aws configure get region`.

## 3. GPU quota — the long pole, do FIRST
GPU instances default to **0** vCPU quota. Request **4** (one `g5.xlarge`).
- 🖥️ **Console**: **Service Quotas** → **Amazon EC2** → search **"Running On-Demand G and VT instances"** → **Request increase at account level** → `4`.
- ⌨️ **CLI**:
  ```bash
  aws service-quotas request-service-quota-increase \
    --service-code ec2 --quota-code L-DB2E81BA --desired-value 4 --region ca-central-1
  ```
- **Check approval** (`0.0` → **`4.0`** = granted):
  - 🖥️ Service Quotas → EC2 → that quota → **"Applied quota value"**. Or **Quota request history** → Status `Approved`.
  - ⌨️ `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region ca-central-1 --query 'Quota.Value' --output text`

## 4. Budget alarm — before launching anything
- 🖥️ **Console**: **Billing and Cost Management** → **Budgets** → **Create budget** → template **Monthly cost budget** → `$10` + your email.
- ⌨️ **CLI**:
  ```bash
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
  aws budgets create-budget --account-id $ACCOUNT \
    --budget '{"BudgetName":"rag-demo","BudgetLimit":{"Amount":"10","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
    --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"YOUR_EMAIL"}]}]'
  ```

## 5. EC2 key pair (for SSH)
- 🖥️ **Console**: **EC2** (region ca-central-1) → **Key Pairs** → **Create key pair** → name `rag-key`, type RSA, format `.pem` → downloads → `chmod 400 rag-key.pem`.
- ⌨️ **CLI**:
  ```bash
  aws ec2 create-key-pair --region ca-central-1 --key-name rag-key \
    --query KeyMaterial --output text > ~/.ssh/rag-key.pem && chmod 400 ~/.ssh/rag-key.pem
  ```
- **See it**: `aws ec2 describe-key-pairs --region ca-central-1 --query 'KeyPairs[].KeyName' --output text`

## 6. `terraform.tfvars` (local file — no console)
```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
```
Fill the **uncommented** lines (and **save**):
```hcl
region       = "ca-central-1"
key_name     = "rag-key"
allowed_cidr = "<your-IP>/32"     # curl ifconfig.me  (only your IP can reach the box)
repo_url     = "https://github.com/GYOM15/rag-vector-hybrid-graph.git"
repo_branch  = "feat/aws-serving"
```
The "Optional overrides" lines stay **commented** — the defaults (`g5.xlarge`, `Qwen2.5-7B`) are what we want.
- **Check your active settings**: `grep -vE '^[[:space:]]*#' terraform.tfvars | grep -E '\S'`

## 7. Deploy (Terraform creates the instance + firewall — no console clicks)
```bash
cd infra
terraform init
terraform apply        # type "yes" → prints public_ip, vllm_api, grafana_url
```
- 🖥️ Optional check: **EC2 → Instances** → `rag-serving` is *running*.

## 8. Use it (wait ~5-10 min for the model to download + load)
- Ready check: `curl http://<IP>:8000/v1/models`
- ⌨️ From a local checkout of the branch (pipeline runs as a client):
  ```bash
  LLM_PROVIDER=openai OPENAI_BASE_URL=http://<IP>:8000/v1 \
  OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct OPENAI_API_KEY=EMPTY \
    python -m eval.answer_eval --max-queries 100
  python -m eval.serving_bench --base-url http://<IP>:8000/v1 --model Qwen/Qwen2.5-7B-Instruct --n-prompts 128
  ```
- 🖥️ **Grafana**: open `http://<IP>:3000` → watch latency / throughput / GPU under load.

## 9. Tear down — stop paying
- ⌨️ **CLI**: `terraform destroy` (removes the instance *and* the security group).
- 🖥️ **Console**: EC2 → Instances → Terminate (but `terraform destroy` is cleaner).

---

**Order**: 3 (quota, wait for approval) → 4, 5, 6 in parallel → 7 once the quota shows `4.0`.
