# Inputs — fill terraform.tfvars (copy terraform.tfvars.example).

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU instance type. g5.xlarge = 1x NVIDIA A10G, 24 GB VRAM."
  type        = string
  default     = "g5.xlarge"
}

variable "allowed_cidr" {
  description = "Your public IP as CIDR (e.g. 1.2.3.4/32). Locks SSH / vLLM / Grafana to you."
  type        = string
}

variable "key_name" {
  description = "Name of an existing EC2 key pair (for SSH)."
  type        = string
}

variable "model" {
  description = "HuggingFace model id served by vLLM."
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct"
}

variable "repo_url" {
  description = "Public git URL of this repo; the instance clones it for the compose stack."
  type        = string
}

variable "serving_mode" {
  description = "vllm = single-GPU vLLM; ray = Ray Serve autoscaling (needs a multi-GPU instance_type)."
  type        = string
  default     = "vllm"
}
