output "public_ip" {
  description = "Instance public IP."
  value       = aws_instance.serving.public_ip
}

output "vllm_api" {
  description = "OpenAI-compatible endpoint — point the pipeline here."
  value       = "http://${aws_instance.serving.public_ip}:8000/v1"
}

output "grafana_url" {
  description = "Grafana dashboards (anonymous viewing enabled; admin / admin to edit)."
  value       = "http://${aws_instance.serving.public_ip}:3000"
}
