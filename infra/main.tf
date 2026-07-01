terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Deep Learning AMI: NVIDIA drivers + Docker + nvidia-container-toolkit preinstalled,
# so user_data only has to clone the repo and `docker compose up`.
data "aws_ami" "dl" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# Single-IP security group: nothing is open to the world.
resource "aws_security_group" "serving" {
  name        = "rag-serving"
  description = "SSH, vLLM API and Grafana, locked to one IP."

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  ingress {
    description = "vLLM OpenAI-compatible API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  ingress {
    description = "Grafana"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "serving" {
  ami                    = data.aws_ami.dl.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.serving.id]

  root_block_device {
    volume_size = 100 # GB: model weights + Docker images
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    repo_url     = var.repo_url
    model        = var.model
    serving_mode = var.serving_mode
  })

  tags = { Name = "rag-serving" }
}
