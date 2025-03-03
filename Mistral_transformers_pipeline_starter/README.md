# Mistral Transformers Pipeline Starter

This repository contains a starter project for building a pipeline using Mistral Transformers, along with integrations for image generation and animation using the Stability API, and model pulling via Ollama.

## Project Structure

The project is organized as follows:

| Directory/File | Description |
|---|---|
| `charts/` | Helm charts for deployment |
| `charts/mistral-app/` | Helm chart for the Mistral application |
| `charts/mistral-app/Chart.yaml` | Chart metadata |
| `charts/mistral-app/values.yaml` | Default configuration values |
| `charts/mistral-app/templates/` | Kubernetes resource templates |
| `charts/mistral-app/templates/deployment.yaml` | Deployment template |
| `charts/mistral-app/templates/service.yaml` | Service template |
| `charts/mistral-app/_helpers.tpl` | Helper functions for the chart |
| `scripts/` | Scripts for image generation and other tasks |
| `scripts/animate_stability_stf.py` | Script for generating and animating images using Stability API |
| `scripts/mistral_starter_chat.py` | Example script for Mistral chat |
| `utils.py` | Utility functions |
| `wrapper.py` | Wrapper script for integrating functionalities |
| `ollama_integration.py` | Script for pulling models using Ollama |
| `_starter.py` | Unit tests for the model |
| `.env` | Environment variables |
| `README.md` | This file |

