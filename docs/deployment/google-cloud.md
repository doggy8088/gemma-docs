The Google Cloud platform provides many options for deploying, serving,
and fine-tuning Gemma 4 open models, including the following:

- [Gemini Enterprise Agent Platform](google-cloud.md#gemini-enterprise-agent-platform)
- [Cloud Run](google-cloud.md#cloud-run)
- [Google Kubernetes Engine (GKE)](google-cloud.md#google-kubernetes-engine-gke)
- [Agent Development Kit (ADK)](google-cloud.md#agent-development-kit-adk)
- [Gemini Enterprise Agent Platform Training Clusters](google-cloud.md#gemini-enterprise-agent-platform-training-clusters)
- [MaxText](google-cloud.md#maxtext)
- [vLLM with TPUs](google-cloud.md#vllm-with-tpus)
- [Sovereign Cloud](google-cloud.md#sovereign-cloud)

## Gemini Enterprise Agent Platform

[Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma) is a Google Cloud platform for rapidly
building and scaling machine learning projects. Gemma 4 is available in [Model
Garden](https://console.cloud.google.com/agent-platform/publishers/google/model-garden/gemma4), a curated collection of models on Gemini Enterprise Agent Platform.
You can test and deploy models directly from the console.

To learn more, refer to the following pages:

- [Agent Platform overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/overview): Get started with Gemini Enterprise Agent Platform.
- [Gemma with Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma): Use Gemma open models with Gemini Enterprise Agent Platform.

## Cloud Run

[Cloud Run](https://cloud.google.com/run) is a fully managed platform to run your code or containers on
top of Google's highly scalable infrastructure.
[Deploy Gemma 4 on Cloud Run](https://docs.cloud.google.com/run/docs/run-gemma-on-cloud-run) using GPUs for scale-to-zero, pay-per-use
inference.

For larger mode sizes, [leverage advanced configurations](https://codelabs.developers.google.com/codelabs/cloud-run/cloud-run-gpu-rtx-pro-6000-gemma4-vllm) with RTX 6000 Pro
GPUs and Model Streaming.

## Google Kubernetes Engine (GKE)

[Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine) (GKE) is a managed Kubernetes service from Google
Cloud. [Run Gemma 4 on GKE](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm) for enterprise-grade container orchestration.
Use TPUs and GPUs to serve models with high throughput and low latency.

## Agent Development Kit (ADK)

Build and orchestrate AI agents with Gemma 4 and the [Agent Development Kit
(ADK)](https://adk.dev/agents/models/google-gemma/). Gemma 4's strong reasoning and function-calling capabilities make it
ideal for agentic workflows.

## Gemini Enterprise Agent Platform Training Clusters

[Fine-tune Gemma 4 using Gemini Enterprise Agent Platform Training
Clusters](https://discuss.google.dev/t/end-to-end-guide-fine-tuning-and-serving-gemma-4-on-vertex-ai/345865). Training Clusters provides optimized infrastructure for
large-scale training and fine-tuning of open models.

## vLLM with TPUs

[Serve Gemma 4 on Google Cloud TPUs](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#cloud-tpu-deployment-via-docker) for state-of-the-art serving
performance.

## MaxText

Gemma 4 is supported in [MaxText](https://github.com/google/maxtext), a high-performance, arbitrary-sized JAX
LLM implementation for Google Cloud TPUs.

## Sovereign Cloud

Gemma 4 is available on [Sovereign Cloud](https://cloud.google.com/sovereign-cloud) solutions, providing enhanced
control and compliance for sensitive workloads.
