---
title: Google Cloud
description: "在 Google Cloud 上部署、服務與微調 Gemma 4：Cloud Run、GKE、Agent Development Kit、TPU 上的 vLLM、MaxText 與主權雲。"
models: [gemma-4]
frameworks: [google-cloud, vllm]
source_url: "https://ai.google.dev/gemma/docs/integrations/google-cloud.md.txt"
---
# Google Cloud 上的 Gemma

Google Cloud 提供多種部署、服務與微調 Gemma 4 的途徑。請依您的需求挑選，
而非依產品名稱——每張卡片都會跳到下方的詳細說明。

### 服務它

<div class="card-grid">
<a class="card" href="google-cloud.md#cloud-run">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">Cloud Run</div>
  <p class="card-desc">全託管、可縮減至零、按使用量計費的 GPU 推論。</p>
</a>
<a class="card" href="google-cloud.md#google-kubernetes-engine-gke">
  <span class="card-icon">{{ icon('lucide/boxes') }}</span>
  <div class="card-title">GKE</div>
  <p class="card-desc">在 TPU 或 GPU 上自建服務叢集，實現高吞吐量與低延遲。</p>
</a>
<a class="card" href="google-cloud.md#vllm-with-tpus">
  <span class="card-icon">{{ icon('lucide/zap') }}</span>
  <div class="card-title">vLLM on TPUs</div>
  <p class="card-desc">在 Cloud TPU 上實現頂尖的服務效能。</p>
</a>
<a class="card" href="google-cloud.md#sovereign-cloud">
  <span class="card-icon">{{ icon('lucide/shield') }}</span>
  <div class="card-title">Sovereign Cloud</div>
  <p class="card-desc">為敏感工作負載提供額外的控制與合規。</p>
</a>
</div>

### 用它來建構

<div class="card-grid">
<a class="card" href="google-cloud.md#gemini-enterprise-agent-platform">
  <span class="card-icon">{{ icon('lucide/layout-grid') }}</span>
  <div class="card-title">Agent Platform</div>
  <p class="card-desc">直接從主控台的 Model Garden 測試與部署。</p>
</a>
<a class="card" href="google-cloud.md#agent-development-kit-adk">
  <span class="card-icon">{{ icon('lucide/wrench') }}</span>
  <div class="card-title">Agent Development Kit</div>
  <p class="card-desc">圍繞 Gemma 的推理與函數呼叫編排 Agent。</p>
</a>
</div>

### 訓練它

<div class="card-grid">
<a class="card" href="google-cloud.md#gemini-enterprise-agent-platform-training-clusters">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">Training Clusters</div>
  <p class="card-desc">為大規模開放模型微調提供的託管基礎設施。</p>
</a>
<a class="card" href="google-cloud.md#maxtext">
  <span class="card-icon">{{ icon('lucide/cpu') }}</span>
  <div class="card-title">MaxText</div>
  <p class="card-desc">適用於 Cloud TPU 的高效能 JAX 實作。</p>
</a>
</div>

## Gemini Enterprise Agent Platform

[Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma) 是 Google Cloud 用於快速
建構與擴展機器學習專案的平台。Gemma 4 已收錄於 [Model
Garden](https://console.cloud.google.com/agent-platform/publishers/google/model-garden/gemma4)——Gemini Enterprise Agent Platform 上的精選模型集合。
您可以直接從主控台測試與部署模型。

若要了解更多，請參閱以下頁面：

- [Agent Platform 概覽](https://docs.cloud.google.com/gemini-enterprise-agent-platform/overview)：開始使用 Gemini Enterprise Agent Platform。
- [搭配 Gemini Enterprise Agent Platform 使用 Gemma](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma)：在 Gemini Enterprise Agent Platform 上使用 Gemma 開放模型。

## Cloud Run

[Cloud Run](https://cloud.google.com/run) 是全託管的平台，可在 Google
高度可擴展的基礎設施上執行您的程式碼或容器。
使用 GPU [在 Cloud Run 上部署 Gemma 4](https://docs.cloud.google.com/run/docs/run-gemma-on-cloud-run)，實現可縮減至零、按使用量計費的
推論。

對於較大的模型尺寸，請[採用進階設定](https://codelabs.developers.google.com/codelabs/cloud-run/cloud-run-gpu-rtx-pro-6000-gemma4-vllm){ data-model="gemma-4" }，搭配 RTX 6000 Pro
GPU 與模型串流。

## Google Kubernetes Engine (GKE)

[Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine)（GKE）是 Google
Cloud 的託管 Kubernetes 服務。[在 GKE 上執行 Gemma 4](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm)，實現企業級容器編排。
使用 TPU 與 GPU 以高吞吐量與低延遲服務模型。

## Agent Development Kit (ADK)

使用 [Agent Development Kit
(ADK)](https://adk.dev/agents/models/google-gemma/)，以 Gemma 4 建構並編排 AI Agent。Gemma 4 強大的推理與函數呼叫能力使其
非常適合 Agent 工作流程。

## Gemini Enterprise Agent Platform 訓練叢集

[使用 Gemini Enterprise Agent Platform 訓練叢集
微調 Gemma 4](https://discuss.google.dev/t/end-to-end-guide-fine-tuning-and-serving-gemma-4-on-vertex-ai/345865)。訓練叢集為
大規模訓練與開放模型微調提供最佳化的基礎設施。

## TPU 上的 vLLM

[在 Google Cloud TPU 上服務 Gemma 4](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#cloud-tpu-deployment-via-docker){ data-model="gemma-4" }，實現頂尖的服務
效能。

## MaxText

Gemma 4 已支援於 [MaxText](https://github.com/google/maxtext)——適用於 Google Cloud TPU 的高效能、任意規模 JAX
LLM 實作。

## Sovereign Cloud

Gemma 4 已在 [Sovereign Cloud](https://cloud.google.com/sovereign-cloud) 解決方案上提供，為敏感工作負載提供增強的
控制與合規。
