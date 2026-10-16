---
title: Gemma Scope
description: Gemma Scope 2 是為 Gemma 3 模型系列設計的全面、開放的可解釋性工具套件。
models: [gemma-scope]
source_url: "https://ai.google.dev/gemma/docs/gemma_scope.md.txt"
---
# Gemma Scope

## 使用 Gemma Scope 2 分析 Gemma 3


Gemma Scope 2 是為 **Gemma 3** 模型系列設計的全面、開放的可解釋性工具套件。此工具讓您能檢視各別層的行為，協助研究人員分析複雜的語言模型行為，並除錯如越獄（jailbreak）或幻覺（hallucination）等新興行為。


此工具套件如同模型的顯微鏡，提供在 Gemma 3 家族每一層上訓練的**稀疏自編碼器（Sparse Autoencoders, SAEs）**與**轉換器（Transcoders）**。


**想找舊版嗎？**


原始的 [Gemma Scope（適用於 Gemma 2）](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/)
仍供使用 Gemma 2 模型家族的研究人員使用。

- **模型行為評測** — 使用 SAE 與 Transcoder 分析 Gemma 3 中複雜的內部行為與多步驟演算法。
- **聊天機器人安全與除錯** — 分析特定聊天行為、拒答機制與思維鏈忠實度，打造更安全的 AI 智慧代理。

## 深入了解

### [閱讀 Gemma Scope 2 部落格](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior)

了解 Gemma Scope 2 的新架構、訓練資料與能力。

### [從 Hugging Face 下載](https://huggingface.co/collections/google/gemma-scope-2)

取得 Gemma 3 可解釋性套件的權重、程式碼與文件。

### [在 Colab 中執行](https://colab.sandbox.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r?usp=sharing)

試用互動式教學，視覺化特徵並修改模型行為。

### [舊版（適用於 Gemma 2 的 Gemma Scope）](https://huggingface.co/collections/google/gemma-scope-release)

取得原始 Gemma Scope（適用於 Gemma 2）的部落格與資源。
