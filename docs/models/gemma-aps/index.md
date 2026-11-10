---
title: Gemma 抽象命題分割（APS）
description: 抽象命題分割（APS）是一種將文字資訊拆解為原子命題要素的分析技術，可提升檢索與事實查核的精確度。
models: [gemma-aps]
source_url: "https://ai.google.dev/gemma/docs/gemma-aps.md.txt"
---
# Gemma 抽象命題分割（APS）

## Gemma-APS


抽象命題分割（abstractive proposition segmentation, APS）是一種將文字資訊拆解為組成要素的分析技術。Gemma-APS 是 Gemma 的變體，接收給定的文字段落，嘗試將內容分割為文字中表達的個別事實、陳述與想法，並以完整句子重新陳述，僅對原文做小幅改動。


此模型可用於需要將文字內容拆解為有意義組成要素的研究。應用包括生成內容落地（grounding）、內容檢索、事實查核，以及生成輸出的評測（例如摘要），這些情境中將個別事實資料分開以便獨立驗證很有幫助。更多資訊請見[研究論文](https://arxiv.org/pdf/2406.19803)。

- **分析內容的組成要素** — 應用生成式人工智慧（AI）將內容拆解為離散陳述，以供分析或進一步處理。
- **結合更大規模的內容分析** — 將生成的內容片段用於更大型的內容分析專案，例如落地、檢索、事實查核、生成任務評測與重複內容偵測。

## 深入了解

### [檢視 2B 模型](https://huggingface.co/google/gemma-2b-aps-it)

在 Hugging Face 上查看更多關於 Gemma-APS 2B 模型的程式碼、筆記本、資訊與討論。

### [檢視 7B 模型](https://huggingface.co/google/gemma-7b-aps-it)

在 Hugging Face 上查看更多關於 Gemma-APS 7B 模型的程式碼、筆記本、資訊與討論。

### [閱讀研究](https://arxiv.org/pdf/2406.19803)

深入了解 Gemma-APS 模型的研究、開發與運作。
