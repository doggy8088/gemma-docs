---
title: DataGemma
description: DataGemma 是一款研究工具，讓使用者能以淺白語言提問，並根據 Data Commons 儲存庫中公開的統計資料獲得答案。
models: [datagemma]
source_url: "https://ai.google.dev/gemma/docs/datagemma.md.txt"
---
# DataGemma

DataGemma 是一款研究工具，讓使用者能以淺白語言提問，並根據 [Data Commons](https://datacommons.org/) 儲存庫中公開的統計資料獲得答案。此工具使用特別打造的 [Gemma](../gemma-4/index.md) 版本、搭載 Gemini 1.5 Pro 的 [Gemini API](https://ai.google.dev/gemini-api)，以及一組專為搭配 Data Commons 設計的函式庫。


此研究工具提供兩種根據 Data Commons 統計資料回答問題的技術：

- **檢索交錯生成（Retrieval-Interleaved Generation, RIG）** - 此方法使用 Gemma 2 的變體，該變體經過微調，能識別何時需要以 Data Commons 更準確的資訊取代生成的數字。詳情請見 Colab [筆記本](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rig.ipynb){ data-model="datagemma" }，模型可於 [Kaggle](https://www.kaggle.com/models/google/datagemma-rig){ data-model="datagemma" } 或 [Hugging Face](https://huggingface.co/google/datagemma-rig-27b-it){ data-model="datagemma" } 取得。
- **檢索增強生成（Retrieval-Augmented Generation, RAG）** - 此方法使用 Gemma 2 的變體，從 Data Commons 檢索相關資訊，再用該資訊為 Gemini 1.5 Pro 模型建立擴展提示。詳情請見 Colab [筆記本](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rag.ipynb){ data-model="datagemma" }，模型可於 [Kaggle](https://www.kaggle.com/models/google/datagemma-rag){ data-model="datagemma" } 或 [Hugging Face](https://huggingface.co/google/datagemma-rag-27b-it){ data-model="datagemma" } 取得。


如需 DataGemma 的更多研究與技術細節，請見 [DataGemma 技術論文](http://datacommons.org/link/DataGemmaPaper){ data-model="datagemma" }。

- **以真實資料生成答案** — 將生成式人工智慧（AI）應用於龐大的公開統計資料庫，探索並發掘新的洞察。
- **評測 AI 資料落地技術** — 研究以檢索增強與資料交錯技術引導生成式 AI 模型輸出的方法。

## 深入了解

### [在 Kaggle 上檢視 RIG](https://www.kaggle.com/models/google/datagemma-rig){ data-model="datagemma" }

在 Kaggle 上查看更多關於 DataGemma RIG 模型的程式碼、筆記本、資訊與討論。

### [在 Colab 中執行 RIG](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rig.ipynb){ data-model="datagemma" }

使用檢索交錯技術，試用 DataGemma 回答問題。

### [在 Colab 中執行 RAG](https://colab.research.google.com/github/datacommonsorg/llm-tools/blob/main/notebooks/datagemma_rag.ipynb){ data-model="datagemma" }

使用檢索增強技術，試用 DataGemma 回答問題。
