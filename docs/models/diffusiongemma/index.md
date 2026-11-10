---
title: DiffusionGemma 模型總覽
description: DiffusionGemma 是一個實驗性的開放模型，探索文字擴散（text diffusion）——一種極為快速的文字生成方式。
models: [diffusiongemma]
source_url: "https://ai.google.dev/gemma/docs/diffusiongemma.md.txt"
---
# DiffusionGemma 模型總覽

DiffusionGemma 是一個實驗性的開放模型，探索文字擴散（text diffusion）——一種極為快速的文字生成方式。基於 26B（4B 啟動）混合專家（MoE）Gemma 4 架構，DiffusionGemma 使用離散擴散生成詞元。這個開放權重模型具備多模態能力，可處理文字、影像與影片輸入，產生文字輸出。

DiffusionGemma 建立在 MoE 基礎上，旨在提升生成速度（每秒詞元數），同時保持能在多種硬體環境部署。DiffusionGemma 繼承 Gemma 4 的架構與能力進展，並引入幾項核心功能：

- **離散文字擴散：**從傳統的因果式逐詞元生成，轉為區塊自迴歸多畫布取樣。模型透過平行地反覆去除詞元區塊（「畫布」）的雜訊來生成文字，大幅提升解碼速度。
- **多模態處理：**原生接受文字、影像（支援可變長寬比與解析度）與影片輸入。（注意：不支援音訊輸入）。
- **編碼器-解碼器架構：**使用自迴歸編碼器處理並快取提示上下文，搭配在生成畫布上套用雙向注意力的去噪器。
- **混合專家（MoE）效率：**採用基於 26B（4B 啟動）MoE 變體的稀疏 MoE 設計，以極低開銷提供深度的推理（reasoning）能力。量化後可裝進消費級 GPU 的 18GB VRAM 限制內，非常適合本機執行。
- **思考模式：**內建可設定的推理通道，讓模型在輸出最終答案前逐步思考。

## 與傳統模型的取捨

傳統語言模型非常適合大規模雲端部署，因為可以同時批次處理數千個請求，但在單一使用者本機上執行時會讓硬體閒置。DiffusionGemma 的解決方式是同時生成整個 256 詞元的區塊，而不是一次一個詞元，最大化本機硬體效能。

然而，這種做法嚴格鎖定在面向消費者、低併發的本機使用場景；因為在高 QPS 的雲端工作負載下，平行解碼的效益會遞減，所以吞吐量優勢在單一加速器上低至中批次規模時最為明顯。

## 建議的部署設定

為達到最佳延遲與品質，建議使用以下擴散取樣設定的預設參數來部署：

| 參數 | 建議值 | 功能 | 理由 |
|---|---|---|---|
| 最大去噪步數 | 48 | 每塊畫布的去噪步數上限。 | 去噪步數的安全上限。啟用自適應提早停止時，實際步數會更少，通常依任務不同為 12–16 步。 |
| 溫度排程 | 線性 0.8 -\> 0.4 | 溫度縮放排程，起始較高並隨去噪步數降低。 | 高溫度（0.8）鼓勵早期探索；低溫度（0.4）鎖定最終詞元。 |
| 自適應提早停止 | 熵閾值：0.005 | 若 A) 整塊畫布的平均模型熵低於閾值，且 B) 連續兩次去噪器預測完全相同，則提早停止執行。 | 簡單的提示與程式碼這類結構化任務所需的去噪步數較少，能依任務複雜度動態調整每秒詞元數。 |
| 詞元選擇 | 熵上限：0.1 | 在每個步驟中，取樣器選擇熵最低的詞元，使其互資訊上限保持在熵上限以下。取樣器會對未被選中的詞元完全重新注入雜訊。 | 確保只有模型相對有把握的詞元會被選出來精修畫布，其餘詞元留到後續去噪步驟再精修。 |

<div class="cookbook-grid">
<a class="cookbook-card" href="https://huggingface.co/collections/google/diffusiongemma" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">在 Hugging Face 上取得</p>
</a>
<a class="cookbook-card" href="https://www.kaggle.com/models/google/diffusiongemma" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">在 Kaggle 上取得</p>
</a>
</div>

[在 Vertex 上使用](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/diffusiongemma){ data-model="diffusiongemma" }

取得實驗性模型權重（以 Apache 2.0 授權釋出），讓您能在自己的專案與應用中部署。

[深入了解 DiffusionGemma 架構](explained.md)
[試用 DiffusionGemma](inference.md)

[微調 DiffusionGemma](https://github.com/google/hackable_diffusion)
[部署 DiffusionGemma](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm)

--8<-- "glossary.md"
