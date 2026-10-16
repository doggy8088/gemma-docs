---
title: PaliGemma 模型卡
description: PaliGemma 是一款多功能、輕量的視覺語言模型（VLM），設計靈感來自 PaLI-3，並以 SigLIP 視覺模型與 Gemma 語言模型等開放元件為基礎。
models: [paligemma]
source_url: "https://ai.google.dev/gemma/docs/paligemma/model-card.md.txt"
---
# 模型卡

**模型頁面：** [PaliGemma](index.md)

**資源與技術文件：**

- [Kaggle 上的 PaliGemma](https://www.kaggle.com/models/google/paligemma){ data-model="paligemma" }
- [負責任的生成式 AI 工具組][responsible-ai]

**使用條款：** [使用條款][gemma-terms]

**作者：** Google

## 模型資訊

### 模型摘要

#### 說明

PaliGemma 是一款多功能、輕量的視覺語言模型（VLM），設計靈感來自
[PaLI-3](https://arxiv.org/abs/2310.09199)，並以
[SigLIP 視覺模型](https://arxiv.org/abs/2303.15343)與 [Gemma
語言模型](https://arxiv.org/abs/2403.08295)等開放元件為基礎。它同時接受影像與
文字作為輸入，並產生文字作為輸出，支援多種語言。其設計目標是在影像與短片
圖說、視覺問答、文字閱讀、物件偵測與物件分割等廣泛的視覺語言任務上，達到同級
最佳的微調效能。

#### 模型架構

PaliGemma 由 [Transformer
解碼器](https://arxiv.org/abs/1706.03762)與 [Vision Transformer 影像
編碼器](https://arxiv.org/abs/2010.11929)組成，參數量共計 30 億。文字解碼器以
[Gemma-2B](https://www.kaggle.com/models/google/gemma) 初始化；影像編碼器以
[SigLIP-So400m/14](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb)
初始化。PaliGemma 的訓練遵循 PaLI-3 的配方。

#### 輸入與輸出

- **輸入：**影像與文字字串，例如為影像加上圖說的提示，或是一個問題。
- **輸出：**針對輸入產生的文字，例如影像的圖說、問題的答案、物件邊界框座標清單，或分割碼字。

#### 引用

```python
@article{
    title={PaliGemma: A versatile 3B VLM for transfer},
    author={Lucas Beyer and Andreas Steiner and André Susano Pinto and Alexander Kolesnikov and Xiao Wang* and Daniel Salz and Maxim Neumann and Ibrahim Alabdulmohsin and Michael Tschannen and Emanuele Bugliarello and Thomas Unterthiner and Daniel Keysers and Skanda Koppula and Fangyu Liu and Adam Grycner and Alexey Gritsenko and Neil Houlsby and Manoj Kumar and Keran Rong and Julian Eisenschlos and Rishabh Kabra and Matthias Bauer and Matko Bošnjak and Xi Chen and Matthias Minderer and Paul Voigtlaender and Ioana Bica and Ivana Balazevic and Joan Puigcerver and Pinelopi Papalampidi and Olivier Henaff and Xi Xiong and Radu Soricut and Jeremiah Harmsen and Xiaohua Zhai},
    year={2024},
    journal={arXiv preprint arXiv:2407.07726}
}
```

### 模型資料

#### 預訓練資料集

PaliGemma 在以下資料集混合上進行預訓練：

- **WebLI：**[WebLI (Web Language Image)](https://arxiv.org/abs/2209.06794) 是一個來自公開網路的網路規模多語言影像文字資料集。我們使用了多種 WebLI 資料分割，以獲得多樣的模型能力，例如視覺語意理解、物件定位、具視覺情境的文字理解、多語言能力等。
- **CC3M-35L：**來自網頁的精選英文影像-alt_text 配對（[Sharma et
  al., 2018](https://aclanthology.org/P18-1238/)）。我們使用 [Google Cloud
  Translation API](https://cloud.google.com/translate) 將其翻譯成另外 34 種語言。
- **VQ²A-CC3M-35L/VQG-CC3M-35L：**VQ2A-CC3M 的子集（[Changpinyo et al.,
  2022a](https://aclanthology.org/2022.naacl-main.142/)），使用 [Google Cloud
  Translation API](https://cloud.google.com/translate) 翻譯成與 CC3M-35L 相同的另外 34 種語言。
- **OpenImages：**以人工設計的規則在 [OpenImages 資料集](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) 上產生的偵測與物件感知問答（[Piergiovanni et al. 2022](https://arxiv.org/abs/2209.04372)）。
- **WIT：**從維基百科收集的影像與文字（[Srinivasan et al.,
  2021](https://arxiv.org/abs/2103.01913)）。

#### 資料責任篩選

我們對 WebLI 套用了以下篩選器，目標是讓 PaliGemma 在乾淨的資料上訓練：

- **色情影像篩選：**此篩選器會移除被判定為色情性質的影像。
- **文字安全篩選：**我們會辨識並篩除搭配不安全文字的影像。不安全文字是指任何被判定為含有或涉及兒童性虐待影像（CSAI）、色情、粗俗用語，或其他冒犯性內容的文字。
- **文字毒性篩選：**我們進一步使用 [Perspective
  API](https://perspectiveapi.com/) 來辨識並篩除搭配被判定為侮辱、淫穢、仇恨或其他有毒文字的影像。
- **文字個人資訊篩選：**我們使用 [Cloud Data Loss Prevention
  (DLP) API](https://cloud.google.com/security/products/dlp) 篩除特定的個人資訊與其他敏感資料，以保護個人隱私。社會安全號碼等識別碼與[其他敏感資訊類型](https://cloud.google.com/sensitive-data-protection/docs/high-sensitivity-infotypes-reference?_gl=1*jg604m*_ga*ODk5MzA3ODQyLjE3MTAzMzQ3NTk.*_ga_WH2QY8WWF5*MTcxMDUxNTkxMS4yLjEuMTcxMDUxNjA2NC4wLjAuMA..&_ga=2.172110058.-899307842.1710334759)皆已移除。
- **其他方法：**依據我們的政策與實務，進行內容品質與安全性的篩選。

## 實作資訊

### 硬體

PaliGemma 使用最新一代的 Tensor Processing Unit（TPU）
硬體（TPUv5e）進行訓練。

### 軟體

訓練使用 [JAX](https://github.com/jax-ml/jax)、
[Flax](https://github.com/google/flax)、
[TFDS](https://github.com/tensorflow/datasets) 與
[`big_vision`](https://github.com/google-research/big_vision) 完成。

JAX 讓研究人員能夠利用最新一代的硬體（包括 TPU），以更快、更有效率的方式
訓練大型模型。

TFDS 用於存取資料集，Flax 用於模型架構。PaliGemma 的微調程式碼與推論程式碼
已發布於 `big_vision` GitHub 存放庫。

## 評估資訊

### 基準測試結果

為了驗證 PaliGemma 在各種學術任務上的遷移能力，我們在每個任務上對預訓練
模型進行微調。此外，我們也以遷移任務的混合資料訓練混合模型。我們在不同
解析度下報告結果，以呈現哪些任務能從提高解析度中受益。重要的是，這些任務或
資料集都不屬於預訓練資料混合的一部分，其影像已明確從網路規模的預訓練資料中
移除。

#### 單一任務（在單一任務上微調）

| 基準測試（訓練分割） | 指標（分割） | pt-224 | pt-448 | pt-896 |
|---|---|---|---|---|
| 圖說 |||||
| [COCO captions](https://cocodataset.org/#home) (train+restval) | CIDEr (val) | 141.92 | 144.60 |
| [NoCaps](https://nocaps.org/)（COCO captions 遷移評估） | CIDEr (val) | 121.72 | 123.58 |
| [COCO-35L](https://arxiv.org/pdf/2205.12522) (train) | CIDEr dev (en/avg-34/avg) | 139.2 115.8 116.4 | 141.2 118.0 118.6 |
| [XM3600](https://arxiv.org/pdf/2205.12522)（COCO-35L 遷移評估） | CIDEr dev (en/avg-34/avg) | 78.1 41.3 42.4 | 80.0 41.9 42.9 |
| [TextCaps](https://textvqa.org/textcaps/) (train) | CIDEr (val) | 127.48 | 153.94 |
| [SciCap](https://arxiv.org/abs/2110.11624)（首句、無子圖）(train+val) | CIDEr/BLEU-4 (test) | 162.25 0.192 | 181.49 0.211 |
| [Screen2words](https://arxiv.org/abs/2108.03353) (train+dev) | CIDEr (test) | 117.57 | 119.59 |
| [Widget Captioning](https://arxiv.org/abs/2010.04295) (train+dev) | CIDEr (test) | 136.07 | 148.36 |
| [VQAv2](https://visualqa.org/index.html) (train+validation) | 準確度（Test server - std） | 83.19 | 85.64 |
| [MMVP](https://arxiv.org/abs/2401.06209)（VQAv2 遷移評估） | 配對準確度 | 47.33 | 45.33 |
| [POPE](https://arxiv.org/abs/2305.10355)（VQAv2 遷移評估） | 準確度（random/popular/adversarial） | 87.80 85.87 84.27 | 88.23 86.77 85.90 |
| [OKVQA](https://okvqa.allenai.org/) (train) | 準確度（val） | 63.54 | 63.15 |
| [A-OKVQA](https://allenai.org/project/a-okvqa/home) (MC) (train+val) | 準確度（Test server） | 76.37 | 76.90 |
| [A-OKVQA](https://allenai.org/project/a-okvqa/home) (DA) (train+val) | 準確度（Test server） | 61.85 | 63.22 |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html) (train_balanced+val_balanced) | 準確度（testdev balanced） | 65.61 | 67.03 |
| [xGQA](https://aclanthology.org/2022.findings-acl.196/)（GQA 遷移評估） | 平均準確度（bn, de, en, id, ko, pt, ru, zh） | 58.37 | 59.07 |
| [NLVR2](https://lil.nlp.cornell.edu/nlvr/) (train+dev) | 準確度（test） | 90.02 | 88.93 |
| [MaRVL](https://marvl-challenge.github.io/)（NLVR2 遷移評估） | 平均準確度（test）(id, sw, ta, tr, zh) | 80.57 | 76.78 |
| [AI2D](https://allenai.org/data/diagrams) (train) | 準確度（test） | 72.12 | 73.28 |
| [ScienceQA](https://scienceqa.github.io/)（Img 子集、無 CoT）(train+val) | 準確度（test） | 95.39 | 95.93 |
| [RSVQA-LR](https://zenodo.org/records/6344334)（非數值）(train+val) | 平均準確度（test） | 92.65 | 93.11 |
| [RSVQA-HR](https://zenodo.org/records/6344367)（非數值）(train+val) | 平均準確度（test/test2） | 92.61 90.58 | 92.79 90.54 |
| [ChartQA](https://arxiv.org/abs/2203.10244) (human+aug)x(train+val) | 平均寬鬆準確度（test_human, test_aug） | 57.08 | 71.36 |
| [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/) (train+val) | 準確度（Test server - std） | 73.7 | 75.52 |
| [TallyQA](https://arxiv.org/abs/1810.12440) (train) | 準確度（test_simple/test_complex） | 81.72 69.56 | 84.86 72.27 |
| [OCR-VQA](https://ocr-vqa.github.io/) (train+val) | 準確度（test） | 72.32 | 74.61 | 74.93 |
| [TextVQA](https://textvqa.org/) (train+val) | 準確度（Test server - std） | 55.47 | 73.15 | 76.48 |
| [DocVQA](https://www.docvqa.org/) (train+val) | ANLS（Test server） | 43.74 | 78.02 | 84.77 |
| [Infographic VQA](https://openaccess.thecvf.com/content/WACV2022/papers/Mathew_InfographicVQA_WACV_2022_paper.pdf) (train+val) | ANLS（Test server） | 28.46 | 40.47 | 47.75 |
| [SceneText VQA](https://arxiv.org/abs/1905.13648) (train+val) | ANLS（Test server） | 63.29 | 81.82 | 84.40 |
| [RefCOCO](https://arxiv.org/abs/1608.00272)（合併 refcoco、refcoco+、refcocog，排除驗證與測試影像） | MIoU（驗證）refcoco/refcoco+/refcocog | 73.40 68.32 67.65 | 75.57 69.76 70.17 | 76.94 72.18 72.22 |
| MSR-VTT（圖說） | CIDEr (test) | 70.54 |
| MSR-VTT（問答） | 準確度（test） | 50.09 |
| ActivityNet（圖說） | CIDEr (test) | 34.62 |
| ActivityNet（問答） | 準確度（test） | 50.78 |
| VATEX（圖說） | CIDEr (test) | 79.73 |
| MSVD（問答） | 準確度（test） | 60.22 |

#### 混合模型（在遷移任務混合上微調）

| 基準測試 | 指標（分割） | mix-224 | mix-448 |
|---|---|---|---|
| [MMVP](https://arxiv.org/abs/2401.06209) | 配對準確度 | 46.00 | 45.33 |
| [POPE](https://arxiv.org/abs/2305.10355) | 準確度（random/popular/adversarial） | 88.00 86.63 85.67 | 89.37 88.40 87.47 |

## 倫理與安全

### 評估方法

我們的評估方法包括結構化評估，以及針對相關內容政策的內部紅隊測試。紅隊測試
由多個不同的團隊執行，各自有不同的目標與人工評估指標。這些模型已針對與倫理
和安全相關的多個類別進行評估，包括：

- 針對涵蓋兒童安全、內容安全與代表性傷害的提示進行人工評估。有關評估方法的更多細節，請參閱 [Gemma 模型卡](../gemma-1/index.md#評測方法)，但採用的是影像圖說與視覺問答設定。
- 影像轉文字基準測試評估：以相關學術資料集為基準進行評估，例如 FairFace 資料集（[Karkkainen et al., 2021](https://arxiv.org/abs/1908.04913)）。

### 評估結果

- 倫理與安全評估的人工評估結果，均在符合[內部政策](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11)可接受門檻內，類別包括兒童安全、內容安全與代表性傷害。
- 除了嚴謹的內部評估外，我們也會使用 Perspective API（門檻為 0.8）來測量以 FairFace 資料集影像產生的圖說中的毒性、髒話與其他潛在問題。我們報告各感知性別、種族與年齡屬性子群中觀察到的最大值與中位數。

| 指標 | 感知性別 || 種族 || 年齡組 ||
|---|---|---|---|---|---|---|
|   | 最大值 | 中位數 | 最大值 | 中位數 | 最大值 | 中位數 |
| 毒性 | 0.04% | 0.03% | 0.08% | 0.00% | 0.09% | 0.00% |
| 身分攻擊 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 侮辱 | 0.06% | 0.04% | 0.09% | 0.07% | 0.16% | 0.00% |
| 威脅 | 0.06% | 0.05% | 0.14% | 0.05% | 0.17% | 0.00% |
| 髒話 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## 用途與限制

### 預期用途

開放的視覺語言模型（VLM）在各產業與領域中有廣泛的應用。以下潛在用途清單並
不全面。此清單的目的是提供背景資訊，說明模型建立者在模型訓練與開發過程中
考慮過的可能使用情境。Gemma 模型的禁用用途列於 [Gemma 禁用用途政策][prohibited-use]。

針對特定視覺語言任務微調：

- 預訓練模型可針對影像圖說、短片圖說、視覺問答、文字閱讀、物件偵測與物件分割等廣泛的視覺語言任務進行微調。
- 預訓練模型可針對特定領域進行微調，例如遙感問答、視障人士的視覺問題、科學問答、描述 UI 元件功能。
- 預訓練模型可針對非文字輸出（例如邊界框或分割遮罩）的任務進行微調。

視覺語言研究：

- 預訓練模型與微調模型可作為研究人員的基礎，用於實驗 VLM 技術、開發演算法，並促進該領域的發展。

### 倫理考量與風險

視覺語言模型（VLM）的開發引發了多項倫理疑慮。在建立開放模型時，我們已仔細
考量以下幾點：

- 偏見與公平
    - 以大規模真實世界影像文字資料訓練的 VLM，可能反映訓練資料中嵌入的社會文化偏見。這些模型已接受嚴格審查，本卡片說明了輸入資料的預處理，以及後續的評估結果。
- 錯誤資訊與濫用
    - VLM 可能被濫用，以產生虛假、誤導或有害的文字。
    - 模型的負責任使用指南，請參閱[負責任的生成式 AI 工具組][responsible-ai]。
- 透明度與課責
    - 本模型卡摘要了模型的架構、能力、限制與評估流程等細節。
    - 負責任開發的開放模型，透過讓 AI 生態系中的開發者與研究人員都能取得 VLM 技術，提供了分享創新的機會。

已識別的風險與緩解措施：

- **偏見的延續：**建議在模型訓練、微調與其他使用情境中持續監控（使用評估指標、人工審查），並探索去偏見技術。
- **有害內容的產生：**內容安全機制與指南不可或缺。建議開發者保持謹慎，並根據其產品政策與應用使用情境，實施適當的內容安全防護。
- **惡意用途的濫用：**技術限制，以及開發者與終端使用者的教育，有助於緩解 LLM 的惡意應用。我們也提供教育資源與回報機制，供使用者檢舉濫用情形：請參閱[負責任的生成式 AI 工具組][responsible-ai]。Gemma 模型的禁用用途列於 [Gemma 禁用用途政策][prohibited-use]。
- **隱私侵犯：**模型訓練所用的資料已篩除特定的個人資訊與敏感資料。建議開發者使用隱私保護技術，遵守隱私法規。

### 限制

- 底層 Gemma 模型的大多數限制仍然適用：
    - VLM 更擅長處理能以明確提示與指示框定的任務。開放式或高度複雜的任務可能具有挑戰性。
    - 自然語言本質上很複雜。VLM 可能難以掌握細微的語義差別、諷刺或比喻。
    - VLM 根據從訓練資料集學到的資訊產生回應，但它們不是知識庫。它們可能會產生錯誤或過時的事實陳述。
    - VLM 依賴語言與影像中的統計模式。在某些情況下，它們可能缺乏運用常識推理的能力。
- PaliGemma 首要設計目標是作為可微調至專門任務的通用預訓練模型。因此，其「開箱即用」或「零樣本」效能可能不如專為通用用途設計的模型。
- PaliGemma 不是多輪對話機器人。它專為單輪影像與文字輸入而設計。
