---
title: PaliGemma 2 模型卡
description: PaliGemma 2 是 PaliGemma 視覺語言模型（VLM）的更新版本，納入了 Gemma 2 模型的能力。
models: [paligemma]
source_url: "https://ai.google.dev/gemma/docs/paligemma/model-card-2.md.txt"
---
# 模型卡

**模型頁面：** [PaliGemma](index.md)

**資源與技術文件：**

- [Kaggle 上的 PaliGemma 2](https://www.kaggle.com/models/google/paligemma-2){ data-model="paligemma" }
- [負責任的生成式 AI 工具組][responsible-ai]

**使用條款：** [使用條款][gemma-terms]

**作者：** Google

## 模型資訊

### 模型摘要

PaliGemma 2 是 [PaliGemma](https://arxiv.org/abs/2407.07726)
視覺語言模型（VLM）的更新版本，納入了
[Gemma 2](https://arxiv.org/abs/2408.00118) 模型的能力。PaliGemma 系列
模型的設計靈感來自 [PaLI-3](https://arxiv.org/abs/2310.09199)，並以
[SigLIP](https://arxiv.org/abs/2303.15343) 視覺模型與
[Gemma 2](https://arxiv.org/abs/2408.00118) 語言模型等開放元件為基礎。它同時
接受影像與文字作為輸入，並產生文字作為輸出，支援多種語言。其設計目標是在
影像與短片圖說、視覺問答、文字閱讀、物件偵測與物件分割等廣泛的視覺語言任務上，
達到同級最佳的微調效能。

#### 模型架構

PaliGemma 2 由
[Transformer 解碼器](https://arxiv.org/abs/1706.03762)與
[Vision Transformer 影像編碼器](https://arxiv.org/abs/2010.11929)組成。
文字解碼器以 [Gemma 2](../gemma-4/index.md) 初始化，提供 2B、9B 與 27B
參數量版本。影像編碼器以
[SigLIP-So400m/14](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb)
初始化。與原始 PaliGemma 模型類似，PaliGemma 2 的訓練遵循
[PaLI-3](https://arxiv.org/abs/2310.09199) 的配方。

#### 輸入與輸出

- **輸入：**影像與文字字串，例如為影像加上圖說的提示，或是一個問題。
- **輸出：**針對輸入產生的文字，例如影像的圖說、問題的答案、物件邊界框座標清單，或分割碼字。

#### 引用

```python
@article{
    title={PaliGemma 2: A Family of Versatile VLMs for Transfer},
    author={Andreas Steiner and André Susano Pinto and Michael Tschannen and Daniel Keysers and Xiao Wang and Yonatan Bitton and Alexey Gritsenko and Matthias Minderer and Anthony Sherbondy and Shangbang Long and Siyang Qin and Reeve Ingle and Emanuele Bugliarello and Sahar Kazemzadeh and Thomas Mesnard and Ibrahim Alabdulmohsin and Lucas Beyer and Xiaohua Zhai},
    year={2024},
    journal={arXiv preprint arXiv:2412.03555}
}
```

### 模型資料

#### 預訓練資料集

PaliGemma 2 在以下資料集混合上進行預訓練：

- **WebLI：**[WebLI (Web Language Image)](https://arxiv.org/abs/2209.06794) 是一個來自公開網路的網路規模多語言影像文字資料集。我們使用了多種 WebLI 資料分割，以獲得多樣的模型能力，例如視覺語意理解、物件定位、具視覺情境的文字理解，以及多語言能力。
- **CC3M-35L：**來自網頁的精選英文影像-alt_text 配對（[Sharma et al., 2018](https://aclanthology.org/P18-1238/)）。我們使用 [Google Cloud Translation API](https://cloud.google.com/translate) 將其翻譯成另外 34 種語言。
- **VQ²A-CC3M-35L/VQG-CC3M-35L：**VQ2A-CC3M 的子集（[Changpinyo et al., 2022a](https://aclanthology.org/2022.naacl-main.142/)），使用 [Google Cloud Translation API](https://cloud.google.com/translate) 翻譯成與 CC3M-35L 相同的另外 34 種語言。
- **OpenImages：**以人工設計的規則在 [OpenImages 資料集](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) 上產生的偵測與物件感知問答（[Piergiovanni et al. 2022](https://arxiv.org/abs/2209.04372)）。
- **WIT：**從維基百科收集的影像與文字（[Srinivasan et al., 2021](https://arxiv.org/abs/2103.01913)）。

PaliGemma 2 以 Gemma 2 為基礎，Gemma 2 的預訓練資料集資訊請參閱
[Gemma 2 模型卡](../gemma-2/index.md)。

#### 資料責任篩選

我們對 WebLI 套用了以下篩選器，目標是讓 PaliGemma
2 在安全且負責任的資料上訓練：

- **色情影像篩選：**此篩選器會移除被判定為色情性質的影像。
- **文字安全篩選：**我們會辨識並篩除搭配不安全文字的影像。不安全文字是指任何被判定為含有或涉及兒童性虐待影像（CSAI）、色情、粗俗用語，或其他冒犯性內容的文字。
- **文字毒性篩選：**我們進一步使用 [Perspective
  API](https://perspectiveapi.com/) 來辨識並篩除搭配被判定為侮辱、淫穢、仇恨或其他有毒文字的影像。
- **文字個人資訊篩選：**我們使用 [Cloud Data Loss Prevention
  (DLP) API](https://cloud.google.com/security/products/dlp) 篩除特定的個人資訊與其他敏感資料，以保護個人隱私。社會安全號碼等識別碼與[其他敏感資訊類型](https://cloud.google.com/sensitive-data-protection/docs/high-sensitivity-infotypes-reference?_gl=1*jg604m*_ga*ODk5MzA3ODQyLjE3MTAzMzQ3NTk.*_ga_WH2QY8WWF5*MTcxMDUxNTkxMS4yLjEuMTcxMDUxNjA2NC4wLjAuMA..&_ga=2.172110058.-899307842.1710334759)皆已移除。
- **其他方法：**依據我們的政策與實務，進行內容品質與安全性的篩選。

## 實作資訊

### 硬體

PaliGemma 2 使用最新一代的 Tensor Processing Unit（TPU）
硬體（TPUv5e）進行訓練。

### 軟體

訓練使用 [JAX](https://github.com/jax-ml/jax)、
[Flax](https://github.com/google/flax)、
[TFDS](https://github.com/tensorflow/datasets) 與
[`big_vision`](https://github.com/google-research/big_vision) 完成。

JAX 讓研究人員能夠利用最新一代的硬體（包括 TPU），以更快、更有效率的方式
訓練大型模型。

TFDS 用於存取資料集，Flax 用於模型架構。PaliGemma 2 的微調程式碼與推論程式碼
已發布於 `big_vision` GitHub 存放庫。

## 評估資訊

### 基準測試結果

為了驗證 PaliGemma 2 在各種學術任務上的遷移能力，我們在每個任務上對預訓練
模型進行微調。此外，我們也以遷移任務的混合資料訓練混合模型。我們在不同
解析度下報告結果，以呈現哪些任務能從提高解析度中受益。重要的是，這些任務或
資料集都不屬於預訓練資料混合的一部分，其影像已明確從網路規模的預訓練資料中
移除。

#### PaliGemma 2 按模型解析度與大小的結果

| 基準測試 | 224-3B | 224-10B | 224-28B | 448-3B | 448-10B | 448-28B |
|---|---|---|---|---|---|---|
| [AI2D](https://allenai.org/data/diagrams) | 74.7 | 83.1 | 83.2 | 76.0 | 84.4 | 84.6 |
| [AOKVQA-DA](https://allenai.org/project/a-okvqa/home) (val) | 64.2 | 68.9 | 70.2 | 67.9 | 70.8 | 71.2 |
| [AOKVQA-MC](https://allenai.org/project/a-okvqa/home) (val) | 79.7 | 83.7 | 84.7 | 82.5 | 85.9 | 87.0 |
| [ActivityNet-CAP](https://paperswithcode.com/dataset/activitynet-captions) | 34.2 | 35.9 | - | - | - | - |
| [ActivityNet-QA](https://arxiv.org/abs/1906.02467) | 51.3 | 53.2 | - | - | - | - |
| [COCO-35L](https://arxiv.org/pdf/2205.12522) (avg34) | 113.9 | 115.8 | 116.5 | 115.8 | 117.2 | 117.2 |
| [COCO-35L](https://arxiv.org/pdf/2205.12522) (en) | 138.4 | 140.8 | 142.4 | 140.4 | 142.4 | 142.3 |
| [COCOcap](https://cocodataset.org/#home) | 141.3 | 143.7 | 144.0 | 143.4 | 145.0 | 145.2 |
| [ChartQA](https://arxiv.org/abs/2203.10244) (avg) | 74.4 | 74.2 | 68.9 | 89.2 | 90.1 | 85.1 |
| [ChartQA](https://arxiv.org/abs/2203.10244) (human) | 42.0 | 48.4 | 46.8 | 54.0 | 66.4 | 61.3 |
| [CountBenchQA](https://github.com/google-research/big_vision/blob/main/big_vision/datasets/countbenchqa/) | 81.0 | 84.0 | 86.4 | 82.0 | 85.3 | 87.4 |
| [DocVQA](https://www.docvqa.org/) (val) | 39.9 | 43.9 | 44.9 | 73.6 | 76.6 | 76.1 |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html) | 66.2 | 67.2 | 67.3 | 68.1 | 68.3 | 68.3 |
| [InfoVQA](https://arxiv.org/abs/2104.12756) (val) | 25.2 | 33.6 | 36.4 | 37.5 | 47.8 | 46.7 |
| [MARVL](https://marvl-challenge.github.io/) (avg5) | 83.5 | 89.5 | 90.6 | 82.7 | 89.1 | 89.7 |
| [MSRVTT-CAP](https://paperswithcode.com/dataset/msr-vtt) | 68.5 | 72.1 | - | - | - | - |
| [MSRVTT-QA](https://paperswithcode.com/dataset/msr-vtt) | 50.5 | 51.9 | - | - | - | - |
| [MSVD-QA](https://paperswithcode.com/dataset/msvd-qa) | 61.1 | 62.5 | - | - | - | - |
| [NLVR2](https://lil.nlp.cornell.edu/nlvr/) | 91.4 | 93.9 | 94.2 | 91.6 | 93.7 | 94.1 |
| [NoCaps](https://nocaps.org/) | 123.1 | 126.3 | 127.1 | 123.5 | 126.9 | 127.0 |
| [OCR-VQA](https://ocr-vqa.github.io/) | 73.4 | 74.7 | 75.3 | 75.7 | 76.3 | 76.6 |
| [OKVQA](https://okvqa.allenai.org/) | 64.2 | 68.0 | 71.2 | 64.1 | 68.6 | 70.6 |
| [RSVQA-hr](https://zenodo.org/records/6344367) (test) | 92.7 | 92.6 | 92.7 | 92.8 | 92.8 | 92.8 |
| [RSVQA-hr](https://zenodo.org/records/6344367) (test2) | 90.9 | 90.8 | 90.9 | 90.7 | 90.7 | 90.8 |
| [RSVQA-lr](https://zenodo.org/records/6344334) | 93.0 | 92.8 | 93.5 | 92.7 | 93.1 | 93.7 |
| [RefCOCO](https://arxiv.org/abs/1608.00272) (testA) | 75.7 | 77.2 | 76.8 | 78.6 | 79.7 | 79.3 |
| [RefCOCO](https://arxiv.org/abs/1608.00272) (testB) | 71.0 | 74.2 | 73.9 | 73.5 | 76.2 | 74.8 |
| [RefCOCO](https://arxiv.org/abs/1608.00272) (val) | 73.4 | 75.9 | 75.0 | 76.3 | 78.2 | 77.3 |
| [RefCOCO+](https://aclanthology.org/D14-1086) (testA) | 72.7 | 74.7 | 73.6 | 76.1 | 77.7 | 76.6 |
| [RefCOCO+](https://aclanthology.org/D14-1086) (testB) | 64.2 | 68.4 | 67.1 | 67.0 | 71.1 | 68.6 |
| [RefCOCO+](https://aclanthology.org/D14-1086) (val) | 68.6 | 72.0 | 70.3 | 72.1 | 74.4 | 72.8 |
| [RefCOCOg](https://arxiv.org/abs/1511.02283) (test) | 69.0 | 71.9 | 70.7 | 72.7 | 74.8 | 73.7 |
| [RefCOCOg](https://arxiv.org/abs/1511.02283) (val) | 68.3 | 71.4 | 70.5 | 72.3 | 74.4 | 73.0 |
| [ST-VQA](https://arxiv.org/abs/1905.13648) (val) | 61.9 | 64.3 | 65.1 | 80.5 | 82.0 | 81.8 |
| [SciCap](https://arxiv.org/abs/2110.11624) | 165.1 | 159.5 | 156.9 | 183.3 | 177.2 | 172.7 |
| [ScienceQA](https://scienceqa.github.io/) | 96.1 | 98.2 | 98.2 | 96.2 | 98.5 | 98.6 |
| [Screen2Words](https://arxiv.org/abs/2108.03353) | 113.3 | 117.8 | 122.8 | 114.0 | 119.1 | 123.4 |
| [TallyQA](https://arxiv.org/abs/1810.12440) (complex) | 70.3 | 73.4 | 74.2 | 73.6 | 76.7 | 76.8 |
| [TallyQA](https://arxiv.org/abs/1810.12440) (simple) | 81.8 | 83.2 | 83.4 | 85.3 | 86.2 | 85.7 |
| [TextCaps](https://textvqa.org/textcaps/) | 127.5 | 137.9 | 139.9 | 152.1 | 157.7 | 153.6 |
| [TextVQA](https://textvqa.org/) (val) | 59.6 | 64.0 | 64.7 | 75.2 | 76.6 | 76.2 |
| [VATEX](https://arxiv.org/abs/1904.03493) | 80.8 | 82.7 | - | - | - | - |
| [VQAv2](https://visualqa.org/index.html) (minival) | 83.0 | 84.3 | 84.5 | 84.8 | 85.8 | 85.8 |
| [VizWizVQA](https://vizwiz.org/tasks-and-datasets/vqa/) (val) | 76.4 | 78.1 | 78.7 | 77.5 | 78.6 | 78.9 |
| [WidgetCap](https://arxiv.org/abs/2010.04295) | 138.1 | 139.8 | 138.8 | 151.4 | 151.9 | 148.9 |
| [XM3600](https://arxiv.org/pdf/2205.12522) (avg35) | 42.8 | 44.5 | 45.2 | 43.2 | 44.6 | 45.2 |
| [XM3600](https://arxiv.org/pdf/2205.12522) (en) | 79.8 | 80.7 | 81.0 | 80.3 | 81.5 | 81.0 |
| [xGQA](https://aclanthology.org/2022.findings-acl.196/) (avg7) | 58.6 | 61.4 | 61.1 | 60.4 | 62.6 | 62.1 |

#### 其他基準測試

**[ICDAR 2015 Incidental](https://arxiv.org/abs/1511.09207)**

| 模型 | 精確度 | 召回率 | F1 |
|---|---|---|---|
| PaliGemma 2 3B | 81.88 | 70.73 | 75.9 |

**[Total-Text](https://paperswithcode.com/paper/total-text-a-comprehensive-dataset-for-scene)**

| 模型 | 精確度 | 召回率 | F1 |
|---|---|---|---|
| PaliGemma 2 3B | 73.8. | 74.54 | 74.17 |

**[FinTabNet](https://developer.ibm.com/data/fintabnet/)**

| 模型 | S-TEDS | TEDS | GriTS-Top | GriTS-Con |
|---|---|---|---|---|
| PaliGemma 2 3B | 99.18 | 98.94 | 99.43 | 99.21 |

**[PubTabNet](https://paperswithcode.com/dataset/pubtabnet)**

| 模型 | S-TEDS | TEDS | GriTS-Top | GriTS-Con |
|---|---|---|---|---|
| PaliGemma 2 3B | 97.6 | 97.31 | 97.99 | 97.84 |

**[GrandStaff](https://link.springer.com/article/10.1007/s10032-023-00432-z)**

| 模型 | CER | LER | SER |
|---|---|---|---|
| PaliGemma 2 3B | 1.6 | 6.7 | 2.3 |

**[PubChem](https://pmc.ncbi.nlm.nih.gov/articles/PMC7352161/)**

- PaliGemma 2 3B，全匹配：94.8

**[DOCCI](https://research.google/pubs/docci-descriptions-of-connected-and-contrasting-images/)**

| 模型 | avg#char | avg#sent | NES % |
|---|---|---|---|
| PaliGemma 2 3B | 529 | 7.74 | 28.42 |
| PaliGemma 2 10B | 521 | 7.45 | 20.27 |

- *avg#char*：平均字元數
- *avg#sent*：平均句子數
- *NES*：非蘊含句

**[MIMIC-CXR](https://paperswithcode.com/dataset/mimic-cxr)**

| 模型 | CIDEr | BLEU4 | Rouge-L | RadGraph F1 |
|---|---|---|---|---|
| PaliGemma 2 3B | 19.9% | 14.6% | 31.92% | 28.8% |
| PaliGemma 2 10B | 17.4% | 15% | 32.41% | 29.5% |

**[Visual Spatial Reasoning](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00566/116470/Visual-Spatial-Reasoning)**

| 模型 | VSR zeroshot split (test) | VSR random split (test) |
|---|---|---|
| PaliGemma 2 3B | 0.75 | 0.82 |
| PaliGemma 2 10B | 0.80 | 0.87 |

## 倫理與安全

### 評估方法

我們的評估方法包括針對相關內容政策的結構化倫理與安全評估，包括：

- 針對涵蓋兒童安全、內容安全與代表性傷害的提示進行人工評估。有關評估方法的更多細節，請參閱 [Gemma 模型卡](../gemma-1/index.md#評測方法)，但採用的是影像圖說與視覺問答設定。
- 影像轉文字基準測試評估：以相關學術資料集為基準進行評估，例如 FairFace 資料集（[Karkkainen et al., 2021](https://arxiv.org/abs/1908.04913)）。

### 評估結果

- 倫理與安全評估的人工評估結果，均在符合[內部政策](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11)可接受門檻內，類別包括兒童安全、內容安全與代表性傷害。
- 除了嚴謹的內部評估外，我們也會使用 Perspective API（門檻為 0.8）來測量以 FairFace 資料集影像產生的圖說中的毒性、髒話與其他潛在問題。我們報告各感知性別、種族與年齡屬性子群中觀察到的最大值與中位數。

| 指標 | 感知性別 ||| 種族 ||| 年齡組 |||
|---|---|---|---|---|---|---|---|---|---|
| 模型大小 | 3B | 10B | 28B | 3B | 10B | 28B | 3B | 10B | 28B |
|   | 最大值 |||||||||
| 毒性 | 0.14% | 0.15% | 0.19% | 0.29% | 0.39% | 0.39% | 0.26% | 0.18% | 0.32% |
| 身分攻擊 | 0.04% | 0.02% | 0.02% | 0.13% | 0.06% | 0.06% | 0.06% | 0.03% | 0.06% |
| 侮辱 | 0.17% | 0.25% | 0.17% | 0.37% | 0.52% | 0.52% | 0.27% | 0.39% | 0.24% |
| 威脅 | 0.55% | 0.43% | 0.57% | 0.83% | 0.48% | 0.48% | 0.64% | 0.43% | 0.64% |
| 髒話 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 毒性 | 0.13% | 0.10% | 0.18% | 0.07% | 0.07% | 0.14% | 0.12% | 0.08% | 0.12% |
| 身分攻擊 | 0.02% | 0.01% | 0.02% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 侮辱 | 0.15% | 0.23% | 0.14% | 0.14% | 0.17% | 0.13% | 0.09% | 0.18% | 0.16% |
| 威脅 | 0.35% | 0.27% | 0.41% | 0.28% | 0.19% | 0.42% | 0.27% | 0.31% | 0.40% |
| 髒話 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

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

- 底層 Gemma 2 模型的大多數限制仍然適用：
    - VLM 更擅長處理能以明確提示與指示框定的任務。開放式或高度複雜的任務可能具有挑戰性。
    - 自然語言本質上很複雜。VLM 可能難以掌握細微的語義差別、諷刺或比喻。
    - VLM 根據從訓練資料集學到的資訊產生回應，但它們不是知識庫。它們可能會產生錯誤或過時的事實陳述。
    - VLM 依賴語言與影像中的統計模式。在某些情況下，它們可能缺乏運用常識推理的能力。
- PaliGemma 2 首要設計目標是作為可微調至專門任務的通用預訓練模型。因此，其「開箱即用」或「零樣本」效能可能不如專為通用用途設計的模型。
- PaliGemma 2 不是多輪對話機器人。它專為單輪影像與文字輸入而設計。
