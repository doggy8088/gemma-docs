---
title: ShieldGemma 2 模型卡
description: ShieldGemma 2 是在 Gemma 3 的 4B IT 檢查點上訓練的模型，用於跨關鍵類別的影像安全分類：接收影像，依政策輸出安全標籤。
models: [shieldgemma]
source_url: "https://ai.google.dev/gemma/docs/shieldgemma/model_card_2.md.txt"
---
# ShieldGemma 模型卡

**模型頁面** : [ShieldGemma](index.md)

**資源與技術文件**:

- [Responsible Generative AI Toolkit][responsible-ai]
- [Kaggle 上的 ShieldGemma](https://www.kaggle.com/models/google/shieldgemma){ data-model="shieldgemma" }
- [Hugging Face Hub 上的 ShieldGemma](https://huggingface.co/models?search=shieldgemma){ data-model="shieldgemma" }

**使用條款** : [條款][gemma-terms]

**作者**: Google

## 模型資訊

ShieldGemma 2 是在 Gemma 3 的 4B IT 檢查點上訓練的模型，用於跨關鍵類別的影像安全分類：接收影像，依政策輸出安全標籤。

### 描述

ShieldGemma 2 建構於 Gemma 3 之上，是一個 40 億（4B）參數模型，用於檢查合成與自然影像在關鍵類別中的安全性，協助您建構穩健的資料集與模型。隨著這個 Gemma 系列模型的加入，研究人員與開發者現在能將有害內容影響其生成式 AI 應用程式的風險降至最低，涵蓋關鍵[傷害類型](model-card-v2.md#傷害類型)。

我們建議將 ShieldGemma 2 用作視覺語言模型的輸入過濾器，或影像生成系統的輸出過濾器。為了訓練出穩健的影像安全模型，我們整理了自然與合成影像的訓練資料集，並對 Gemma 3 進行指令微調，展現出強勁效能。我們將安全政策與不同基準測試進行比較，並發布一份納入第三方基準測試的技術報告。

### 輸入與輸出

- **輸入：** 影像與包含政策定義的提示指令。
- **輸出：** 'Yes' 或 'No' 詞元的機率。'Yes' 詞元的分數越高，表示模型越有信心認定內容違反所含政策；'No' 詞元則相反。兩個分數皆會提供，以便進行分類指標計算與校準。

#### 在提示中使用的準則

以下是我們的提示指令 `[Prompt Instruction]`：

```text
You are a policy expert trying to help determine whether a generated or real
image is in violation of the safety policy below. Your decision must very
clearly follow the specific language used in the safety policy and should not
include any interpretation outside of the safety policy.

<BEGIN_SAFETY_POLICY>
{policy}
</END_SAFETY_POLICY>

Does the image violate the above policy? Your Answer must start with 'Yes' or 'No'.
```

##### 傷害類型

**使用場景：影像分類**

| 傷害類型 | 準則 |
|---|---|
| 色情露骨內容 | `"No Sexually Explicit content"`：影像不得包含描繪露骨或圖像化性行為的內容（例如色情、情色裸露、對強暴或性侵犯的描繪）。 |
| 危險內容 | `"No Dangerous content"`：影像不得包含促成或鼓勵可能造成現實世界傷害之活動的內容（例如製造槍械與爆裂物裝置、宣揚恐怖主義、自殺指示）。 |
| 暴力內容 | `"No Violence/Gore content"`：影像不得包含描繪令人震驚、聳動或無端暴力的內容（例如大量血腥、對動物無端的暴力、極端傷害或死亡瞬間）。 |

### 引用

```python
@misc{zeng2025shieldgemma2robusttractable,
      title={ShieldGemma 2: Robust and Tractable Image Content Moderation},
      author={Wenjun Zeng and Dana Kurniawan and Ryan Mullins and Yuchi Liu and Tamoghna Saha and Dirichi Ike-Njoku and Jindong Gu and Yiwen Song and Cai Xu and Jingjing Zhou and Aparna Joshi and Shravan Dheep and Mani Malek and Hamid Palangi and Joon Baek and Rick Pereira and Karthik Narasimhan},
      year={2025},
      eprint={2504.01081},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.01081},
}
```

## 模型資料

用於模型訓練的資料，以及資料處理方式。

### 訓練資料集

我們的訓練資料集包含自然影像與合成影像。自然影像方面，我們從 WebLI（Web Language and Image）
資料集中抽取與安全任務相關的影像子集。合成影像方面，我們使用內部資料生成管線，對提示與對應影像進行可控生成，以平衡影像的多樣性與嚴重程度。在本研究中，傷害類型限於危險、色情露骨與暴力內容，且僅限英文。此外，對抗性與子主題採用對應各政策的分類架構（taxonomy）進行組織，並涵蓋不同人口特徵、情境與地區面向。

## 資料前處理

以下是套用於訓練資料的關鍵資料清理與過濾方法：CSAM 過濾：在資料準備過程中套用 CSAM（兒童性虐待材料）過濾，以確保排除非法內容。

## 實作資訊

### 硬體

ShieldGemma 2 使用最新一代
[Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) 硬體（TPUv5e）進行訓練，更多詳情請參閱
[Gemma 3 模型卡](../gemma-3/index.md)。

### 軟體

訓練使用 [JAX](https://github.com/jax-ml/jax) 與 [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)。更多
詳情請參閱 [Gemma 3 模型卡](../gemma-3/index.md)。

## 評估

### 基準測試結果

ShieldGemma 2 4B 在內部與外部資料集上皆有評估。我們的內部資料集透過內部影像資料整理管線合成生成。該管線包含問題定義、安全分類架構生成、影像查詢生成、影像生成、屬性分析、標籤品質驗證等關鍵步驟。每個傷害政策約有 500 個範例。色情、危險內容、暴力的正例比例分別為 39%、67%、32%。我們也將發布一份包含外部資料集評估的技術報告。

**內部基準測試評估結果**

| 模型 | 色情露骨 | 危險內容 | 暴力與血腥 |
|---|---|---|---|
| LlavaGuard 7B | 47.6/93.1/63.0 | 67.8/47.2/55.7 | 36.8/100.0/53.8 |
| GPT-4o mini | 68.3/97.7/80.3 | 84.4/99.0/91.0 | 40.2/100.0/57.3 |
| Gemma-3-4B-IT | 77.7/87.9/82.5 | 75.9/94.5/84.2 | 78.2/82.2/80.1 |
| ShieldGemma-2-Image-4B | 87.6/89.7/88.6 | 95.6/91.9/93.7 | 80.3/90.4/85.0 |

## 道德與安全

### 評估方法

雖然 ShieldGemma 模型是生成式模型，但它們設計為以 *scoring mode*（評分模式）執行，預測下一個詞元為 `Yes`
或 `No` 的機率。因此，安全評估主要聚焦於輸出有效的影像安全標籤。

### 評估結果

這些模型已就道德、安全與公平性考量進行評估，並符合內部準則。與基準測試相比，評估資料集經過反覆迭代，並依據多樣化的分類架構進行平衡。影像安全標籤亦經過人工標註，並針對模型未能處理的使用場景進行檢查，使我們能在多輪評估中持續改進。

## 用途與限制

這些模型存在使用者應注意的若干限制。

### 預期用途

ShieldGemma 2 預期用作安全內容審核器，可用於人類使用者輸入、模型輸出，或兩者皆用。這些模型是
[Responsible Generative AI Toolkit][responsible-ai] 的一部分，該工具組是一套建議、工具、資料集與模型，旨在提升作為 Gemma 生態系一員之 AI 應用的安全性。

### 限制

大型語言模型的所有常見限制皆適用，更多詳情請參閱
[Gemma 3 模型卡](../gemma-3/index.md)。此外，可用於評估內容審核的基準測試有限，因此
訓練與評估資料可能無法代表真實世界場景。

ShieldGemma 2 對使用者提供的特定安全原則描述高度敏感，在需要充分理解語言歧義與細微差異的情況下，表現可能難以預測。

與 Gemma 生態系的其他模型一樣，ShieldGemma 受 Google 的 [禁止使用政策][prohibited-use] 約束。

### 道德考量與風險

大型語言模型（LLM）的開發引發了多項道德疑慮。
我們在開發這些模型時已審慎考量多個面向。

更多詳情請參閱 [Gemma 3 模型卡](../gemma-3/index.md)。

### 優勢

在發布時，相較於同等規模的模型，此系列模型提供了從頭為負責任 AI 開發而設計的高效能開放大型語言模型實作。

根據本文件所述的基準測試評估指標，這些模型在與其他規模相當的開放模型替代方案相比，已展現出更優異的效能。
