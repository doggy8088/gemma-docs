---
title: 使用 JAX 與 Flax 以 RecurrentGemma 執行推論
description: 使用 Google DeepMind 的 recurrentgemma 函式庫與 JAX，執行 RecurrentGemma 2B Instruct 的取樣與推論。
models: [recurrentgemma]
frameworks: [jax]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/recurrentgemma-inference.ipynb
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_inference.md.txt"
---
# 使用 JAX 與 Flax 以 RecurrentGemma 執行推論


本教學示範如何使用 [Google DeepMind 的 `recurrentgemma` 函式庫](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" }（以 [JAX](https://jax.readthedocs.io)（高效能數值運算函式庫）、[Flax](https://flax.readthedocs.io)（基於 JAX 的類神經網路函式庫）、[Orbax](https://orbax.readthedocs.io/)（基於 JAX 的訓練工具函式庫，例如檢查點）與 [SentencePiece](https://github.com/google/sentencepiece)（分詞/還原詞元函式庫）撰寫）對 [RecurrentGemma](index.md) 2B Instruct 模型執行基本的取樣/推論。雖然本筆記本未直接使用 Flax，但 Gemma 與 RecurrentGemma（Griffin 模型）正是用 Flax 建立的。

這本筆記本可在 Google Colab 上使用 T4 GPU 執行（前往**編輯** \> **筆記本設定** \> 在**硬體加速器**下方選取 **T4 GPU**）。

## 設定

以下各節說明準備筆記本以使用 RecurrentGemma 模型的步驟，包括模型存取、取得 API 金鑰，以及設定筆記本執行環境

### 設定 Kaggle 存取以使用 Gemma

要完成本教學，您首先需要依照與 [Gemma 設定](../../get-started/index.md)「類似」的設定說明進行，但有以下例外：

- 在 [kaggle.com](https://www.kaggle.com/models/google/recurrentgemma){ data-model="recurrentgemma" } 取得 RecurrentGemma（而非 Gemma）的存取權。
- 選取資源足以執行 RecurrentGemma 模型的 Colab 執行環境。
- 產生並設定 Kaggle 使用者名稱與 API 金鑰。

完成 RecurrentGemma 設定後，請繼續下一節，在 Colab 環境中設定環境變數。

### 設定環境變數

設定 `KAGGLE_USERNAME` 與 `KAGGLE_KEY` 的環境變數。當出現「是否授權存取？」的提示時，請同意提供密鑰存取權。

```python { .notebook-cell }
import os
from google.colab import userdata # `userdata` is a Colab API.

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### 安裝 `recurrentgemma` 函式庫

本筆記本聚焦於使用免費的 Colab GPU。若要啟用硬體加速，請點選**編輯** \> **筆記本設定** \> 選取 **T4 GPU** \> **儲存**。

接著，您需要從 [`github.com/google-deepmind/recurrentgemma`](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" } 安裝 Google DeepMind 的 `recurrentgemma` 函式庫。如果看到關於「pip 依賴解析器」的錯誤，通常可以忽略。

!!! Note
    安裝 `recurrentgemma` 時，也會一併安裝 [`flax`](https://flax.readthedocs.io)、核心 [`jax`](https://jax.readthedocs.io)、[`optax`](https://optax.readthedocs.io/en/latest/)（基於 JAX 的梯度處理與最佳化函式庫）、[`orbax`](https://orbax.readthedocs.io/) 與 [`sentencepiece`](https://github.com/google/sentencepiece)。


```bash { .notebook-cell }
pip install git+https://github.com/google-deepmind/recurrentgemma.git
```

## 載入並準備 RecurrentGemma 模型

1. 使用 [`kagglehub.model_download`](https://github.com/Kaggle/kagglehub/blob/bddefc718182282882b72f814d407d89e5d178c4/src/kagglehub/models.py#L12) 載入 RecurrentGemma 模型，它接受三個參數：

- `handle`：來自 Kaggle 的模型代碼
- `path`：（選用字串）本機路徑
- `force_download`：（選用布林值）強制重新下載模型

!!! Note
    請注意，`recurrentgemma-2b-it` 模型大小約為 3.85Gb。


```python { .notebook-cell }
RECURRENTGEMMA_VARIANT = '2b-it' # @param ['2b', '2b-it'] {type:"string"}

import kagglehub

RECURRENTGEMMA_PATH = kagglehub.model_download(f'google/recurrentgemma/flax/{RECURRENTGEMMA_VARIANT}')
```

``` { .no-copy .output }
Downloading from https://www.kaggle.com/api/v1/models/google/recurrentgemma/flax/2b-it/1/download...
100%|██████████| 3.85G/3.85G [00:52<00:00, 78.2MB/s]
Extracting model files...
```

```python { .notebook-cell }
print('RECURRENTGEMMA_PATH:', RECURRENTGEMMA_PATH)
```

``` { .no-copy .output }
RECURRENTGEMMA_PATH: /root/.cache/kagglehub/models/google/recurrentgemma/flax/2b-it/1
```

!!! Note
    上方的輸出路徑是模型權重與分詞器在本機的儲存位置，稍後會用到。


1. 檢查模型權重與分詞器的位置，然後設定路徑變數。分詞器目錄位於您下載模型的主目錄中，而模型權重則位於子目錄中。例如：

- `tokenizer.model` 檔案位於 `/LOCAL/PATH/TO/recurrentgemma/flax/2b-it/1`。
- 模型檢查點位於 `/LOCAL/PATH/TO/recurrentgemma/flax/2b-it/1/2b-it`。

```python { .notebook-cell }
CKPT_PATH = os.path.join(RECURRENTGEMMA_PATH, RECURRENTGEMMA_VARIANT)
TOKENIZER_PATH = os.path.join(RECURRENTGEMMA_PATH, 'tokenizer.model')
print('CKPT_PATH:', CKPT_PATH)
print('TOKENIZER_PATH:', TOKENIZER_PATH)
```

``` { .no-copy .output }
CKPT_PATH: /root/.cache/kagglehub/models/google/recurrentgemma/flax/2b-it/1/2b-it
TOKENIZER_PATH: /root/.cache/kagglehub/models/google/recurrentgemma/flax/2b-it/1/tokenizer.model
```
## 執行取樣/推論

1. 使用 [`recurrentgemma.jax.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31){ data-model="recurrentgemma" } 方法載入 RecurrentGemma 模型檢查點。`sharding` 參數設為 `"single_device"` 會將所有模型參數載入單一裝置。

```python { .notebook-cell }
import recurrentgemma
from recurrentgemma import jax as recurrentgemma

params = recurrentgemma.load_parameters(checkpoint_path=CKPT_PATH, sharding="single_device")
```

1. 載入 RecurrentGemma 模型分詞器，它是以 [`sentencepiece.SentencePieceProcessor`](https://github.com/google/sentencepiece/blob/4d6a1f41069c4636c51a5590f7578a0dbed83450/python/src/sentencepiece/__init__.py#L423) 建構的：

```python { .notebook-cell }
import sentencepiece as spm

vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)
```

``` { .no-copy .output }
True
```

1. 若要從 RecurrentGemma 模型檢查點自動載入正確的設定，請使用 [`recurrentgemma.GriffinConfig.from_flax_params_or_variables`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/common.py#L128){ data-model="recurrentgemma" }。接著，使用 [`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" } 具現化 [Griffin](https://arxiv.org/abs/2402.19427) 模型。

```python { .notebook-cell }
model_config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(
    flax_params_or_variables=params)

model = recurrentgemma.Griffin(model_config)
```

1. 在 RecurrentGemma 模型檢查點/權重與分詞器之上，使用 [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74){ data-model="recurrentgemma" } 建立 `sampler`：

```python { .notebook-cell }
sampler = recurrentgemma.Sampler(
    model=model,
    vocab=vocab,
    params=params,
)
```

1. 在 `prompt` 中寫入提示並執行推論。您可以調整 `total_generation_steps`（生成回應時執行的步數——本範例使用 `50` 以節省主機記憶體）。

!!! Note
    如果記憶體不足，請點選**執行階段** \> **中斷連線並刪除執行階段**，然後點選**執行階段** \> **全部執行**。


```python { .notebook-cell }
prompt = [
    "\n# 5+9=?",
]

reply = sampler(input_strings=prompt,
                total_generation_steps=50,
                )

for input_string, out_string in zip(prompt, reply.text):
    print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
```

``` { .no-copy .output }
/usr/local/lib/python3.10/dist-packages/jax/_src/interpreters/mlir.py:920: UserWarning: Some donated buffers were not usable: ShapedArray(int32[1,8]).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer_donation.
  warnings.warn("Some donated buffers were not usable:"
Prompt:

# 5+9=?
Output:


# Answer: 14

# Explanation: 5 + 9 = 14.
```

## 了解更多

- 您可以前往 GitHub 上的 Google DeepMind [`recurrentgemma` 函式庫](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" } 了解更多，其中包含本教學所使用方法與模組的 docstring，例如 [`recurrentgemma.jax.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31){ data-model="recurrentgemma" }、[`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" } 與 [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74){ data-model="recurrentgemma" }。
- 以下函式庫各有自己的文件網站：[核心 JAX](https://jax.readthedocs.io)、[Flax](https://flax.readthedocs.io) 與 [Orbax](https://orbax.readthedocs.io/)。
- 關於 `sentencepiece` 分詞/還原詞元文件，請參閱 [Google 的 `sentencepiece` GitHub 儲存庫](https://github.com/google/sentencepiece)。
- 關於 `kagglehub` 文件，請參閱 [Kaggle 的 `kagglehub` GitHub 儲存庫](https://github.com/Kaggle/kagglehub) 中的 `README.md`。
- 了解如何[在 Google Cloud Vertex AI 上使用 Gemma 模型][vertex-use-gemma]。
- 閱讀 Google DeepMind 的論文 [RecurrentGemma: Moving Past Transformers
  for Efficient Open Language Models](https://arxiv.org/pdf/2404.07839)。
- 閱讀 Google DeepMind 的論文 [Griffin: Mixing Gated Linear Recurrences with
  Local Attention for Efficient Language Models](https://arxiv.org/pdf/2402.19427)，深入了解 RecurrentGemma 所使用的模型架構。
