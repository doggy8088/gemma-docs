---
title: 使用 JAX 與 Flax 微調 PaliGemma
description: 本筆記本教學說明如何使用 JAX 與 Flax，在視覺語言任務上微調 PaliGemma 模型，包含完整的訓練流程。
models: [paligemma]
frameworks: [jax]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/paligemma-fine-tuning.ipynb
source_url: "https://ai.google.dev/gemma/docs/paligemma/fine-tuning-paligemma.md.txt"
---
# 使用 JAX 與 Flax 微調 PaliGemma


本筆記本說明如何使用 [JAX](https://jax.readthedocs.io/en/latest/index.html) 在視覺語言任務上微調 [PaliGemma](index.md)。*微調*（fine-tuning）是一種可以提升模型在特定任務上的效能的流程，也可以在指示不足時，幫助模型符合特定的輸出需求（前提是您有一組能展示期望輸出的範例）。像 PaliGemma 這類基於 Gemma 的模型需要經過微調，才能產生符合預期的結果。

### 本筆記本包含哪些內容

本筆記本使用來自 [`big_vision`](https://github.com/google-research/big_vision) 的模型參考實作，並說明如何：

- 安裝相依套件，並下載 PaliGemma 模型檢查點與訓練資料
- 將模型載入 GPU 裝置
- 為訓練與推論準備模型的輸入
- 微調模型
- 檢視輸出

本筆記本的訓練資料由 90 組圖片及其長篇描述字幕組成。為了讓它能在 T4 Colab 執行環境上運行，您只會微調語言模型的注意力層，並凍結其他參數。

此範例僅供學習用途。在實際使用案例中，資料量、可訓練參數、訓練步數與超參數，以及得到的結果都可能有顯著差異。

### 開始之前

在開始本筆記本之前，您應該熟悉 Python 程式碼，以及大型語言模型（LLM）的訓練方式。您不需要熟悉 JAX，但在閱讀範例程式碼時，具備 JAX（或 Keras 等類似技術）的基礎知識會很有幫助。

## 設定

以下各節說明讓筆記本能使用 PaliGemma 模型的前置步驟，包括取得模型存取權、取得 API 金鑰，以及設定筆記本執行環境。

### 取得 PaliGemma 存取權

首次使用 PaliGemma 之前，您必須透過 Kaggle 完成以下步驟來申請模型存取權：

1. 登入 [Kaggle](https://www.kaggle.com)，若尚未擁有帳號，請建立新的 Kaggle 帳號。
2. 前往 [PaliGemma 模型卡](https://www.kaggle.com/models/google/paligemma-2){ data-model="paligemma" }，然後點選 **Request Access**。
3. 填寫同意表單並接受條款與條件。

### 設定您的 API 金鑰

要使用 PaliGemma，您必須提供 Kaggle 使用者名稱與 Kaggle API 金鑰。

要產生 Kaggle API 金鑰，請開啟您的 [**Kaggle 設定頁面**](https://www.kaggle.com/settings)，然後點選 **Create New Token**。這會觸發下載包含您 API 憑證的 `kaggle.json` 檔案。

接著在 Colab 中，選取左側窗格的 **Secrets**（🔑），新增您的 Kaggle 使用者名稱與 Kaggle API 金鑰。將使用者名稱存放在 `KAGGLE_USERNAME` 名稱下，將 API 金鑰存放在 `KAGGLE_KEY` 名稱下。

### 選擇執行環境

要完成本教學課程，您需要具備足夠資源來執行 PaliGemma 模型的 Colab 執行環境。在這個案例中，您可以使用 T4 GPU：

1. 在 Colab 視窗右上角，點選 **▾ (Additional connection options)** 下拉選單。
2. 選取 **Change runtime type**。
3. 在 **Hardware accelerator** 下選取 **T4 GPU**。

### 安裝 Python 套件

執行以下儲存格以安裝 KaggleHub。

```bash { .notebook-cell }
pip install -U -q kagglehub
```

### 設定環境變數

設定環境變數並登入 Kaggle。

```python { .notebook-cell }
import os
# The T4 runtime is tight on memory to finetune this model. Preallocate
# all memory ahead of time to avoid out-of-memory due to fragmentation.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

# This will prompt you to enter your Kaggle API token
import kagglehub
kagglehub.login()
```

### 擷取 `big_vision` 存放庫並安裝相關相依套件

執行以下程式碼，從 GitHub 將 `big_vision` 存放庫下載到您的 Colab 筆記本，並安裝與 `big_vision` 相關的相依套件。

```python { .notebook-cell }
import os
import sys

# TPUs with
if "COLAB_TPU_ADDR" in os.environ:
  raise "It seems you are using Colab with remote TPUs which is not supported."

# Fetch big_vision repository if python doesn't know about it and install
# dependencies needed for this notebook.
if not os.path.exists("big_vision_repo"):
  !git clone --quiet --branch=main --depth=1 \
     https://github.com/google-research/big_vision big_vision_repo

# Append big_vision code to python import path
if "big_vision_repo" not in sys.path:
  sys.path.append("big_vision_repo")

# Install missing dependencies. Assume jax~=0.4.25 with GPU available.
!pip3 install -q "overrides" "ml_collections" "einops~=0.7" "sentencepiece"
```

### 匯入 JAX 與其他相依套件

匯入 PaliGemma 所需的 JAX 與其他相依套件，例如 TensorFlow 與 NumPy。

```python { .notebook-cell }
import base64
import functools
import html
import io
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections

import tensorflow as tf
import sentencepiece

from IPython.core.display import display, HTML
from PIL import Image

# Import model definition from big_vision
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

# Import big vision utilities
import big_vision.datasets.jsonl
import big_vision.utils
import big_vision.sharding

# Don't let TF use the GPU or TPUs
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

backend = jax.extend.backend.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")
```

``` { .no-copy .output }
JAX version:  0.7.2
JAX platform: gpu
JAX devices:  1
```

## 下載並設定模型

在此步驟中，您將下載模型檢查點並進行設定，以便稍後進行微調。這個步驟也會說明如何將模型參數移入 TPU 記憶體，這對於在資源有限的裝置上微調模型很有用。

### 下載模型檢查點

PaliGemma 包含數種模型變體。本教學課程使用的是基礎的 [JAX/FLAX PaliGemma 3B 權重模型](https://www.kaggle.com/models/google/paligemma-2/jax/paligemma2-3b-pt-224){ data-model="paligemma" }。

執行以下程式碼從 Kaggle 下載模型檢查點。這個過程需要幾分鐘才能完成。

```python { .notebook-cell }
import os
import kagglehub

# Use these for PaliGemma-2 3B 224px²
LLM_VARIANT = "gemma2_2b"
MODEL_PATH = "./paligemma2-3b-pt-224.b16.npz"
KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"  # Path to fetch from Kaggle.

# Use these for PaliGemma 1:
# LLM_VARIANT = "gemma_2b"
# MODEL_PATH = "./paligemma-3b-pt-224.f16.npz"
# KAGGLE_HANDLE = "google/paligemma/jax/paligemma-3b-pt-224"

if not os.path.exists(MODEL_PATH):
  print("Downloading the checkpoint from Kaggle, this could take a few minutes....")
  MODEL_PATH = kagglehub.model_download(KAGGLE_HANDLE, MODEL_PATH)
  print(f"Model path: {MODEL_PATH}")

TOKENIZER_PATH = "./paligemma_tokenizer.model"
if not os.path.exists(TOKENIZER_PATH):
  print("Downloading the model tokenizer...")
  !gcloud storage cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}
  print(f"Tokenizer path: {TOKENIZER_PATH}")

DATA_DIR="./longcap100"
if not os.path.exists(DATA_DIR):
  print("Downloading the dataset...")
  !gcloud storage cp --no-clobber --recursive gs://longcap100/ .
  print(f"Data path: {DATA_DIR}")
```

``` { .no-copy .output }
Downloading the checkpoint from Kaggle, this could take a few minutes....
Downloading to /root/.cache/kagglehub/models/google/paligemma-2/jax/paligemma2-3b-pt-224/1/./paligemma2-3b-pt-224.b16.npz...
100%|██████████| 5.65G/5.65G [00:54<00:00, 112MB/s]
Model path: /root/.cache/kagglehub/models/google/paligemma-2/jax/paligemma2-3b-pt-224/1/./paligemma2-3b-pt-224.b16.npz
Downloading the model tokenizer...
Copying gs://big_vision/paligemma_tokenizer.model...

- [1 files][  4.1 MiB/  4.1 MiB]                                                
Operation completed over 1 objects/4.1 MiB.                                      
Tokenizer path: ./paligemma_tokenizer.model
Downloading the dataset...
Data path: ./longcap100
```

### 設定模型

現在開始真正設定您要使用的模型。

在本筆記本中，您需要讓模型能裝進 T4 GPU。受限於記憶體空間這類有限資源，設定模型時必須格外留意。

如果微調所有參數，模型將無法在筆記本環境中執行。因此，在本筆記本的這一部分，您將設定模型使其能凍結部分參數，只微調真正需要微調、才能讓模型給出準確結果的參數。在 LLM 中，當參數不再主動用於訓練模型時，就稱為被*凍結*（frozen）。

要設定模型，您需要：

- 將 `model_config` 初始化為 [`FrozenConfigDict`](https://github.com/google/ml_collections/tree/master#frozenconfigdict)，以便凍結部分參數並降低記憶體用量
- 使用 `model_config` 作為設定，初始化 PaliGemma `Model` 類別的實例
- 將模型參數載入 RAM
- 定義 `decode` 函式，從模型取樣輸出

這個儲存格的程式碼執行約需一分鐘。

```python { .notebook-cell }
# Define model

# IMPORTANT: Gemma-2 has a "final_logits_softcap" property. Set it to 0.0
# for better transfer results.
model_config = ml_collections.FrozenConfigDict({
    "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
    "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
})
model = paligemma.Model(**model_config)
tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

# Load params - this can take up to 1 minute in T4 colabs.
params = paligemma.load(None, MODEL_PATH, model_config)

# Define `decode` function to sample outputs from the model.
decode_fn = predict_fns.get_all(model)['decode']
decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())
```

### 將模型參數移入 GPU/TPU 記憶體

現在您需要將模型參數移入 GPU/TPU 記憶體。首先，將參數分片（shard）到可用的 GPU 上，然後載入參數。在這裡，您將依序載入參數。這個過程比同時載入更耗時，但同時載入需要的 RAM 超過了本筆記本可用的量。

最後，列印出所有參數，查看每個參數被轉換為哪種型別。凍結的參數維持為 `float16`，而可訓練的參數會轉換為 `float32`。檢視清單時，您會看到大部分參數已被凍結且為 `float16`。

```python { .notebook-cell }
# Create a pytree mask of the trainable params.
def is_trainable_param(name, param):  # pylint: disable=unused-argument
  if name.startswith("llm/layers/attn/"):  return True
  if name.startswith("llm/"):              return False
  if name.startswith("img/"):              return False
  raise ValueError(f"Unexpected param name {name}")
trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

# If more than one device is available (e.g. multiple GPUs) the parameters can
# be sharded across them to reduce HBM usage per device.
mesh = jax.sharding.Mesh(jax.devices(), ("data"))

data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("data"))

params_sharding = big_vision.sharding.infer_sharding(
    params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)

# Yes: Some donated buffers are not usable.
warnings.filterwarnings(
    "ignore", message="Some donated buffers were not usable")

@functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def maybe_cast_to_f32(params, trainable):
  # Cast others to float16, since some GPUs don't support bf16.
  return jax.tree.map(lambda p, m: p.astype(jnp.float32)
                      if m else p.astype(jnp.float16),
                      params, trainable)

# Loading all params in simultaneous - albeit much faster and more succinct -
# requires more RAM than the T4 colab runtimes have by default.
# Instead, do it param by param.
params, treedef = jax.tree.flatten(params)
sharding_leaves = jax.tree.leaves(params_sharding)
trainable_leaves = jax.tree.leaves(trainable_mask)
for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
  params[idx] = big_vision.utils.reshard(params[idx], sharding)
  params[idx] = maybe_cast_to_f32(params[idx], trainable)
  params[idx].block_until_ready()
params = jax.tree.unflatten(treedef, params)

# Print params to show what the model is made of.
def parameter_overview(params):
  for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
    print(f"{path:80s} {str(arr.shape):22s} {arr.dtype}")

print(" == Model params == ")
parameter_overview(params)
```

``` { .no-copy .output }
== Model params == 
img/Transformer/encoder_norm/bias                                                (1152,)                float16
img/Transformer/encoder_norm/scale                                               (1152,)                float16
img/Transformer/encoderblock/LayerNorm_0/bias                                    (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_0/scale                                   (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_1/bias                                    (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_1/scale                                   (27, 1152)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias                             (27, 4304)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel                           (27, 1152, 4304)       float16
img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias                             (27, 1152)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel                           (27, 4304, 1152)       float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias             (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel           (27, 1152, 16, 72)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias             (27, 1152)             float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel           (27, 16, 72, 1152)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias           (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel        (27, 1152, 16, 72)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias           (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel         (27, 1152, 16, 72)     float16
img/embedding/bias                                                               (1152,)                float16
img/embedding/kernel                                                             (14, 14, 3, 1152)      float16
img/head/bias                                                                    (2304,)                float16
img/head/kernel                                                                  (1152, 2304)           float16
img/pos_embedding                                                                (1, 256, 1152)         float16
llm/embedder/input_embedding                                                     (257152, 2304)         float16
llm/final_norm/scale                                                             (2304,)                float16
llm/layers/attn/attn_vec_einsum/w                                                (26, 8, 256, 2304)     float32
llm/layers/attn/kv_einsum/w                                                      (26, 2, 4, 2304, 256)  float32
llm/layers/attn/q_einsum/w                                                       (26, 8, 2304, 256)     float32
llm/layers/mlp/gating_einsum                                                     (26, 2, 2304, 9216)    float16
llm/layers/mlp/linear                                                            (26, 9216, 2304)       float16
llm/layers/post_attention_norm/scale                                             (26, 2304)             float16
llm/layers/post_ffw_norm/scale                                                   (26, 2304)             float16
llm/layers/pre_attention_norm/scale                                              (26, 2304)             float16
llm/layers/pre_ffw_norm/scale                                                    (26, 2304)             float16
```

## 準備微調模型

模型設定完成後，就可以開始微調了。在此步驟中，您將建立模型的輸入以及訓練與驗證迭代器、查看訓練範例，並定義訓練與驗證迴圈。

### 建立模型輸入

您使用的模型檢查點已經在各種長寬比、已調整為 224x224 像素的圖片上訓練過，並且能處理經過詞元化的文字。

以下程式碼定義了三個函式，您將在下一步使用它們來建立模型的輸入：

- **`preprocess_image`：** 正規化圖片資料。在此案例中，前處理會將傳入的圖片轉為灰階、移除 Alpha 圖層，並將圖片調整為模型圖片輸入所需的尺寸（224x224 像素）。
- **`preprocess_tokens`：** 拆分詞元，並加入旗標來標記詞元是前綴（prefix）還是後綴（suffix）詞元。這些旗標稍後會在程式碼中、訓練步驟與評估迴圈期間使用。
- **`postprocess_tokens`：** 移除序列結束（EOS）詞元處及之後殘留的任何詞元，並回傳剩餘的解碼詞元。

```python { .notebook-cell }
def preprocess_image(image, size=224):
  # Model has been trained to handle images of different aspects ratios
  # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
  # options are helpful to improve quality in some tasks.
  image = np.asarray(image)
  if image.ndim == 2:  # Convert image without last channel into greyscale.
    image = np.stack((image,)*3, axis=-1)
  image = image[..., :3]  # Remove alpha layer.
  assert image.shape[-1] == 3

  image = tf.constant(image)
  image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
  return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

def preprocess_tokens(prefix, suffix=None, seqlen=None):
  # Model has been trained to handle tokenized text composed of a prefix with
  # full attention and a suffix with causal attention.
  separator = "\n"
  tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
  mask_ar = [0] * len(tokens)    # 0 to use full attention for prefix.
  mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

  if suffix:
    suffix = tokenizer.encode(suffix, add_eos=True)
    tokens += suffix
    mask_ar += [1] * len(suffix)    # 1 to use causal attention for suffix.
    mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.

  mask_input = [1] * len(tokens)    # 1 if it's a token, 0 if padding.
  if seqlen:
    padding = [0] * max(0, seqlen - len(tokens))
    tokens = tokens[:seqlen] + padding
    mask_ar = mask_ar[:seqlen] + padding
    mask_loss = mask_loss[:seqlen] + padding
    mask_input = mask_input[:seqlen] + padding

  return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

def postprocess_tokens(tokens):
  tokens = tokens.tolist()  # np.array to list[int]
  try:  # Remove tokens at and after EOS if any.
    eos_pos = tokens.index(tokenizer.eos_id())
    tokens = tokens[:eos_pos]
  except ValueError:
    pass
  return tokenizer.decode(tokens)
```

### 建立訓練與驗證迭代器

建立兩個迭代器：

- **訓練迭代器**，讓訓練過程能分批處理資料，而非一次處理全部資料
    - 這讓您可以在使用前進行一些資料前處理
- **驗證迭代器**，讓訓練過程能迭代驗證資料集，以查看微調後的模型與所提供結果的吻合程度

```python { .notebook-cell }
SEQLEN = 128

train_dataset = big_vision.datasets.jsonl.DataSource(
    os.path.join(DATA_DIR, "data_train90.jsonl"),
    fopen_keys={"image": DATA_DIR})

val_dataset = big_vision.datasets.jsonl.DataSource(
    os.path.join(DATA_DIR, "data_val10.jsonl"),
    fopen_keys={"image": DATA_DIR})


def train_data_iterator():
  """Never ending iterator over training examples."""
  # Shuffle examples and repeat so one can train for many epochs.
  dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
  for example in dataset.as_numpy_iterator():
    image = Image.open(io.BytesIO(example["image"]))
    image = preprocess_image(image)

    prefix = "caption en"  # Could also be a different prefix per example.
    suffix = example["suffix"].decode().lower()
    tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, suffix, SEQLEN)

    yield {
        "image": np.asarray(image),
        "text": np.asarray(tokens),
        "mask_ar": np.asarray(mask_ar),
        "mask_loss": np.asarray(mask_loss),
    }


def validation_data_iterator():
  """Single iterator over validation examples."""
  for example in val_dataset.get_tfdata(ordered=True).as_numpy_iterator():
    image = Image.open(io.BytesIO(example["image"]))
    image = preprocess_image(image)

    prefix = "caption en"  # Could also be a different prefix per example.
    tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)

    yield {
        "image": np.asarray(image),
        "text": np.asarray(tokens),
        "mask_ar": np.asarray(mask_ar),
        "mask_input": np.asarray(mask_input),
    }
```

### 查看訓練範例

在本筆記本中，訓練資料包含 90 張圖片，每張都搭配描述圖片內容的長篇描述。

!!! Note
    用於實際案例的一般訓練資料集應包含更多圖片，但本筆記本限制了資料點數量，以便您能在合理的時間內訓練模型、作為範例。


以下程式碼會從訓練資料集中隨機選取圖片並列印其描述，讓您可以看到用來訓練模型的圖片與描述是什麼樣子。每張圖片會顯示為 128x128 像素的 JPEG，描述則列印在圖片右側。

```python { .notebook-cell }
def render_inline(image, resize=(128, 128)):
  """Convert image into inline html."""
  image = Image.fromarray(image)
  image.resize(resize)
  with io.BytesIO() as buffer:
    image.save(buffer, format='jpeg')
    image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
    return f"data:image/jpeg;base64,{image_b64}"

def render_example(image, caption):
  image = ((image + 1)/2 * 255).astype(np.uint8)  # [-1,1] -> [0, 255]
  return f"""
    <div style="display: inline-flex; align-items: center; justify-content: center;">
        <img style="width:128px; height:128px;" src="{render_inline(image, resize=(64,64))}" />
        <p style="width:256px; margin:10px; font-size:small;">{html.escape(caption)}</p>
    </div>
    """

html_out = ""
for idx, example in zip(range(8), train_data_iterator()):
  caption = postprocess_tokens(example["text"])  # detokenize model input.
  caption = caption[len("caption en\n"):]        # strip prefix
  html_out += render_example(example["image"], caption)

print("Training examples")
display(HTML(html_out))
```

### 定義訓練與評估迴圈

定義訓練迴圈來在所提供的資料集上訓練模型，並定義評估迴圈來查看驗證資料集中的所有範例並進行預測。

#### 定義訓練迴圈

`update_fn` 函式定義了訓練步驟。在訓練步驟期間，會計算每個範例的損失，並對可訓練參數套用隨機梯度下降（SGD）。

回想一下，本筆記本稍早在 `preprocess_tokens` 函式中加入了包含 `mask_loss` 的旗標。在這裡，您將使用 `mask_loss` 旗標將前綴與填充詞元排除在損失之外。若沒有它，損失計算會被扭曲。您還需要對每個範例進行正規化，因為每個範例的詞元數量不同。排除前綴與填充詞元並正規化範例之後，就可以計算每個範例的損失。

訓練步驟還包含一個套用 SGD 以優化訓練的函式。

#### 定義評估迴圈

`make_predictions` 函式就是您的評估迴圈。評估迴圈相當直觀，只有一處值得注意的變化。回想一下，本筆記本開頭提到訓練資料集只有 90 個範例。這是非常少的訓練範例數，導致執行訓練時，模型的範例數不足以填滿批次大小。這表示在評估迴圈中，您需要透過重複範例來填充批次。

為確保評估迴圈只計算實際範例而不計入填充範例，您必須對填充範例套用遮罩，將它們排除在輸出之外。

```python { .notebook-cell }
# The main update_fn using a simple stochastic gradient descent (SGD).
@functools.partial(jax.jit, donate_argnums=(0,))
def update_fn(params, batch, learning_rate):
  imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

  def loss_fn(params):
    text_logits, _ = model.apply({"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True)
    logp = jax.nn.log_softmax(text_logits, axis=-1)

    # The model takes as input txts[:, :-1] but the loss is defined as predicting
    # next tokens txts[:, 1:]. Additionally, mask_loss[:, 1:] indicates which tokens
    # are part of the loss (e.g. prefix and padded tokens are not included).
    mask_loss = batch["mask_loss"][:, 1:]
    targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

    # Compute the loss per example. i.e. the mean of per token pplx.
    # Since each example has a different number of tokens, normalize it.
    token_pplx = jnp.sum(logp * targets, axis=-1)  # sum across vocab_size.
    example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)  # sum across seq_len.
    example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)  # weight by num of tokens.

    # batch_loss: mean of per example loss.
    return jnp.mean(example_loss)

  loss, grads = jax.value_and_grad(loss_fn)(params)

  # Apply gradients to trainable params using SGD.
  def apply_grad(param, gradient, trainable):
    if not trainable: return param
    return param - learning_rate * gradient

  params = jax.tree_util.tree_map(apply_grad, params, grads, trainable_mask)

  return params, loss

# Evaluation/inference loop.
def make_predictions(data_iterator, *, num_examples=None,
                     batch_size=4, seqlen=SEQLEN, sampler="greedy"):
  outputs = []
  while True:
    # Construct a list of examples in the batch.
    examples = []
    try:
      for _ in range(batch_size):
        examples.append(next(data_iterator))
        examples[-1]["_mask"] = np.array(True)  # Indicates true example.
    except StopIteration:
      if len(examples) == 0:
        return outputs

    # Not enough examples to complete a batch. Pad by repeating last example.
    while len(examples) % batch_size:
      examples.append(dict(examples[-1]))
      examples[-1]["_mask"] = np.array(False)  # Indicates padding example.

    # Convert list of examples into a dict of np.arrays and load onto devices.
    batch = jax.tree.map(lambda *x: np.stack(x), *examples)
    batch = big_vision.utils.reshard(batch, data_sharding)

    # Make model predictions
    tokens = decode({"params": params}, batch=batch,
                    max_decode_len=seqlen, sampler=sampler)

    # Fetch model predictions to device and detokenize.
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t) for t in tokens]

    # Append to html output.
    for example, response in zip(examples, responses):
      outputs.append((example["image"], response))
      if num_examples and len(outputs) >= num_examples:
        return outputs
```

## 微調模型

一切設定就緒並查看過訓練資料後，終於可以開始微調模型了。以下程式碼會讓模型執行 64 步的訓練迴圈，並列印每個步驟的學習率（列印輸出中的 `lr`）與損失率。

每 16 步，模型會列印該訓練步驟的預測結果。這段程式碼會對同一組圖片列印預測，讓您可以看到模型預測描述的能力隨時間逐步改善。

在訓練的早期步驟，描述可能會有問題，例如模型卡在預測迴圈中而出現重複的句子，或句子未完成。隨著訓練進行，模型的預測會穩步變得更準確。到第 64 步時，模型的預測應該會與訓練資料提供的描述非常接近。

在 T4 TPU 上，這個過程約需 15 分鐘完成。

```python { .notebook-cell }
# Run a short training loop with cosine learning rate schedule.
#
# Note: the first step can be quite slow on some machines (up to several minutes)
# due to XLA compilation of the jax.jit'd function.
#
%%time

BATCH_SIZE = 8
TRAIN_EXAMPLES = 512
LEARNING_RATE = 0.03

TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
EVAL_STEPS = TRAIN_STEPS // 4

train_data_it = train_data_iterator()

sched_fn = big_vision.utils.create_learning_rate_schedule(
    total_steps=TRAIN_STEPS+1, base=LEARNING_RATE,
    decay_type="cosine", warmup_percent=0.10)

for step in range(1, TRAIN_STEPS+1):
  # Make list of N training examples.
  examples = [next(train_data_it) for _ in range(BATCH_SIZE)]

  # Convert list of examples into a dict of np.arrays and load onto devices.
  batch = jax.tree.map(lambda *x: np.stack(x), *examples)
  batch = big_vision.utils.reshard(batch, data_sharding)

  # Training step and report training loss
  learning_rate = sched_fn(step)
  params, loss = update_fn(params, batch, learning_rate)

  loss = jax.device_get(loss)
  print(f"step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}")

  if (step % EVAL_STEPS) == 0:
    print(f"Model predictions at step {step}")
    html_out = ""
    for image, caption in make_predictions(
        validation_data_iterator(), num_examples=4, batch_size=4):
      html_out += render_example(image, caption)
    display(HTML(html_out))
```

## 輸出

本筆記本的驗證資料只有 10 張圖片。在一般的程式碼中，驗證的資料點可能會多得多，但在本筆記本中，請執行以下程式碼為全部 10 張圖片產生描述。微調模型之後，這些描述在形式與內容涵蓋面上，應該會與您稍早在本筆記本中查看的訓練資料描述非常相似。

執行以下程式碼，為驗證資料集產生描述。

```python { .notebook-cell }
# The validation data consists of 10 images in a different domain than training
# data.
%%time

print("Model predictions")
html_out = ""
for image, caption in make_predictions(validation_data_iterator(), batch_size=4):
  html_out += render_example(image, caption)
display(HTML(html_out))
```
