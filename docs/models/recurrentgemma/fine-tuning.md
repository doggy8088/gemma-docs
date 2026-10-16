---
title: 使用 JAX 與 Flax 進行微調
description: 使用 JAX、Flax 與 MTNT 噪聲文本資料集，將 RecurrentGemma 2B Instruct 微調為英法翻譯模型。
models: [recurrentgemma]
frameworks: [jax]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/recurrentgemma-fine-tuning.ipynb
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_finetune.md.txt"
---
# 使用 JAX 與 Flax 進行微調

本教學示範如何使用 [Google DeepMind 的 `recurrentgemma` 函式庫](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" }、[JAX](https://jax.readthedocs.io)（高效能數值運算函式庫）、[Flax](https://flax.readthedocs.io)（基於 JAX 的類神經網路函式庫）、[Chex](https://chex.readthedocs.io/en/latest/)（用於撰寫可靠 JAX 程式碼的工具函式庫）、[Optax](https://optax.readthedocs.io/en/latest/)（基於 JAX 的梯度處理與最佳化函式庫），以及 [MTNT（噪聲文本機器翻譯，Machine Translation of Noisy Text）資料集](https://arxiv.org/abs/1809.00388)，將 [RecurrentGemma](index.md) 2B Instruct 模型微調為英法翻譯任務。雖然本筆記本未直接使用 Flax，但 Gemma 正是用 Flax 建立的。

`recurrentgemma` 函式庫以 JAX、Flax、[Orbax](https://orbax.readthedocs.io/)（基於 JAX 的訓練工具函式庫，例如檢查點）與 [SentencePiece](https://github.com/google/sentencepiece)（分詞/還原詞元函式庫）撰寫。

這本筆記本可在 Google Colab 上使用 T4 GPU 執行（前往**編輯** \> **筆記本設定** \> 在**硬體加速器**下方選取 **T4 GPU**）。

## 設定

以下各節說明準備筆記本以使用 RecurrentGemma 模型的步驟，包括模型存取、取得 API 金鑰，以及設定筆記本執行環境。

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

免費的 Colab 硬體加速目前*不足以*執行本筆記本。如果您使用 [Colab 隨用隨付或 Colab Pro](https://colab.research.google.com/signup)，請點選**編輯** \> **筆記本設定** \> 選取 **A100 GPU** \> **儲存**以啟用硬體加速。

接著，您需要從 [`github.com/google-deepmind/recurrentgemma`](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" } 安裝 Google DeepMind 的 `recurrentgemma` 函式庫。如果看到關於「pip 依賴解析器」的錯誤，通常可以忽略。

!!! Note
    安裝 `recurrentgemma` 時，也會一併安裝 [`flax`](https://flax.readthedocs.io)、核心 [`jax`](https://jax.readthedocs.io)、[`optax`](https://optax.readthedocs.io/en/latest/)（基於 JAX 的梯度處理與最佳化函式庫）、[`orbax`](https://orbax.readthedocs.io/) 與 [`sentencepiece`](https://github.com/google/sentencepiece)。


```bash { .notebook-cell }
pip install -q git+https://github.com/google-deepmind/recurrentgemma.git
```

### 匯入函式庫

本筆記本使用 [Flax](https://flax.readthedocs.io)（類神經網路）、核心 [JAX](https://jax.readthedocs.io)、[SentencePiece](https://github.com/google/sentencepiece)（分詞）、[Chex](https://chex.readthedocs.io/en/latest/)（用於撰寫可靠 JAX 程式碼的工具函式庫）、[Optax](https://optax.readthedocs.io/en/latest/)（梯度處理與最佳化函式庫）與 TensorFlow Datasets。

```python { .notebook-cell }
import pathlib
from typing import Any, Mapping, Iterator
import enum
import functools

import chex
import jax
import jax.numpy as jnp
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

import sentencepiece as spm

from recurrentgemma import jax as recurrentgemma
```

## 載入 RecurrentGemma 模型

1. 使用 [`kagglehub.model_download`](https://github.com/Kaggle/kagglehub/blob/bddefc718182282882b72f814d407d89e5d178c4/src/kagglehub/models.py#L12) 載入 RecurrentGemma 模型，它接受三個參數：

- `handle`：來自 Kaggle 的模型代碼
- `path`：（選用字串）本機路徑
- `force_download`：（選用布林值）強制重新下載模型

!!! Note
    請注意，RecurrentGemma 2B (IT) 模型大小約為 3.85Gb。


```python { .notebook-cell }
RECURRENTGEMMA_VARIANT = '2b-it' # @param ['2b', '2b-it'] {type:"string"}

import kagglehub

RECURRENTGEMMA_PATH = kagglehub.model_download(f'google/recurrentgemma/flax/{RECURRENTGEMMA_VARIANT}')
```

``` { .no-copy .output }
Downloading from https://www.kaggle.com/api/v1/models/google/recurrentgemma/flax/2b-it/1/download...
100%|██████████| 3.85G/3.85G [00:50<00:00, 81.5MB/s]
Extracting model files...
```

```python { .notebook-cell }
print('RECURRENTGEMMA_VARIANT:', RECURRENTGEMMA_VARIANT)
```

``` { .no-copy .output }
RECURRENTGEMMA_VARIANT: 2b-it
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
## 載入並準備 MTNT 資料集與 Gemma 分詞器

您將使用 [MTNT（噪聲文本機器翻譯）](https://arxiv.org/abs/1809.00388) 資料集，可從 [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mtnt) 取得。

下載 MTNT 資料集中的英翻法部分，然後取兩個範例。資料集中的每個樣本包含兩個欄位：`src`：原始英文句子；`dst`：對應的法文翻譯。

```python { .notebook-cell }
ds = tfds.load("mtnt/en-fr", split="train")

ds = ds.take(2)
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
```

``` { .no-copy .output }
Downloading and preparing dataset 35.08 MiB (download: 35.08 MiB, generated: 11.33 MiB, total: 46.41 MiB) to /root/tensorflow_datasets/mtnt/en-fr/1.0.0...
Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s]
Extraction completed...: 0 file [00:00, ? file/s]
Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]
Generating train examples...:   0%|          | 0/35692 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/mtnt/en-fr/1.0.0.incompleteJLH33K/mtnt-train.tfrecord*...:   0%|          …
Generating test examples...:   0%|          | 0/1020 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/mtnt/en-fr/1.0.0.incompleteJLH33K/mtnt-test.tfrecord*...:   0%|          …
Generating valid examples...:   0%|          | 0/811 [00:00<?, ? examples/s]
Shuffling /root/tensorflow_datasets/mtnt/en-fr/1.0.0.incompleteJLH33K/mtnt-valid.tfrecord*...:   0%|          …
Dataset mtnt downloaded and prepared to /root/tensorflow_datasets/mtnt/en-fr/1.0.0. Subsequent calls will reuse this data.
Example 0:
dst: b'Le groupe de " toutes les \xc3\xa9toiles potentielles de la conf\xc3\xa9rence de l\'Est mais qui ne s\'en sortent pas dans le groupe de l\'Ouest ".'
src: b'The group of \xe2\x80\x9ceastern conference potential all stars but not making it in the West\xe2\x80\x9d group.'

Example 1:
dst: b"Kameron est-elle un peu aigrie de son manque de temps \xc3\xa0 l'\xc3\xa9cran ?"
src: b'Is Kameron a Little Salty About Her Lack of Air Time?'
```

使用 [`sentencepiece.SentencePieceProcessor`](https://github.com/google/sentencepiece/blob/4d6a1f41069c4636c51a5590f7578a0dbed83450/python/src/sentencepiece/__init__.py#L423) 建構的 Gemma 分詞器載入：

```python { .notebook-cell }
vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)
```

``` { .no-copy .output }
True
```

為英法翻譯任務客製化 [`SentencePieceProcessor`](https://github.com/google/sentencepiece/blob/4d6a1f41069c4636c51a5590f7578a0dbed83450/python/src/sentencepiece/__init__.py#L423)。由於您將微調 RecurrentGemma（Griffin）模型的英文部分，需要做一些調整，例如：

- *輸入前綴*：在每個輸入前加上共同的前綴，以標示翻譯任務。例如，您可以使用帶有前綴的提示，例如 `Translate this into French: [INPUT_SENTENCE]`。

- *翻譯起始後綴*：在每個提示末尾加上後綴，指示 Gemma 模型何時開始翻譯。一個換行字元即可。

- *語言模型詞元*：RecurrentGemma（Griffin）模型預期每個序列開頭有一個「序列開始」詞元。同樣地，您需要在每個訓練範例的結尾加上一個「序列結束」詞元。

以下列方式為 `SentencePieceProcessor` 建立自訂包裝器：

```python { .notebook-cell }
class GriffinTokenizer:
  """A custom wrapper around a SentencePieceProcessor."""

  def __init__(self, spm_processor: spm.SentencePieceProcessor):
    self._spm_processor = spm_processor

  @property
  def pad_id(self) -> int:
    """Fast access to the pad ID."""
    return self._spm_processor.pad_id()

  def tokenize(
      self,
      example: str | bytes,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> jax.Array:
    """
    A tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an end of sentence token at the end of the output
               sequence.
    Returns:
      Tokens corresponding to the input string.
    """
    int_list = [self._spm_processor.bos_id()]
    int_list.extend(self._spm_processor.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self._spm_processor.eos_id())

    return jnp.array(int_list, dtype=jnp.int32)

  def tokenize_tf_op(
      self,
      str_tensor: tf.Tensor,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> tf.Tensor:
    """A TensforFlow operator for the `tokenize` function."""
    encoded = tf.numpy_function(
        self.tokenize,
        [str_tensor, prefix, suffix, add_eos],
        tf.int32)
    encoded.set_shape([None])
    return encoded

  def to_string(self, tokens: jax.Array) -> str:
    """Convert an array of tokens to a string."""
    return self._spm_processor.EncodeIds(tokens.tolist())
```

具現化您新建立的自訂 `GriffinTokenizer`，並將它套用在 MTNT 資料集的一小部分樣本上試試：

```python { .notebook-cell }
def tokenize_source(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(
      example,
      prefix='Translate this into French:\n',
      suffix='\n',
      add_eos=False
  )
def tokenize_destination(tokenizer, example: tf.Tensor):
  return tokenizer.tokenize_tf_op(example, add_eos=True)

tokenizer = GriffinTokenizer(vocab)

ds = tfds.load("mtnt/en-fr",split="train")
ds = ds.take(2)
ds = ds.map(lambda x: {
    'src': tokenize_source(tokenizer, x['src']),
    'dst': tokenize_destination(tokenizer, x['dst'])
  })
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
```

``` { .no-copy .output }
Example 0:
src: [     2  49688    736   1280   6987 235292    108    651   2778    576
   1080 104745  11982   5736    832   8995    901    780   3547    665
    575    573   4589 235369   2778 235265    108]
dst: [     2   2025  29653    581    664  16298   1437  55563  41435   7840
    581    683 111452    581    533 235303   9776   4108   2459    679
    485 235303    479   6728    579   1806   2499    709  29653    581
    533 235303 101323  16054      1]

Example 1:
src: [     2  49688    736   1280   6987 235292    108   2437  87150    477
    476  11709 230461   8045   3636  40268    576   4252   4897 235336
    108]
dst: [     2 213606    477   1455 235290   3510    748   8268 191017   2809
    581   2032  69972    581  11495   1305    533 235303  65978   1654
      1]
```
為整個 MTNT 資料集建立資料載入器：

```python { .notebook-cell }
@chex.dataclass(frozen=True)
class TrainingInput:
  # Input tokens provided to the model.
  input_tokens: jax.Array

  # A mask that determines which tokens contribute to the target loss
  # calculation.
  target_mask: jax.Array

class DatasetSplit(enum.Enum):
  TRAIN = 'train'
  VALIDATION = 'valid'


class MTNTDatasetBuilder:
  """A data loader for the MTNT dataset."""

  N_ITEMS = {DatasetSplit.TRAIN: 35_692, DatasetSplit.VALIDATION: 811}

  BUFFER_SIZE_SHUFFLE = 10_000
  TRANSLATION_PREFIX = 'Translate this into French:\n'
  TRANSLATION_SUFFIX = '\n'

  def __init__(self,
               tokenizer : GriffinTokenizer,
               max_seq_len: int):
    """A constructor.

    Args:
      tokenizer: The tokenizer to use.
      max_seq_len: The size of each sequence in a given batch.
    """
    self._tokenizer = tokenizer
    self._base_data = {
        DatasetSplit.TRAIN: tfds.load("mtnt/en-fr",split="train"),
        DatasetSplit.VALIDATION: tfds.load("mtnt/en-fr",split="valid"),
    }
    self._max_seq_len = max_seq_len

  def _tokenize_source(self, example: tf.Tensor):
    """A tokenization function for the source."""
    return self._tokenizer.tokenize_tf_op(
        example, prefix=self.TRANSLATION_PREFIX, suffix=self.TRANSLATION_SUFFIX,
        add_eos=False
    )

  def _tokenize_destination(self, example: tf.Tensor):
    """A tokenization function for the French translation."""
    return self._tokenizer.tokenize_tf_op(example, add_eos=True)

  def _pad_up_to_max_len(self,
                         input_tensor: tf.Tensor,
                         pad_value: int | bool,
                         ) -> tf.Tensor:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = tf.shape(input_tensor)[0]
    to_pad = tf.maximum(self._max_seq_len - seq_len, 0)
    return tf.pad(
        input_tensor, [[0, to_pad]], mode='CONSTANT', constant_values=pad_value,
    )

  def _to_training_input(
      self,
      src_tokens: jax.Array,
      dst_tokens: jax.Array,
  ) -> TrainingInput:
    """Build a training input from a tuple of source and destination tokens."""

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = tf.concat([src_tokens, dst_tokens], axis=0)

    # You want to prevent the model from updating based on the source (input)
    # tokens. To achieve this, add a target mask to each input.
    q_mask = tf.zeros_like(src_tokens, dtype=tf.bool)
    a_mask = tf.ones_like(dst_tokens, dtype=tf.bool)
    mask = tf.concat([q_mask, a_mask], axis=0)

    # If the output tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)

    # You don't want to perform the backward on the pad tokens.
    mask = self._pad_up_to_max_len(mask, False)

    return TrainingInput(input_tokens=tokens, target_mask=mask)


  def get_train_dataset(self, batch_size: int, num_epochs: int):
    """Build the training dataset."""

    # Tokenize each sample.
    ds = self._base_data[DatasetSplit.TRAIN].map(
        lambda x : (self._tokenize_source(x['src']),
                    self._tokenize_destination(x['dst']))
    )

    # Convert them to training inputs.
    ds = ds.map(lambda x, y: self._to_training_input(x, y))

    # Remove the samples which are too long.
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)

    # Shuffle the dataset.
    ds = ds.shuffle(buffer_size=self.BUFFER_SIZE_SHUFFLE)

    # Repeat if necessary.
    ds = ds.repeat(num_epochs)

    # Build batches.
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def get_validation_dataset(self, batch_size: int):
    """Build the validation dataset."""

    # Same as the training dataset, but no shuffling and no repetition
    ds = self._base_data[DatasetSplit.VALIDATION].map(
        lambda x : (self._tokenize_source(x['src']),
                    self._tokenize_destination(x['dst']))
    )
    ds = ds.map(lambda x, y: self._to_training_input(x, y))
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
```

再次具現化自訂 `GriffinTokenizer` 並將它套用在 MTNT 資料集上，試用 `MTNTDatasetBuilder` 並取兩個範例：

```python { .notebook-cell }
dataset_builder = MTNTDatasetBuilder(tokenizer, max_seq_len=20)
ds = dataset_builder.get_train_dataset(3, 1)
ds = ds.take(2)
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()
```

``` { .no-copy .output }
WARNING:tensorflow:Mapping types may not work well with tf.nest. Prefer using MutableMapping for <class '__main__.TrainingInput'>
WARNING:tensorflow:Mapping types may not work well with tf.nest. Prefer using MutableMapping for <class '__main__.TrainingInput'>
WARNING:tensorflow:Mapping types may not work well with tf.nest. Prefer using MutableMapping for <class '__main__.TrainingInput'>
Example 0:
input_tokens: [[     2  49688    736   1280   6987 235292    108  12583    665 235265
     108      2   6151  94975   1320   6238 235265      1      0      0]
 [     2  49688    736   1280   6987 235292    108   4899  29960  11270
  108282 235265    108      2   4899  79025  11270 108282      1      0]
 [     2  49688    736   1280   6987 235292    108  26620 235265    108
       2  26620 235265      1      0      0      0      0      0      0]]
target_mask: [[False False False False False False False False False False False  True
   True  True  True  True  True  True False False]
 [False False False False False False False False False False False False
  False  True  True  True  True  True  True False]
 [False False False False False False False False False False  True  True
   True  True False False False False False False]]

Example 1:
input_tokens: [[     2  49688    736   1280   6987 235292    108    527   5174   1683
  235336    108      2 206790    581  20726    482   2208   1654      1]
 [     2  49688    736   1280   6987 235292    108  28484 235256 235336
     108      2 120500  13832   1654      1      0      0      0      0]
 [     2  49688    736   1280   6987 235292    108 235324 235304   2705
  235265    108      2 235324 235304  19963 235265      1      0      0]]
target_mask: [[False False False False False False False False False False False False
   True  True  True  True  True  True  True  True]
 [False False False False False False False False False False False False
  False  True  True  True  True  True False False False]
 [False False False False False False False False False False False False
   True  True  True  True  True  True False False]]
```
## 設定模型

在開始微調 Gemma 模型之前，您需要先設定它。

使用 [`recurrentgemma.jax.utils.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31){ data-model="recurrentgemma" } 方法載入 RecurrentGemma（Griffin）模型檢查點：

```python { .notebook-cell }
params =  recurrentgemma.load_parameters(CKPT_PATH, "single_device")
```

若要從 RecurrentGemma 模型檢查點自動載入正確的設定，請使用 [`recurrentgemma.GriffinConfig.from_flax_params_or_variables`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/common.py#L128){ data-model="recurrentgemma" }：

```python { .notebook-cell }
config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(params)
```

使用 [`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" } 具現化 [Griffin](https://arxiv.org/abs/2402.19427) 模型：

```python { .notebook-cell }
model = recurrentgemma.Griffin(config)
```

在 RecurrentGemma 模型檢查點/權重與分詞器之上，使用 [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74){ data-model="recurrentgemma" } 建立 `sampler`，檢查您的模型是否能執行翻譯：

```python { .notebook-cell }
sampler = recurrentgemma.Sampler(model=model, vocab=vocab, params=params)
```

## 微調模型

在本節中，您將：

- 使用 `gemma.transformer.Transformer` 類別建立前向傳遞與損失函數。
- 為詞元建立位置與注意力遮罩向量。
- 使用 Flax 建立訓練步階函數。
- 建立不含反向傳遞的驗證步階。
- 建立訓練迴圈。
- 微調 Gemma 模型。

使用 [`recurrentgemma.jax.griffin.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" }
類別定義前向傳遞與損失函數。RecurrentGemma 的 `Griffin` 繼承自 [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)，並提供兩個重要方法：

- `init`：初始化模型的參數。
- `apply`：使用給定的參數集執行模型的 `__call__` 函數。

由於您使用的是預訓練的 Gemma 權重，因此不需要使用 `init` 函數。

```python { .notebook-cell }
def forward_and_loss_fn(
    params,
    *,
    model: recurrentgemma.Griffin,
    input_tokens: jax.Array,            # Shape [B, L]
    input_mask: jax.Array,              # Shape [B, L]
    positions: jax.Array,               # Shape [B, L]
) -> jax.Array:
  """Forward pass and loss function.

  Args:
    params: model's input parameters.
    model: Griffin model to call.
    input_tokens: input tokens sequence, shape [B, L].
    input_mask: tokens to ignore when computing the loss, shape [B, L].
    positions: relative position of each token, shape [B, L].

  Returns:
    Softmax cross-entropy loss for the next-token prediction task.
  """
  batch_size = input_tokens.shape[0]
  # Forward pass on the input data.
  # No attention cache is needed here.
  # Exclude the last step as it does not appear in the targets.
  logits, _ = model.apply(
        {"params": params},
        tokens=input_tokens[:, :-1],
        segment_pos=positions[:, :-1],
        cache=None,
    )

  # Similarly, the first token cannot be predicteds.
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels into one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[...,None]

  # Normalization factor.
  norm_factor = batch_size * (jnp.sum(target_mask) + 1e-8)

  # Return the negative log-likelihood loss (NLL) function.
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) / norm_factor
```

建立執行反向傳遞並相應更新模型參數的 `train_step` 函數，其中：

- [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html) 用於在前向與反向傳遞期間評估損失函數與梯度。
- [`optax.apply_updates`](https://optax.readthedocs.io/en/latest/api/apply_updates.html#optax.apply_updates) 用於更新參數。

```python { .notebook-cell }
Params = Mapping[str, Any]

def get_positions(example: jax.Array, pad_id : int) -> jax.Array:
  """Builds the position vector from the given tokens."""
  pad_mask = example != pad_id
  positions = jnp.cumsum(pad_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  positions = positions - (positions >= 1)
  return positions

@functools.partial(
    jax.jit,
    static_argnames=['model', 'optimizer'],
    donate_argnames=['params', 'opt_state'],
)
def train_step(
    model: recurrentgemma.Griffin,
    params: Params,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    pad_id: int,
    example: TrainingInput,
) -> tuple[jax.Array, Params, optax.OptState]:
  """The train step.

  Args:
    model: The RecurrentGemma (Griffin) model.
    params: The model's input parameters.
    optimizer: The Optax optimizer to use.
    opt_state: The input optimizer's state.
    pad_id: The ID of the pad token.
    example: The input batch.

  Returns:
    Training loss, updated parameters, updated optimizer state.
  """

  positions = get_positions(example.input_tokens, pad_id)

  # Forward and backward passes.
  train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(
      params,
      model=model,
      input_tokens=example.input_tokens,
      input_mask=example.target_mask,
      positions=positions,
  )
  # Update the parameters.
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)

  return train_loss, params, opt_state
```

建立不含反向傳遞的 `validation_step` 函數：

```python { .notebook-cell }
@functools.partial(jax.jit, static_argnames=['model'])
def validation_step(
    model: recurrentgemma.Griffin,
    params: Params,
    pad_id: int,
    example: TrainingInput,
) -> jax.Array:
  return forward_and_loss_fn(
      params,
      model=model,
      input_tokens=example.input_tokens,
      input_mask=example.target_mask,
      positions=get_positions(example.input_tokens, pad_id),
  )
```
定義訓練迴圈：

```python { .notebook-cell }
def train_loop(
    model: recurrentgemma.Griffin,
    params: Params,
    optimizer: optax.GradientTransformation,
    train_ds: Iterator[TrainingInput],
    validation_ds: Iterator[TrainingInput],
    num_steps: int | None = None,
    eval_every_n: int = 20,
):
  opt_state = jax.jit(optimizer.init)(params)

  step_counter = 0
  avg_loss=0

  # The first round of the validation loss.
  n_steps_eval = 0
  eval_loss = 0
  for val_example in validation_ds.as_numpy_iterator():
    eval_loss += validation_step(
        model, params, dataset_builder._tokenizer.pad_id, val_example
    )
    n_steps_eval += 1
  print(f"Start, validation loss: {eval_loss/n_steps_eval}")

  for train_example in train_ds:
    train_loss, params, opt_state = train_step(
        model=model,
        params=params,
        optimizer=optimizer,
        opt_state=opt_state,
        pad_id=dataset_builder._tokenizer.pad_id,
        example=train_example,
    )

    step_counter += 1
    avg_loss += train_loss
    if step_counter % eval_every_n == 0:
      eval_loss = 0

      n_steps_eval = 0
      val_iterator = validation_ds.as_numpy_iterator()
      for val_example in val_iterator:
        eval_loss += validation_step(
            model,
            params,
            dataset_builder._tokenizer.pad_id,
            val_example,
        )
        n_steps_eval +=1
      avg_loss /= eval_every_n
      eval_loss /= n_steps_eval
      print(f"STEP {step_counter} training loss: {avg_loss} - eval loss: {eval_loss}")
      avg_loss=0
    if num_steps is not None and step_counter > num_steps:
      break
  return params
```

這裡您必須選擇一個（Optax）最佳化器。對於記憶體較小的裝置，建議使用 SGD，因為它的記憶體佔用低得多。若要達到最佳的微調效能，可以嘗試 Adam-W。本範例針對 `2b-it` 檢查點提供了各個最佳化器在本筆記本特定任務上的最佳超參數。

```python { .notebook-cell }
def griffin_weight_decay_mask(params_like: optax.Params) -> Any:
  # Don't put weight decay on the RGLRU, the embeddings and any biases
  def enable_weight_decay(path: list[Any], _: Any) -> bool:
    # Parameters in the LRU and embedder
    path = [dict_key.key for dict_key in path]
    if 'rg_lru' in path or 'embedder' in path:
      return False
    # All biases and scales
    if path[-1] in ('b', 'scale'):
      return False
    return True

  return jax.tree_util.tree_map_with_path(enable_weight_decay, params_like)

optimizer_choice = "sgd"

if optimizer_choice == "sgd":
  optimizer = optax.sgd(learning_rate=1e-3)
  num_steps = 300
elif optimizer_choice == "adamw":
  optimizer = optax.adamw(
        learning_rate=1e-4,
        b2=0.96,
        eps=1e-8,
        weight_decay=0.1,
        mask=griffin_weight_decay_mask,
    )
  num_steps = 100
else:
  raise ValueError(f"Unknown optimizer: {optimizer_choice}")
```

準備訓練與驗證資料集：

```python { .notebook-cell }
# Choose a small sequence length size, so that everything fits in memory.
num_epochs = 1
batch_size = 1
sequence_length = 32

# Make the dataset builder.
tokenizer = GriffinTokenizer(vocab)
dataset_builder= MTNTDatasetBuilder(tokenizer, sequence_length + 1)

# Build the training dataset.
train_ds = dataset_builder.get_train_dataset(
    batch_size=batch_size,
    num_epochs=num_epochs,
).as_numpy_iterator()

# Build the validation dataset, with a limited number of samples for this demo.
validation_ds = dataset_builder.get_validation_dataset(
    batch_size=batch_size,
).take(50)
```

開始在有限的步數（`num_steps`）上微調 RecurrentGemma（Griffin）模型：

```python { .notebook-cell }
trained_params = train_loop(
    model=model,
    params=params,
    optimizer=optimizer,
    train_ds=train_ds,
    validation_ds=validation_ds,
    num_steps=num_steps,
)
```

``` { .no-copy .output }
Start, validation loss: 7.894117832183838
/usr/local/lib/python3.10/dist-packages/jax/_src/interpreters/mlir.py:920: UserWarning: Some donated buffers were not usable: ShapedArray(int32[1,33]), ShapedArray(bool[1,33]), ShapedArray(int32[], weak_type=True).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer_donation.
  warnings.warn("Some donated buffers were not usable:"
STEP 20 training loss: 4.592616081237793 - eval loss: 2.847407102584839
STEP 40 training loss: 2.7537424564361572 - eval loss: 2.9258534908294678
STEP 60 training loss: 2.835618257522583 - eval loss: 2.4382340908050537
STEP 80 training loss: 2.6322107315063477 - eval loss: 2.3696839809417725
STEP 100 training loss: 1.8703256845474243 - eval loss: 2.355681896209717
STEP 120 training loss: 2.7280433177948 - eval loss: 2.4059958457946777
STEP 140 training loss: 2.3047447204589844 - eval loss: 2.083082914352417
STEP 160 training loss: 2.3432137966156006 - eval loss: 2.095074415206909
STEP 180 training loss: 2.1081202030181885 - eval loss: 2.006460189819336
STEP 200 training loss: 2.5359647274017334 - eval loss: 1.9667452573776245
STEP 220 training loss: 2.202195644378662 - eval loss: 1.9440618753433228
STEP 240 training loss: 2.756615400314331 - eval loss: 2.1073737144470215
STEP 260 training loss: 2.5128934383392334 - eval loss: 2.117241859436035
STEP 280 training loss: 2.73045015335083 - eval loss: 1.9159646034240723
STEP 300 training loss: 2.0918595790863037 - eval loss: 1.9742532968521118
```

訓練損失與驗證損失都應該隨步數增加而下降。

為確保您的輸入符合訓練格式，請記得使用前綴 `Translate this into French:\n`，並在結尾加上換行字元。這會指示模型開始翻譯。

```python { .notebook-cell }
sampler.params = trained_params
output = sampler(
    ["Translate this into French:\nHello, my name is Morgane.\n"],
    total_generation_steps=100,
)
print(output.text[0])
```

``` { .no-copy .output }
/usr/local/lib/python3.10/dist-packages/jax/_src/interpreters/mlir.py:920: UserWarning: Some donated buffers were not usable: ShapedArray(int32[1,16]).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer_donation.
  warnings.warn("Some donated buffers were not usable:"
Mais je m'appelle Morgane.
```

## 了解更多

- 您可以前往 GitHub 上的 Google DeepMind [`recurrentgemma` 函式庫](https://github.com/google-deepmind/recurrentgemma){ data-model="recurrentgemma" } 了解更多，其中包含本教學所使用方法與模組的 docstring，例如 [`recurrentgemma.jax.load_parameters`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/utils.py#L31){ data-model="recurrentgemma" }、[`recurrentgemma.jax.Griffin`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/griffin.py#L29){ data-model="recurrentgemma" } 與 [`recurrentgemma.jax.Sampler`](https://github.com/google-deepmind/recurrentgemma/blob/e4939f9b7edf8baa1d512fb86bfc2e206044d66b/recurrentgemma/jax/sampler.py#L74){ data-model="recurrentgemma" }。
- 以下函式庫各有自己的文件網站：[核心 JAX](https://jax.readthedocs.io)、[Flax](https://flax.readthedocs.io)、[Chex](https://chex.readthedocs.io/en/latest/)、[Optax](https://optax.readthedocs.io/en/latest/) 與 [Orbax](https://orbax.readthedocs.io/)。
- 關於 `sentencepiece` 分詞/還原詞元文件，請參閱 [Google 的 `sentencepiece` GitHub 儲存庫](https://github.com/google/sentencepiece)。
- 關於 `kagglehub` 文件，請參閱 [Kaggle 的 `kagglehub` GitHub 儲存庫](https://github.com/Kaggle/kagglehub) 中的 `README.md`。
- 了解如何[在 Google Cloud Vertex AI 上使用 Gemma 模型][vertex-use-gemma]。
- 如果您使用 Google Cloud TPU（v3-8 及更新版本），請務必同時更新到最新的 `jax[tpu]` 套件（`!pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`），重新啟動執行環境，並確認 `jax` 與 `jaxlib` 版本相符（`!pip list | grep jax`）。這可以避免因 `jaxlib` 與 `jax` 版本不符而產生的 `RuntimeError`。更多 JAX 安裝說明請參閱 [JAX 文件](https://jax.readthedocs.io/en/latest/tutorials/installation.html#install-google-tpu)。
- 閱讀 Google DeepMind 的論文 [RecurrentGemma: Moving Past Transformers
  for Efficient Open Language Models](https://arxiv.org/pdf/2404.07839)。
- 閱讀 Google DeepMind 的論文 [Griffin: Mixing Gated Linear Recurrences with
  Local Attention for Efficient Language Models](https://arxiv.org/pdf/2402.19427)，深入了解 RecurrentGemma 所使用的模型架構。
