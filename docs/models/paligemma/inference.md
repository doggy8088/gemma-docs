---
title: 使用 Keras 產生 PaliGemma 輸出
description: PaliGemma 模型具備多模態能力，讓您能同時使用文字與圖片輸入資料來產生輸出。
models: [paligemma]
frameworks: [keras]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/paligemma-inference.ipynb
source_url: "https://ai.google.dev/gemma/docs/paligemma/inference-with-keras.md.txt"
---
# 使用 Keras 產生 PaliGemma 輸出


PaliGemma 模型具備*多模態*能力，讓您能同時使用文字與圖片輸入資料來產生輸出。您可以使用圖片資料為請求提供額外上下文，或使用模型分析圖片內容。本教學課程說明如何使用 PaliGemma 搭配 Keras 來分析圖片並回答相關問題。

## 本筆記本包含哪些內容

本筆記本使用 PaliGemma 搭配 Keras，並說明如何：

- 安裝 Keras 與所需的相依套件
- 下載 `PaliGemmaCausalLM`（用於因果視覺語言建模的預訓練 PaliGemma 變體），並用它來建立模型
- 測試模型推論所提供圖片資訊的能力

## 開始之前

在開始本筆記本之前，您應該熟悉 Python 程式碼，以及大型語言模型（LLM）的訓練方式。您不需要熟悉 Keras，但在閱讀範例程式碼時，具備 Keras 的基礎知識會很有幫助。

## 設定

以下各節說明讓筆記本能使用 PaliGemma 模型的前置步驟，包括取得模型存取權、取得 API 金鑰，以及設定筆記本執行環境。

### 取得 PaliGemma 存取權

首次使用 PaliGemma 之前，您必須透過 Kaggle 完成以下步驟來申請模型存取權：

1. 登入 [Kaggle](https://www.kaggle.com)，若尚未擁有帳號，請建立新的 Kaggle 帳號。
2. 前往 [PaliGemma 模型卡](https://www.kaggle.com/models/google/paligemma-2/){ data-model="paligemma" }，然後點選 **Request Access**。
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

### 設定環境變數

為 `KAGGLE_USERNAME`、`KAGGLE_KEY` 與 `KERAS_BACKEND` 設定環境變數。

```python { .notebook-cell }
import os
from google.colab import userdata

# Set up environmental variables
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
os.environ["KERAS_BACKEND"] = "jax"
```

### 安裝 Keras

執行以下儲存格以安裝 Keras。

```bash { .notebook-cell }
pip install -U -q keras-nlp keras-hub kagglehub
```

### 匯入相依套件並設定 Keras

安裝本筆記本所需的相依套件並設定 Keras 的後端。您還會將 Keras 設定為使用 `bfloat16`，讓框架使用較少的記憶體。

```python { .notebook-cell }
import keras
import keras_hub
import numpy as np
import PIL
import requests
import io
import matplotlib
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

keras.config.set_floatx("bfloat16")
```

## 載入模型

設定完成後，您可以下載預訓練模型，並建立一些公用方法來協助模型產生回應。在此步驟中，您將使用來自 Keras Hub 的 `PaliGemmaCausalLM` 下載模型。這個類別可協助您管理並執行 PaliGemma 的因果視覺語言模型結構。*因果視覺語言模型*會根據先前的詞元預測下一個詞元。Keras Hub 提供了許多熱門[模型架構](https://keras.io/keras_hub/api/models/)的實作。

使用 `from_preset` 方法建立模型並列印其摘要。這個過程約需一分鐘完成。

```python { .notebook-cell }
paligemma = keras_hub.models.PaliGemmaCausalLM.from_preset("kaggle://keras/paligemma2/keras/pali_gemma2_mix_3b_224")
paligemma.summary()
```

## 建立公用方法

為了協助您從模型產生回應，請建立兩個公用方法：

- **`crop_and_resize`：** `read_img` 的輔助方法。這個方法會裁剪並調整圖片為傳入的尺寸，讓最終圖片在調整大小時不會扭曲圖片比例。
- **`read_img`：** `read_img_from_url` 的輔助方法。這個方法會實際開啟圖片、調整大小以符合模型的限制，並將其放入模型可解讀的陣列中。
- **`read_img_from_url`：** 透過有效的 URL 接收圖片。您需要這個方法將圖片傳遞給模型。

您將在本筆記本的下一步使用 `read_img_from_url`。

```python { .notebook-cell }
def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def read_image(url, target_size):
    contents = io.BytesIO(requests.get(url).content)
    image = PIL.Image.open(contents)
    image = crop_and_resize(image, target_size)
    image = np.array(image)
    # Remove alpha channel if necessary.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def parse_bbox_and_labels(detokenized_output: str):
  matches = re.finditer(
      '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
      ' (?P<label>.+?)( ;|$)',
      detokenized_output,
  )
  labels, boxes = [], []
  fmt = lambda x: float(x) / 1024.0
  for m in matches:
    d = m.groupdict()
    boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
    labels.append(d['label'])
  return np.array(boxes), np.array(labels)

def display_boxes(image, boxes, labels, target_image_size):
  h, l = target_size
  fig, ax = plt.subplots()
  ax.imshow(image)
  for i in range(boxes.shape[0]):
      y, x, y2, x2 = (boxes[i]*h)
      width = x2 - x
      height = y2 - y
      # Create a Rectangle patch
      rect = patches.Rectangle((x, y),
                               width,
                               height,
                               linewidth=1,
                               edgecolor='r',
                               facecolor='none')
      # Add label
      plt.text(x, y, labels[i], color='red', fontsize=12)
      # Add the patch to the Axes
      ax.add_patch(rect)

  plt.show()

def display_segment_output(image, bounding_box, segment_mask, target_image_size):
    # Initialize a full mask with the target size
    full_mask = np.zeros(target_image_size, dtype=np.uint8)
    target_width, target_height = target_image_size

    for bbox, mask in zip(bounding_box, segment_mask):
        y1, x1, y2, x2 = bbox
        x1 = int(x1 * target_width)
        y1 = int(y1 * target_height)
        x2 = int(x2 * target_width)
        y2 = int(y2 * target_height)

        # Ensure mask is 2D before converting to Image
        if mask.ndim == 3:
            mask = mask.squeeze(axis=-1)
        mask = Image.fromarray(mask)
        mask = mask.resize((x2 - x1, y2 - y1), resample=Image.NEAREST)
        mask = np.array(mask)
        binary_mask = (mask > 0.5).astype(np.uint8)


        # Place the binary mask onto the full mask
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], binary_mask)
    cmap = plt.get_cmap('jet')
    colored_mask = cmap(full_mask / 1.0)
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    if isinstance(image, Image.Image):
        image = np.array(image)
    blended_image = image.copy()
    mask_indices = full_mask > 0
    alpha = 0.5

    for c in range(3):
        blended_image[:, :, c] = np.where(mask_indices,
                                          (1 - alpha) * image[:, :, c] + alpha * colored_mask[:, :, c],
                                          image[:, :, c])

    fig, ax = plt.subplots()
    ax.imshow(blended_image)
    plt.show()
```

## 產生輸出

載入模型並建立公用方法後，您可以使用圖片與文字資料提示模型以產生回應。PaliGemma 模型針對特定任務使用特定的提示語法進行訓練，例如 `answer`、`caption` 與 `detect`。如需更多關於 PaliGemma 提示任務語法的資訊，請參閱 [PaliGemma 提示與系統指令](prompting.md#提示任務語法)。

使用以下程式碼將測試圖片載入物件，為生成提示準備圖片：

```python { .notebook-cell }
target_size = (224, 224)
image_url = 'https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png'
cow_image = read_image(image_url, target_size)
matplotlib.pyplot.imshow(cow_image)
```

### 以特定語言回答

以下範例程式碼說明如何提示 PaliGemma 模型，詢問所提供圖片中出現的物體資訊。這個範例使用 `answer {lang}` 語法，並展示其他語言的額外問題：

```python { .notebook-cell }
prompt = 'answer en where is the cow standing?\n'
# prompt = 'svar no hvor står kuen?\n'
# prompt = 'answer fr quelle couleur est le ciel?\n'
# prompt = 'responda pt qual a cor do animal?\n'

output = paligemma.generate(
    inputs={
        "images": cow_image,
        "prompts": prompt,
    }
)
print(output)
```

!!! Note
    使用 PaliGemma 提示命令語法的提示，結尾必須加上 "`\n`" 字元。


### 使用 `detect` 提示

以下範例程式碼使用 `detect` 提示語法，在所提供的圖片中定位物體。這段程式碼使用先前定義的 `parse_bbox_and_labels()` 與 `display_boxes()` 函式來解讀模型輸出，並顯示產生的邊界框。

```python { .notebook-cell }
prompt = 'detect cow\n'
output = paligemma.generate(
    inputs={
        "images": cow_image,
        "prompts": prompt,
    }
)
boxes, labels = parse_bbox_and_labels(output)
display_boxes(cow_image, boxes, labels, target_size)
```

### 使用 `segment` 提示

以下範例程式碼使用 `segment` 提示語法，在圖片中定位物體所佔的區域。它使用 Google 的 `big_vision` 程式庫來解讀模型輸出，並為分割的物體產生遮罩。

開始之前，請如下方程式碼範例所示，安裝 `big_vision` 程式庫及其相依套件：

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

對於這個分割範例，請載入並準備一張包含貓的不同圖片。

```python { .notebook-cell }
cat = read_image('https://big-vision-paligemma.hf.space/file=examples/barsik.jpg', target_size)
matplotlib.pyplot.imshow(cat)
```

以下是協助解析 PaliGemma 分割輸出的函式

```python { .notebook-cell }
import  big_vision.evaluators.proj.paligemma.transfers.segmentation as segeval
reconstruct_masks = segeval.get_reconstruct_masks('oi')
def parse_segments(detokenized_output: str) -> tuple[np.ndarray, np.ndarray]:
  matches = re.finditer(
      '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
        + ''.join(f'<seg(?P<s{i}>\d\d\d)>' for i in range(16)),
      detokenized_output,
  )
  boxes, segs = [], []
  fmt_box = lambda x: float(x) / 1024.0
  for m in matches:
    d = m.groupdict()
    boxes.append([fmt_box(d['y0']), fmt_box(d['x0']), fmt_box(d['y1']), fmt_box(d['x1'])])
    segs.append([int(d[f's{i}']) for i in range(16)])
  return np.array(boxes), np.array(reconstruct_masks(np.array(segs)))
```

查詢 PaliGemma 以分割圖片中的貓

```python { .notebook-cell }
prompt = 'segment cat\n'
output = paligemma.generate(
    inputs={
        "images": cat,
        "prompts": prompt,
    }
)
```

視覺化 PaliGemma 產生的遮罩

```python { .notebook-cell }
bboxes, seg_masks = parse_segments(output)
display_segment_output(cat, bboxes, seg_masks, target_size)
```

### 批次提示

您可以在單一提示中提供多個提示命令，作為一批指令。以下範例示範如何建構提示文字以提供多個指令。

!!! Important
    每個提示命令結尾都必須加上 "\\n" 字元，如範例所示。


```python { .notebook-cell }
prompts = [
    'answer en where is the cow standing?\n',
    'answer en what color is the cow?\n',
    'describe en\n',
    'detect cow\n',
    'segment cow\n',
]
images = [cow_image, cow_image, cow_image, cow_image, cow_image]
outputs = paligemma.generate(
    inputs={
        "images": images,
        "prompts": prompts,
    }
)
for output in outputs:
    print(output)
```
