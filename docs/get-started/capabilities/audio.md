---
title: 音訊理解
description: 從 Gemma 3n 開始，您可以直接在提示與工作流程中使用音訊輸入，打造語音辨識、翻譯等各類音訊應用。
models: [gemma-4]
requires: audio
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/audio-gemma-4-e2b.ipynb
---

# 音訊理解

從 [Gemma 3n](../../models/gemma-3n/index.md) 開始，您可以在提示與工作流程中直接使用音訊。音訊與口說語言是豐富的資料來源，可用於捕捉使用者意圖、記錄周遭世界的資訊，以及理解
需要解決的具體問題。

本指南概覽 [Gemma 4](../../models/gemma-4/index.md) 的音訊處理能力，包括自動語音辨識（ASR）、翻譯與通用語音理解。


## 安裝 Python 套件

安裝執行 Gemma 模型與發出請求所需的 Hugging Face 程式庫。

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.10.1"
```

## 載入模型

使用 [`transformers`](https://huggingface.co/docs/transformers/installation) 程式庫，透過 `AutoProcessor` 與 `AutoModelForImageTextToText` 類別建立 `processor` 與 `model` 的實例，如以下程式碼範例所示：

```python { .notebook-cell .model-swap }
MODEL_ID = "google/gemma-4-E2B-it"

from transformers import pipeline

pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)
```

## 音訊資料

數位音訊資料有許多格式與解析度等級。實際能搭配 Gemma 使用的音訊格式（例如 MP3 與 WAV 格式）取決於您選擇用來將聲音資料轉換為張量的框架。以下是為 Gemma 準備音訊資料時的一些具體考量：

- **詞元成本：** Gemma 4 每秒音訊消耗 25 個詞元（Gemma 3n 為每秒 6.25 個詞元）。
- **片段長度：** 音訊支援的最大長度為 30 秒。
- **音訊聲道：** 音訊資料會以單聲道處理。如果您使用多聲道音訊（例如左右聲道），建議移除聲道或將聲音資料混合為單一聲道，藉此將資料縮減為單聲道。
- **技術編碼：**
  - **取樣率：** 16kHz
  - **位元深度：** 32 位元浮點數格式，樣本已正規化於 \[-1, 1\] 範圍內。

如果您打算處理的音訊資料與輸入
處理方式有顯著差異，特別是在聲道、取樣率與位元深度方面，
建議重新取樣或裁剪音訊資料，使其符合模型
可處理的資料解析度。

## 音訊編碼

雖然高階程式庫（例如 Hugging Face `AutoProcessor`）通常會自動處理音訊前處理，但有時您可能需要實作自訂編碼。

當您自行編寫程式碼為 Gemma 編碼音訊資料時，應遵循建議的轉換流程。如果您
處理的是以特定格式編碼的音訊檔案，例如 MP3 或 WAV 編碼資料，
必須先使用 `ffmpeg` 等程式庫將其解碼為樣本。解碼後，將音訊轉換為範圍介於 \[-1, 1\] 之間的單聲道、16 kHz float32 波形。例如，如果您處理的是 44.1 kHz、立體聲、有號 16 位元 PCM 整數 WAV 檔案，請依下列步驟處理：

- 將音訊資料重新取樣為 16 kHz
- 對 2 個聲道取平均，將立體聲降混為單聲道
- 將 int16 轉換為 float32，並除以 32768.0 以縮放至 \[-1, 1\] 範圍

!!! Note
    將音訊重新取樣為 16 kHz 時，為了獲得最佳結果應使用傅立葉方法，例如 `scipy.signal.resample` 或 `librosa.sample(res_type ='scipy')`。

## 語音轉文字

Gemma 4 E2B、E4B 與 12B Unified 經過多語言語音辨識訓練，能將各種語言的音訊輸入轉錄為文字。

請使用以下提示結構進行**語音辨識（Audio Speech Recognition, ASR）**。

```text
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
*   Only output the transcription, with no newlines.
*   When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

以下程式碼範例示範如何使用 Hugging Face Transformers 提示模型，從音訊檔案轉錄文字：

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 64
gen_kwargs = dict(generation_config=config)

RESOURCE_URL_PREFIX = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            #{"type": "text", "text": "Transcribe the following speech segment in English into English text. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal1.wav"},
        ]
    }
]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
I woke up early today feeling really fresh the morning light was beautiful and I enjoyed a nice cup of coffee<turn|>
```

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 1024
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Give me a concise overview of these audio files."},
            {"type": "text", "text": "journal1:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal1.wav"},
            {"type": "text", "text": "journal2:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal2.wav"},
            {"type": "text", "text": "journal3:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal3.wav"},
            {"type": "text", "text": "journal4:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal4.wav"},
            {"type": "text", "text": "journal5:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal5.wav"},
        ]
    }
]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
Here is a concise overview of each audio file:

**journal1:** The speaker describes a fresh and peaceful day, enjoying a cup of coffee.
**journal2:** The speaker had a perfect day at the park, including a walk and watching cherry blossoms.
**journal3:** The speaker finished the day with a good book, feeling grateful for simple moments.
**journal4:** The speaker returned from work and noted the beautiful night sky and a clear view from the train.
**journal5:** The speaker had a great lunch with an old friend, which was a pleasant way to catch up and made their day.
<turn|>
```

## 自動語音翻譯

Gemma 4 E2B、E4B 與 12B Unified 經過多語言語音翻譯訓練，能將口說音訊直接翻譯成另一種語言。

請使用以下提示結構進行**自動語音翻譯（Automatic Speech Translation, AST）**。

```text
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

以下程式碼範例示範如何使用 Hugging Face Transformers 提示模型，將口說音訊翻譯成文字：

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 64
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in English, then translate it into Korean. When formatting the answer, first output the transcription in English, then one newline, then output the string 'Korean: ', then the translation in Korean."},
            {"type": "audio", "audio": "https://ai.google.dev/gemma/docs/audio/roses-are.wav"},
        ]
    }
]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
Roses are red, violets are blue.
Korean: 장미는 빨갛고, 제비꽃은 파랗다.<turn|>
```

## 自動語音翻譯 / 自動語音辨識

自己動手試試

```bash { .notebook-cell }
pip install ipywebrtc
```

按下圓形按鈕開始說話。說完後再按一次圓形按鈕。小工具會立即播放它錄下的內容。

```python { .notebook-cell }
from google.colab import output
output.enable_custom_widget_manager()

from ipywebrtc import AudioRecorder, CameraStream

camera = CameraStream(constraints={'audio': True,'video':False})
recorder = AudioRecorder(stream=camera)
recorder
```

將 webm 檔案轉換為 PyTorch 能理解的 wav 格式。

```python { .notebook-cell }
with open('/content/recording.webm', 'wb') as f:
    f.write(recorder.audio.value)
!ffmpeg -i /content/recording.webm /content/recording.wav -y
```

``` { .no-copy .output }
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Input #0, matroska,webm, from '/content/recording.webm':
  Metadata:
    encoder         : Chrome
  Duration: 00:00:03.00, start: 0.000000, bitrate: 132 kb/s
  Stream #0:0(eng): Audio: opus, 48000 Hz, mono, fltp (default)
Stream mapping:
  Stream #0:0 -> #0:0 (opus (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to '/content/recording.wav':
  Metadata:
    ISFT            : Lavf58.76.100
  Stream #0:0(eng): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, mono, s16, 768 kb/s (default)
    Metadata:
      encoder         : Lavc58.134.100 pcm_s16le
size=     287kB time=00:00:02.99 bitrate= 783.7kbits/s speed=79.4x    
video:0kB audio:287kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.026552%
```

### ASR

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 64
gen_kwargs = dict(generation_config=config)

messages = [{
  "role": "user",
  "content": [
    {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
    {"type": "audio", "audio": "/content/recording.wav"},
  ]
}]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
How can I get to the station?<turn|>
```

### AST

```python { .notebook-cell }
messages = [{
  "role": "user",
  "content": [
    {"type": "text", "text": "Transcribe the following speech segment in English, then translate it into Korean. When formatting the answer, first output the transcription in English, then one newline, then output the string 'Korean: ', then the translation in Korean."},
    {"type": "audio", "audio": "/content/recording.wav"},
  ]
}]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
How can I get to the station?
Korean: 역에 어떻게 가나요?<turn|>
```

## 總結

在本指南中，您學會了如何使用 Gemma 4 模型處理音訊。範例示範了如何執行語音轉文字（Speech-to-Text, ASR）來轉錄口說語言，以及如何執行自動語音翻譯（Automated Speech Translation, AST）將口說音訊直接翻譯成另一種語言。您也看到了如何在筆記本環境中透過麥克風擷取音訊進行處理。
## 下一步

<div class="card-grid">
<a class="card" href="../prompt-formatting.md">
  <span class="card-icon">{{ icon('lucide/braces') }}</span>
  <div class="card-title">提示格式設定</div>
  <p class="card-desc">構成 Gemma 提示的控制詞元。</p>
</a>
<a class="card" href="mtp.md">
  <span class="card-icon">{{ icon('lucide/zap') }}</span>
  <div class="card-title">多詞元預測</div>
  <p class="card-desc">在相同品質下以投機解碼加速推論。</p>
</a>
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">執行 Gemma</div>
  <p class="card-desc">從筆記型電腦到叢集，各種能執行 Gemma 的引擎。</p>
</a>
</div>
