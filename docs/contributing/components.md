---
title: 元件
models: [gemma-4]
frameworks: [ollama]
description: 文件網站所用的可重複使用建構元件：卡片、徽章、步驟清單、嵌入元件，附即時範例與可直接複製的標記語法。
---

# 元件

本站的每個視覺元素都來自這份型錄。請用這些元件來建構頁面，而不是撰寫新的 CSS——如果需要的東西不在這裡，請加到 `docs/stylesheets/components.css`，並在本頁記錄它。

!!! note
    顏色、間距、圓角與陰影都是**設計權杖（token）**，只在 `docs/stylesheets/gemma.css` 定義一次。它是唯一允許包含原始十六進位色碼的樣式表，由 `scripts/lint_docs.py --check css` 強制執行。

## 顏色權杖

品牌藍只有**一個開關**。`docs/stylesheets/gemma.css` 中的 `--gemma-blue-seed` 是唯一需要更改的顏色；每個色階、兩種主題與每個元件都衍生自它：

```css
--gemma-blue-seed: #0B57D0;
```

色階是種子色在 sRGB 中與白色混合（深色端為黑色）的結果，因此保持在種子色自身的色相上。以目前的種子色為例：

| 色階 | 十六進位 | 對比度 | 用途 |
|---|---|---|---|
| `--gemma-blue-50` / `-100` | `#EBF2FB` / `#CEDDF6` | — | 淺色調、懸停淡色。 |
| `--gemma-blue-300` | `#80A8E7` | 6.65:1（ink 上） | 深色主題文字。 |
| `--gemma-blue-500` | `#417CDA` | 4.10:1（白色下） | **僅裝飾用**——漸層、2px 線條。 |
| `--gemma-blue-600` | `#2368D5` | 5.23:1（白色下） | 深色主題填色。 |
| `--gemma-blue-700` | `#0B57D0` | 6.39:1（雙向） | 主色。 |
| `--gemma-blue-800` | `#0944A2` | 8.90:1（白色下） | 按壓／強調狀態。 |

提交新的種子色之前，先檢查它：

```bash
uv run python scripts/check_contrast.py --seed "#RRGGBB"
```

它會印出衍生的色階，如果任何色階未達其用途所需的閾值，就以非零狀態碼結束，因此顏色變更永遠不會在無聲中破壞可讀性。

在元件中使用語意角色，不要直接使用色階：

| 權杖 | 用途 |
|---|---|
| `--gemma-accent` | 目前主題中**文字安全**的品牌色。建議優先使用。 |
| `--gemma-accent-fill` | **承載白色文字**的填色。絕對不要指向淺色階。 |
| `--gemma-white`、`--gemma-ink` | 表面。 |
| `--gemma-blue`、`-deep`、`-light` | 色階的舊版別名。請優先使用上述角色。 |

!!! warning "對比是對稱的"
    在白色*上*太淺而難以閱讀的顏色，在白色文字*背後*也一樣太淺——對比率是同一個。被取代的色盤以 3.51:1 的 `#3186FF` 為基礎，雙向都不合格；這就是它看起來像霓虹燈的原因。承載白色文字的填色來自 600 或 700 色階。

比例：`--gemma-space-1…7`、`--gemma-radius-sm|md|lg|pill`、`--gemma-shadow-sm|md|lg`。

## 共用連結

用於多個頁面的 URL 放在 `data/links.yml` 中，絕不直接寫在內文。`ai.google.dev/responsible` 出現在 26 個頁面上；當這類 URL 變動時，只需要改這一行。

```yaml
hf-gemma4:
  url: https://huggingface.co/collections/google/gemma-4
  model: gemma-4        # only when the URL names a generation
```

頁面將它當作一般的參照連結使用——沒有 URL、沒有屬性：

```markdown
Every checkpoint is in the [Hugging Face collection][hf-gemma4].
```

`scripts/generate_snippets.py` 將定義渲染到 `snippets/links.md`，`pymdownx.snippets` 再將它附加到每個頁面。定義渲染出來是空的，所以到處附加它不花任何代價。

`model:` 鍵是 `pinned_links.py` 回報的依據，因此連結的世代在 URL 旁邊**只宣告一次**，而不是在每個使用處重複宣告。

!!! warning "未解析的參照會靜默失敗"
    `[text][typo]` 會渲染成字面文字——連括號一起——而不是報錯。`references` lint 檢查會抓到它：每個參照都必須能解析，來源是 `data/links.yml` 或頁面自身的定義。

## 詞彙表

術語放在 `data/glossary.yml` 中，渲染為帶有懸停定義的縮寫。主題給它們 `cursor: help` 與灰色虛線底線，因此術語看起來像術語而不是連結——連結是藍色加指標游標。

與共用連結不同，詞彙表是**每頁選擇加入**：

```markdown
--8<-- "glossary.md"
```

Python-Markdown 會標記術語的*每一次*出現，因此全域詞彙表會在 MTP 頁面上放上十一個虛線底線、在全站 GGUF 相關處放上四十一個。只在縮寫被使用但未被解釋的地方加入它；頁面本身就是解釋時就不要加。

## 圖示

Zensical 內建五套圖示：**lucide**（線條圖示，本站預設）、**material**、**fontawesome**、**octicons** 與 **simple**（品牌標誌）。專案專用的圖示放在 `overrides/.icons/`。

**在正文中**，使用短碼：

```markdown
Run it locally :lucide-rocket: and you are done.
```

**在標記語法中**——任何卡片、網格或產生的區塊——改用 `icon()` 巨集：

```jinja
{% raw %}<span class="card-icon">{{ icon('lucide/rocket') }}</span>
{{ icon('simple/ollama', 'engine-logo') }}{% endraw %}
```

!!! warning "短碼在 HTML 內無效"
    `:lucide-rocket:` 是*行內 Markdown* 結構。在原始 HTML 區塊內——每個卡片網格都是——它會保持為字面文字。給區塊加上 `markdown` 確實能渲染它，但 `md_in_html` 接著會把每張卡片包在 `<p>` 裡，網格版面就壞了。`icon()` 巨集在 Jinja 預處理器中執行、早於 Markdown，因此 SVG 已經就位。它也能作用於從 `data/*.yml` 產生的網格，短碼在那裡完全派不上用場。

圖示繼承 `currentColor`，所以把顏色設在父層，絕不設在 SVG 上。尺寸來自 `icons.css` 的 CSS；巨集會移除圖示檔案自身的 `width`/`height`，讓各套圖示無法互相衝突。

!!! danger "不要使用表情符號短碼"
    `:rocket:` 會渲染成 `<img src="https://cdn.jsdelivr.net/...twemoji...">`——每個表情符號都是一次外部 CDN 請求，會破壞離線檢視並增加第三方依賴。原始的 Unicode 表情符號避開了請求，但以系統字型渲染，在每個平台上看起來都不同，也無法設定樣式。請使用圖示集。

## 方框

本站每個有邊框的容器——卡片、引擎磁磚、樹節點、表格——都用同一套配方。只有內距不同。

```css
background: var(--gemma-surface);
border: 1px solid var(--gemma-border);
border-radius: var(--gemma-radius-md);
```

```css
/* Interactive boxes, on hover and focus */
border-color: var(--gemma-accent);
box-shadow: var(--gemma-shadow-sm);
```

| 密度 | 內距 | 用於 |
|---|---|---|
| 緊湊 | `--gemma-space-2` / `-3` | 引擎磁磚、樹節點 |
| 預設 | `--gemma-space-4` / `-5` | 卡片 |
| 面板 | `--gemma-space-3` / `-4` | 提示框，在 `--gemma-surface-sunken` 上加側線 |

邊框不要用 Material 的 `--md-default-fg-color--lighter`。它是 0.32 alpha 的黑色，畫出來的線比 `--gemma-border` 明顯更重；卡片用了它很久，這就是它們看起來像另一套設計語言的原因。

## 卡片

連結網格。用於 cookbook、模型總覽與生態系頁面。

<div class="card-grid">
<a class="card" href="#cards">
  <div class="card-title">卡片標題 <span class="card-ext">↗</span></div>
  <p class="card-desc">一到兩行說明這張卡片連往何處。</p>
</a>
<a class="card" href="#cards">
  <div class="card-title">第二張卡片</div>
  <p class="card-desc">網格會自動重排，在行動裝置上收合為單欄。</p>
</a>
</div>

```html
<div class="card-grid">
<a class="card" href="…">
  <div class="card-title">Card title <span class="card-ext">↗</span></div>
  <p class="card-desc">One or two lines describing where this card leads.</p>
</a>
</div>
```

`.cookbook-grid` / `.cookbook-card` 是為轉換器輸出與既有頁面保留的別名。新內容一律使用 `.card-grid` / `.card`。

## 結束頁面

只用兩個標題，且只能是這兩個。絕不用 `Summary and next steps`，絕不用 `Next steps`。

**`## Summary`** 是讀者剛完成內容的文字回顧。選用——用在帶領讀者實際操作的教學，在參考頁則略過。不放卡片。

**`## What's next`** 永遠是 `.card-grid`、永遠有圖示、永遠放在最後。兩到四張卡片，且**每一張都必須指向前方**——絕不連回讀者已經看過的章節。跨章節連結沒問題；Get started 頁面可以連到 Learn。

```markdown
## What's next

<div class="card-grid">
<a class="card" href="../tuning/index.md">
  <span class="card-icon">{% raw %}{{ icon('lucide/graduation-cap') }}{% endraw %}</span>
  <div class="card-title">Fine-tune Gemma</div>
  <p class="card-desc">QLoRA, full fine-tunes and distributed training on your own data.</p>
</a>
</div>
```

連往站外的卡片在標題中放 `↗` 且**不放圖示**，讓箭頭成為「即將離開」的訊號：

```markdown
<a class="card" href="https://…" target="_blank" rel="noopener" data-model="gemma-4">
  <div class="card-title">Gemma on Vertex AI <span class="card-ext">↗</span></div>
  <p class="card-desc">Use Gemma open models on Google Cloud.</p>
</a>
```

注意 `data-model` 以純屬性形式放在錨點上。`attr_list` 語法在 HTML 區塊內無效，而 `scripts/pinned_links.py` 兩種形式都讀得到。

各頁面的結尾配置：

| 章節 | 結尾 |
|---|---|
| Get started、Learn | 每頁都有 `What's next` |
| Run、Fine-tune | `What's next` |
| Models | 有助回顧處用 `Summary`；只有頁面真的有後續可接時才用 `What's next`。模型卡不放任何結尾——規格表就此結束。 |

!!! warning "卡片網格會從筆記本中移除"
    網格是網站導覽：原始 HTML 加上只有 Jinja 預處理器能解析的 {% raw %}`{{ icon() }}`{% endraw %} 呼叫，兩者在 Jupyter 中都會渲染成字面文字。`scripts/notebooks.py` 會捨棄任何 `card-grid` 區塊及其上方的標題，因此註冊為筆記本來源的頁面不需要特殊處理。

## 徽章

模型屬性的行內小標籤。由 `data/models.yml` 驅動，徽章永遠不會與表格不一致。

<p>
  <span class="badge">12B</span>
  <span class="badge">256K 上下文</span>
  <span class="badge badge--accent">推薦</span>
  <span class="badge badge--muted">音訊</span>
</p>

```html
<span class="badge">12B</span>
<span class="badge badge--accent">Recommended</span>
<span class="badge badge--muted">Audio</span>   <!-- capability this model lacks -->
```

`--muted` 是模型切換器（步驟 4.2）顯示所選模型不具備的能力的方式——以刪除線呈現而非隱藏，讓讀者知道這個限制。

## 提示詞元

Gemma 的控制詞元是小標籤，不是程式碼。顏色承載分類資訊，因此瀏覽 `get-started/prompt-formatting.md` 的讀者不用細讀，就能分辨回合標記與工具呼叫。

<p>
  <code class="token-turn">&lt;|turn|&gt;</code>
  <code class="token-role">system</code>
  <code class="token-multi">&lt;|image|&gt;</code>
  <code class="token-tool">&lt;|tool_call|&gt;</code>
  <code class="token-think">&lt;|think|&gt;</code>
</p>

| 類別 | 分類 | 涵蓋 |
|---|---|---|
| `.token-turn` | 回合結構 | <code class="token-turn">&lt;&#124;turn&#124;&gt;</code> <code class="token-turn">&lt;turn&#124;&gt;</code> <code class="token-turn">&lt;start_of_turn&gt;</code> <code class="token-turn">&lt;end_of_turn&gt;</code> |
| `.token-role` | 角色 | <code class="token-role">system</code> <code class="token-role">user</code> <code class="token-role">model</code> <code class="token-role">assistant</code> |
| `.token-multi` | 模態佔位符 | <code class="token-multi">&lt;&#124;image&#124;&gt;</code> <code class="token-multi">&lt;&#124;audio&#124;&gt;</code> |
| `.token-tool` | 工具使用 | <code class="token-tool">&lt;&#124;tool&#124;&gt;</code> <code class="token-tool">&lt;&#124;tool_call&#124;&gt;</code> <code class="token-tool">&lt;&#124;tool_response&#124;&gt;</code> <code class="token-tool">&lt;&#124;"&#124;&gt;</code> |
| `.token-think` | 推理 | <code class="token-think">&lt;&#124;think&#124;&gt;</code> <code class="token-think">&lt;&#124;channel&gt;thought</code> <code class="token-think">&lt;channel&#124;&gt;</code> |

**行內**用法，在正文或表格儲存格中，對行內程式碼使用 `attr_list`：

```markdown
Token to indicate a system instruction: `system`{.token-role}
Enable reasoning with `<|think|>`{.token-think}
```

**在範例對話中**，使用帶 span 的 `<pre><code>` 區塊，讓周圍的提示文字保持無樣式、並保留換行。跳脫角括號：

```html
<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Describe this image: <span class="token-multi">&lt;|image|&gt;</span><span class="token-turn">&lt;turn|&gt;</span></code></pre>
```

渲染結果如下：

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Describe this image: <span class="token-multi">&lt;|image|&gt;</span><span class="token-turn">&lt;turn|&gt;</span></code></pre>

當模型將某個詞元視為一個單位時，就把它放在同一個小標籤裡——<code class="token-think">&lt;|channel&gt;thought</code> 是單一小標籤，不是一個小標籤後接純文字。需要詞元的對話不要用圍欄程式碼區塊：語法高亮與這些類別無法同時套用。

顏色以 `--gemma-cat-*` 三元組形式放在 `gemma.css` 中，且是刻意的分類色——它們彼此保持區分，而不是從品牌藍衍生而來。

## 頁面資訊列

每個 H1 下方的安靜橫帶。你永遠不需要手寫這個標記；它由 `overrides/partials/page-context.html` 從 frontmatter 渲染。*本頁*上的資訊列就是即時範例。

```yaml
models: [gemma-4]        # what the page is about
frameworks: [ollama]     # primary tools the page teaches, linked to their home pages
requires: audio          # narrows the selector to models that accept audio
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/tutorial-gemma-4-e2b.ipynb # opens in Colab
```

這一行有兩種項目，把它們區分開來就是設計重點：

| 項目 | 來源 | 意義 |
|---|---|---|
| `Models` | `models` frontmatter | 頁面**關於**什麼。連結到本地模型頁面。 |
| `Uses` | `frameworks` frontmatter | 頁面教授的主要工具。連到各專案外部。 |
| `Examples` | 頁面自己的程式碼 | **你**希望指令針對的目標。一種偏好設定。 |
| `Colab` | `colab` frontmatter | Colab 筆記本的直接連結；會動態更新以對應所選模型。 |

配對之間以 1px 線分隔，所有值共用同一字重，標籤則為靜音色——光靠 16px 的間距，不足以阻止 `Frameworks Sentence Transformers` 被讀成一串連續的灰色文字。

`frameworks` 保持編輯精選而非鉅細靡遺。列出跟著頁面操作所需、使用者會接觸到的技術堆疊，而不是每個引入或傳遞依賴。例如，引入 PyTorch 張量的 Transformers 教學應寫 `frameworks: [transformers]`；只有當頁面真的教授其 API 或工作流程時，才加上 `pytorch`。

`Examples` 刻意*不是*從 `models:` 衍生的。它們回答不同的問題：Ollama 頁面是關於 Gemma 4，但其指令可以針對 Ollama 發布的任何模型。因此選擇器是根據頁面標記區塊實際可改寫的目標來建構的——這表示在沒有可切換內容的頁面上它會直接消失，而不是宣稱一個頁面從未提及的尺寸。

### 可切換的程式碼區塊

在圍欄加上 `{ .model-swap }`，選擇器就會改寫其中的模型識別碼：

````markdown
```bash { .model-swap }
ollama pull gemma4:12b
```
````

替換是**按區塊選擇加入**，因為頁面的程式碼是按撰寫時的樣子驗證過的。未標記的區塊永遠不會被動——這就是 `deployment/ollama.md` 上的標籤表能列出所有標籤，而其上方的 `pull` 指令能跟隨選擇器的原因。

只有當模型對頁面標記區塊中出現的**每個**引擎都有已發布的 id 時，才會提供該模型，因此切換永遠不會改寫部分程式碼片段、卻靜默留下其他片段。如果頁面在 frontmatter 中指定了 `requires:` 模態（如 `requires: audio`），選擇器與匯出的筆記本會進一步過濾為支援該模態的模型。如果某個刻意獨立的範例缺少某個尺寸——例如 QAT 檢查點——請標記為 `{ .model-fixed }` 並在旁邊說明限制，而不是無謂地縮小頁面主要範例的範圍。

識別碼變體放在 `data/models.yml` 中每個尺寸的 `ids` 映射下。當語法有不同語義時使用不同的變體——例如訓練用 `hf_base`、量化感知 Transformers 檢查點用 `hf_qat`、其 llama.cpp 對應版本用 `gguf_qat`。產生器會把這些變體暴露給切換器；不要在 JavaScript 中手動編碼尺寸替換。

選擇會在全站記住並反映在頁面 URL 中。如果記住的模型不可用，範例維持撰寫時的預設值，選擇器會說明原因；不會靜默替換。當同一頁面對某引擎有可切換範例時，用 `{ .model-fixed }` 標示刻意固定的區塊。

### 可匯出的筆記本儲存格

Markdown 仍是教學的唯一真實來源。只有該成為產生筆記本中可執行儲存格的圍欄，才加上 `.notebook-cell`：

````markdown
```python { .notebook-cell .model-swap }
model_id = "google/gemma-4-E2B-it"
```
````

未標記的圍欄仍是 Markdown 儲存格的一部分，`{ .output }` 區塊會被省略，因為筆記本執行時會重新產生它們。

當教學使用 `{ .model-swap }` 時，`scripts/notebooks.py` 會為頁面引擎與 `requires:` 模態限制所支援的每個候選模型，匯出獨立的模型專用筆記本，命名為 `<page>-gemma-4-<size>.ipynb`（例如 `thinking-gemma-4-e2b.ipynb`、`thinking-gemma-4-12b.ipynb`）。沒有 `.model-swap` 的教學只匯出單一的 `<page>.ipynb`。

在 frontmatter 加上預設模型筆記本的 Colab URL，即可在頁面資訊列顯示低調且無障礙的 Colab 圖示：

```yaml
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/tutorial-gemma-4-e2b.ipynb
```

當讀者在頁面資訊下拉選單中選擇其他模型時，模型切換器會動態更新 Colab 按鈕的目的地，以對應所選模型的筆記本。

將其他僅限網站的控制項包在 `<!-- notebook:skip-start -->` 與 `<!-- notebook:skip-end -->` 之間。在 `scripts/notebooks.py` 的 `SOURCES` 中註冊來源頁面，然後用 `uv run python scripts/notebooks.py` 產生所有已註冊的筆記本；驗證時使用 `--check`。不要直接編輯產生的 `.ipynb`。

## 引擎網格

快速入門上的「用你的方式執行」磁磚。是 `.card` 的緊湊版，適用於讀者要從十幾個工具中挑選其一、而不是細讀某個工具的情境。

由 `data/frameworks.yml` 產生——絕不手寫磁磚。引擎具備 `page`、`blurb` 與 `mark` 鍵後才會出現；頁面決定精選哪些 id 與排序，因為這是編輯決策，而不是關於框架的事實。

```jinja
{% raw %}{% set featured = ["ollama", "llamacpp"] %}
<ul class="engine-grid">
{% for id in featured %}{% for fw in frameworkdata.frameworks if fw.id == id %}
  <li><a class="engine" href="../{{ fw.page }}" title="{{ fw.blurb }}">
    <span class="engine-mark" aria-hidden="true">{{ fw.mark }}</span>
    <span class="engine-name">{{ fw.short_name | default(fw.name) }}</span>
  </a></li>
{% endfor %}{% endfor %}
</ul>{% endraw %}
```

`mark` 是兩到三個字元的字母組合，代替真正的廠商標誌。將 `.engine-mark` 的文字換成 `<img>` 不需要其他標記變更。在結尾磁磚加上 `.engine--all`，讓它讀起來像一個動作而不是另一個引擎。

加上 `.engine-grid--detailed` 可將每個 blurb 顯示在卡片上而不是懸停背後，並將名稱與 blurb 包在 `.engine-body` 中。在網格*就是*頁面、讀者要做決定而非辨認的情境使用它——Run 總覽就是如此；在網格只是跳過名稱清單的捷徑時用緊湊版，如快速入門。

沒有 `page` 的項目是託管端點：直接連到它的 `url`，並標記為外部連結。

### 釘選到模型世代的連結

有些外部 URL 帶有世代名稱——Hugging Face 收藏集、供應商的模型頁面——新一代發布後就會過時。標記它們以便日後找到：

```markdown
[Hugging Face](https://huggingface.co/collections/google/gemma-4){ data-model="gemma-4" }
```

在 `data/frameworks.yml` 中，對應的是項目上的 `pinned:` 鍵。無論哪種：

```bash
uv run python scripts/pinned_links.py --model gemma-4
```

列出所有這類連結。只標記真正版本釘選的 URL——專案首頁不是，標記它只會淹沒真正需要回顧的連結。

有兩件事只看名稱會誤導你。`paligemma-2` 與 `shieldgemma-2` 是 PaliGemma 2 與 ShieldGemma 2，不是 Gemma 2；標記為 `paligemma` 與 `shieldgemma`。而在 `scripts/notebooks.py` 註冊的頁面會匯出到 Jupyter，Jupyter 會把 `attr_list` 渲染成字面文字——那些不要標記。

## 模型家族樹

模型總覽上的血緣視覺圖。由 `data/models.yml` 的 `families` 產生：`generation` 決定主線脊柱的排序，`based_on` 將每個變體掛到其模型頁面所指名的基礎世代下。

互動是純 CSS——懸停或以鍵盤聚焦某個世代時，其他世代會變暗。底層標記是世代的有序清單，每個世代包含其後代的清單，因此即使完全沒有 CSS 也能正確閱讀。

```html
<div class="model-tree">
<ol class="tree-spine">
  <li class="tree-gen">
    <a class="tree-node tree-node--gen" href="…">
      <span class="tree-gen-name">Gemma 4</span>
      <span class="tree-gen-note">One line about the generation.</span>
    </a>
    <ul class="tree-branch">
      <li><a class="tree-node" href="…">
        <span class="tree-name">DiffusionGemma</span>
        <span class="tree-tagline">What it is for.</span>
      </a></li>
    </ul>
  </li>
</ol>
<ul class="tree-aside">…</ul>   <!-- families with no Gemma generation as a base -->
</div>
```

絕不加上模型自身頁面未提及的 `based_on`，也絕不為了排序樹而捏造發布日期——脊柱已經排好序了。頁面描述非 Gemma 架構的家族取 `based_on: null`，渲染在虛線下方。

!!! warning "清單容器需要更高的特異性"
    主題用 `[dir="ltr"] .md-typeset ol li`（0,2,2）與 `.md-typeset ul:not([hidden])`（0,2,1）設定清單樣式。單純的 `.md-typeset .my-grid` 只有 (0,2,0)，會靜默落敗，因此 `display: grid` 永遠不會生效，項目保持 `1.25em` 縮排。對任何基於清單的元件都要限定標籤——`.md-typeset ul.my-grid:not([hidden])`。

## 步驟

數字編號的逐步說明。建構在有序清單上，因此沒有 CSS 也能正確閱讀。

<ol class="steps" markdown>
<li markdown>
**安裝依賴**

步驟內可以放任何東西，包括程式碼區塊。
</li>
<li markdown>
**執行模型**

步驟之間的連接線會自動繪製。
</li>
</ol>

```html
<ol class="steps" markdown>
<li markdown>
**Step title**

Body content, including code blocks.
</li>
</ol>
```

`markdown` 屬性是必需的——它告訴 `md_in_html` 繼續解析標籤內的 Markdown。

!!! warning "內容分頁在步驟內無效"
    `pymdownx.tabbed` 是區塊級擴充，在 `md_in_html` 區塊內不會執行，因此 `<li markdown>` 內的 `=== "Label"` 會渲染成字面文字，出現在未分頁的程式碼區塊上方。在步驟內使用帶標籤的圍欄，或把分頁移到清單外。

## 圖片

帶圖說的有框圖表。

```html
<figure class="figure" markdown>
![Alt text](../assets/example.png)
<figcaption>What the diagram shows.</figcaption>
</figure>
```

需要依主題使用不同圖稿的圖表，標記這對圖並讓 CSS 選擇：

```html
![Alt text](../assets/diagram-light.png){ .only-light }
![Alt text](../assets/diagram-dark.png){ .only-dark }
```

## 影片

由 `scripts/importers/convert_notebook.py` 從獨立的 `[Video](youtube…)` 連結自動產生。全內容寬度、16:9、延遲載入、隱私模式主機。

```html
<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/VIDEO_ID"
          title="Video" loading="lazy" allowfullscreen></iframe>
</div>
```

## 模型卡標頭

將模型卡上的橫幅、資源連結與 License/Authors 行置中。也是由轉換器產生的——你不應該需要手寫。

```html
<div class="model-card-header" markdown>
![Banner](../assets/banner.png)

[Hugging Face](…) \| [GitHub](…)

**License** : [Apache 2.0](…)
</div>
```

## 提示框

標準的 Zensical 提示框，全站使用。轉換器從上游的 `> NOTE:` 引言與 `> [!NOTE]` 標註產生這些提示框。

!!! note
    中性的補充資訊。

!!! tip
    建議或捷徑。

!!! warning
    忽略會讓讀者付出時間或金錢代價的事。

## 動效

共用的元件轉場放在 `docs/stylesheets/motion.css`，包含全面性的 `prefers-reduced-motion` 退出機制——之後加的任何東西都會繼承它。動效是回饋（某物可互動、某物改變了），絕不是裝飾。

獨立的首頁動畫在開始前也會檢查這個偏好。

要檢查減少動效的路徑，請在作業系統的無障礙設定中啟用*減少動態效果*並重新載入；懸停效果應該變成即時生效。

## 樣式表配置

| 檔案 | 負責 |
|---|---|
| `gemma.css` | 設計權杖。唯一含原始十六進位色碼的檔案。 |
| `layout.css` | 頁面網格、側邊欄、字型比例 |
| `nav.css` | 頁首、分頁、側邊欄、目錄 |
| `content.css` | 表格、提示框、程式碼、內容分頁 |
| `components.css` | 卡片、徽章、圖片、步驟、嵌入 |
| `engine-grid.css` | 快速入門的引擎磁磚 |
| `model-tree.css` | 模型家族血緣樹 |
| `page-context.css` | 每個 H1 上方的適用性列 |
| `tokens.css` | 對話詞元徽章（`<start_of_turn>` 等） |
| `home.css` | 登陸頁：外框、背景、主視覺 |
| `motion.css` | 轉場與減少動效的退出機制 |

每個檔案都保持在 250 行以內，編輯前可以從頭讀到尾。為新的關注點新增樣式表是正確的做法；把某個檔案養成大雜燴則不是。
