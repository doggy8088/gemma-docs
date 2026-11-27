---
name: gemma-docs
description: 撰寫、編輯與維護以 Zensical 建置的 Google DeepMind Gemma 文件網站。用於新增或更新文件頁面、修改模型/框架中繼資料、建立含可執行 notebook 的教學、使用設計 tokens 與元件設計樣式，或執行驗證檢查。
---

# Gemma 文件撰寫與維護

這個 skill 指導程式碼代理與貢獻者，維護與撰寫建置於 [Zensical](https://github.com/zensical/zensical) 的 Google DeepMind Gemma 文件儲存庫中的文件。

## 核心原則

1. **單一事實來源**：一個事實在 `data/` 下只屬於一個地方。頁面與指令碼從該來源讀取；絕不要在頁面中重複或硬編碼模型規格、框架清單或重複的 URL。
2. **匯入一次，然後手動維護**：來自上游來源的匯入頁面記錄於 `scripts/importers/sources.toml` 並凍結（`frozen = true`）。絕不要大量重新轉換已凍結的頁面。
3. **Markdown 優先的 Notebook**：Markdown 是教學的單一事實來源。Jupyter notebook（`docs/notebooks/*.ipynb`）是產生的建置產物，絕不直接撰寫。
4. **零警告 CI**：所有 lint 檢查（`scripts/lint_docs.py`）、測試（`tests/`）與建置（`scripts/build_docs.py`）都必須以零新增、零未解決的發現通過。

---

## 儲存庫結構

```text
├── data/                  # Single sources of truth (YAML)
│   ├── models.yml         # Model families, sizes, context, memory, lineage, checkpoint IDs
│   ├── frameworks.yml     # Supported engines, runtimes, tuning paths, and blurbs
│   ├── links.yml          # Repeated external links (appended to every page via snippets)
│   └── glossary.yml       # Acronym definitions (opt-in abbreviation tooltips)
├── docs/                  # Documentation source pages (Markdown)
│   ├── index.md           # Homepage
│   ├── get-started/       # Quickstart, capabilities, prompt formatting, QAT
│   ├── models/            # Family overviews, model cards, specialized variants
│   ├── deployment/        # Run guides (local, serving, cloud, frameworks)
│   ├── tuning/            # Fine-tuning guides (QLoRA, full, vision, distributed, etc.)
│   ├── learn/             # Task guides and visual walkthroughs
│   ├── resources/         # Tools, token calculator, community, skills
│   ├── contributing/      # Contributor guides: components.md and conventions.md
│   ├── stylesheets/       # CSS tokens and component styles
│   ├── javascripts/       # Interactive scripts (model switcher, generated model data)
│   └── notebooks/         # Exported .ipynb notebooks for Colab
├── overrides/             # HTML templates and partials
├── scripts/               # Build, generation, linting, and importer tooling
│   ├── build_docs.py      # Production build entry point (Zensical + agent routes)
│   ├── lint_docs.py       # Comprehensive documentation and invariant linter
│   ├── agent_docs.py      # Generates llms.txt and raw Markdown routes
│   ├── notebooks.py       # Exports registered Markdown tutorials to .ipynb
│   ├── generate_model_names.py # Compiles model data into JS and HTML partials
│   ├── generate_snippets.py    # Compiles links.md and glossary.md snippets
│   ├── vendor_icons.py    # Vendors SVGs referenced in pages
│   └── frontmatter_schema.py   # Shared schema and validation for YAML frontmatter
├── tests/                 # Unit tests (test_agent_docs.py, test_notebooks.py, etc.)
├── zensical.toml          # Site configuration, navigation tree, and theme settings
└── PLAN.md                # Living project roadmap and architectural guidelines
```

---

## 事實存放位置與更新方式

| 資料檔 | 管轄範圍 | 產生器／消費者 |
|---|---|---|
| `data/models.yml` | 模型家族、參數規模、上下文長度、記憶體需求、框架專屬檢查點 ID | `scripts/generate_model_names.py` -> `docs/javascripts/model-data.js` and `overrides/partials/model-names.html`; `scripts/notebooks.py` -> `docs/notebooks/*.ipynb` |
| `data/frameworks.yml` | 引擎中繼資料、簡介、圖示、URL、`tuning_paths` | 由頁面中的 Jinja 巨集（`{{ engines(...) }}`、`{{ paths(...) }}`）渲染 |
| `data/links.yml` | 在多個頁面使用的 URL、模型版本標籤 | `scripts/generate_snippets.py` -> `snippets/links.md`（自動附加到所有頁面） |
| `data/glossary.yml` | 縮寫與術語定義 | `scripts/generate_snippets.py` -> `snippets/glossary.md`（透過 `--8<-- "glossary.md"` 選擇性啟用） |

### 重新產生資料產物
編輯 `data/` 中的檔案後，請執行：
```bash
uv run --frozen python scripts/generate_model_names.py
uv run --frozen python scripts/generate_snippets.py
uv run --frozen python scripts/agent_docs.py
uv run --frozen python scripts/notebooks.py
```
如果產生的檔案與 `data/` 不同步，`scripts/lint_docs.py --check generated` 會失敗。

---

## 撰寫文件頁面

### Front matter 慣例
每個頁面都必須有符合 `scripts/frontmatter_schema.py` 的有效 YAML front matter：

```yaml
---
title: Page Title
description: Concise 1-2 sentence description for search and previews.
models: [gemma-4]          # Optional: list of relevant model family IDs
frameworks: [transformers] # Optional: list of relevant framework IDs
requires: audio            # Optional: required input modality (e.g. audio, vision)
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/tutorial-gemma-4-e2b.ipynb # Optional: Colab link
prerequisites:             # Optional: list of prerequisite topics
  - get-started/index.md
source_url: "https://..."  # Mandatory on imported pages; omitted on original pages
---
```
- **只寫作者鍵**：不要加入計算得出的鍵（例如 token 計數）。
- **詞彙表**：模型與框架識別碼必須存在於 `data/models.yml` 與 `data/frameworks.yml` 中。
- **模態閘控**：使用 `requires:` 將模型選擇與 notebook 匯出限制在支援該模態的模型（例如 `audio` 排除 26B/31B）。
- **Colab 連結**：對於可切換的教學，連結到預設模型 notebook（`-gemma-4-e2b.ipynb`）。當讀者選擇其他模型時，模型切換器會動態更新連結。

### 標題與錨點連結
- 使用標準的 Markdown 標題（`# H1`、`## 1. Step`、`### Substep`）。
- 引用編號章節時（例如 `## 1. Choose a framework`），記得 Python-Markdown 會產生包含編號的 slug（例如 `index.md#1-choose-a-framework`）。
- 以 `uv run --frozen python scripts/lint_docs.py --check internal` 驗證相對連結與錨點。

### 模型切換程式碼區塊
要在程式碼範例中啟用互動式模型切換：
```python { .model-swap }
# Use default model name in authoring; the JS switcher replaces it dynamically:
model_id = "google/gemma-4-E2B-it"
```

### 共用連結與引用
- 對於在多個頁面引用的 URL，將其加入 `data/links.yml` 並以 ID 引用：`[Link text][link-id]`。
- 對於鎖定到特定模型的外部連結：
  ```markdown
  [Kaggle](https://www.kaggle.com/models/google/gemma-4){ data-model="gemma-4" }
  ```

---

## Markdown 優先的 Notebook 工作流程

要建立或編輯可匯出為 Jupyter notebook 的教學：

1. **可執行程式碼儲存格**：在應於 Jupyter 中執行的圍欄加上 `{ .notebook-cell }`：
   ````markdown
   ```python { .notebook-cell }
   import torch
   from transformers import AutoModelForCausalLM
   ```
   ````
2. **可切換模型儲存格**：搭配 `{ .model-swap }` 以匯出多模型 notebook：
   ````markdown
   ```python { .notebook-cell .model-swap }
   model_id = "google/gemma-4-E2B-it"
   ```
   ````
   匯出期間，`scripts/notebooks.py` 會偵測標記區塊中的可切換引擎，以及符合任何 `requires:` front matter 的候選模型。對於可切換的教學，它會產生名為 `<stem>-gemma-4-<size>.ipynb` 的模型專屬 notebook（例如 `thinking-gemma-4-e2b.ipynb`、`thinking-gemma-4-12b.ipynb`）。不可切換的教學只產生單一 `<stem>.ipynb`。
3. **僅顯示的程式碼**：一般圍欄（沒有 `.notebook-cell`）保留在 Markdown 中，並在 notebook 中成為文字儲存格。
4. **輸出**：標記 `{ .output }` 的圍欄在 notebook 匯出時會被省略。
5. **僅網站區塊**：將僅限網頁的元素（如互動小工具或卡片網格）包在跳過註解中：
   ```markdown
   <!-- notebook:skip-start -->
   {{ engines(["ollama", "vllm"]) }}
   <!-- notebook:skip-end -->
   ```
6. **註冊**：在 `scripts/notebooks.py` 的 `SOURCES` 中註冊新的 notebook 來源教學。
7. **Colab 連結**：對於可切換的教學，在 front matter 中加入預設模型 notebook 的 Colab URL（`colab: .../<stem>-gemma-4-e2b.ipynb`）。當讀者選擇不同模型時，頁面情境切換器會動態更新連結。
8. **匯出與驗證**：執行 `uv run --frozen python scripts/notebooks.py` 以重新產生 `docs/notebooks/*.ipynb`，並以 `uv run --frozen python scripts/notebooks.py --check` 驗證。

---

## 設計系統與樣式規則

- **Token**：所有顏色、間距與圓角都位於 `docs/stylesheets/gemma.css`。
  - 品牌藍僅由 `--gemma-blue-seed: #0B57D0` 控制。
  - 以 `uv run --frozen python scripts/check_contrast.py --seed "#0B57D0"` 測試新的 seed。
  - **不允許**在 `gemma.css` 以外的任何樣式表中使用原始 hex 色彩（由 `scripts/lint_docs.py --check css` 強制執行）。
- **元件**：使用 `docs/contributing/components.md` 中的現有元件（卡片、引擎網格、步驟清單、分頁、admonition）。

---

## 驗證與 CI 工作流程

提交變更前務必執行完整的測試與 lint 套件：

```bash
# 1. Run unit tests
uv run --frozen python -m unittest discover -s tests -p "test_*.py"

# 2. Run documentation linter across all 15 checks
uv run --frozen python scripts/lint_docs.py --baseline

# 3. Build documentation site and generate raw agent routes / llms.txt
uv run --frozen python scripts/build_docs.py
```

### 個別 Linter 檢查
開發期間執行特定檢查：
```bash
uv run --frozen python scripts/lint_docs.py --check internal,links,frontmatter,orphans
```
可用的檢查：`links`、`references`、`icons`、`images`、`internal`、`stubs`、`frontmatter`、`orphans`、`fences`、`model-swap`、`admonitions`、`size`、`modeldata`、`css`、`generated`。
