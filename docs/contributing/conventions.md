---
title: 慣例
description: 維持這份文件一致性的約定：事實的單一來源放在哪裡、哪些檔案是自動產生的，以及 lint 會強制執行哪些規則。
---

# 慣例

[元件](components.md) 說明頁面長什麼樣子。本頁說明它們如何保持正確：事實被允許放在哪裡、什麼是從它產生的，以及這些規則中哪些由機器而非審查者強制執行。

主軸很簡單。**一個事實只屬於一個地方。**所有重複它的東西都從那個地方產生，兩者漂移時 lint 就會失敗。

## 事實放在哪裡

`data/` 下的四個 YAML 檔案擁有所有原本會在各頁重複的內容。

| 檔案 | 擁有 | 讀取者 |
|---|---|---|
| `models.yml` | 每個家族、其尺寸、血緣、上下文視窗、記憶體，以及各引擎的檢查點 id | 模型表格、家族樹、模型選擇器、frontmatter 驗證 |
| `frameworks.yml` | 每個引擎：顯示名稱、首頁、我們的頁面、簡介、標誌——外加 `tuning_paths`，即微調總覽上顯示的路線 | Run 與 Fine-tune 上的引擎網格、頁面資訊列、frontmatter 驗證 |
| `links.yml` | 用於多個頁面的任何 URL | 每頁的參照連結、釘選連結掃描 |
| `glossary.yml` | 縮寫定義 | 選擇加入頁面上的縮寫工具提示 |

**在頁面上手動編輯模型表格、引擎清單或重複的 URL 是錯誤**，不是捷徑。改資料檔；頁面會跟著變。

兩條規則讓資料保持誠實：

- **絕不為了填補空缺而捏造值。**KerasHub 沒有發布 12B 預設，LiteRT-LM 只有 E2B 與 E4B。這些缺席以「省略」來記錄，選擇器在那些頁面就只提供較少的尺寸。捏造的 id 會產生 404 的指令。
- **推斷的東西都要註明出處。**`models.yml` 中的每個 `based_on` 都帶有註解，指名該模型自身頁面上的主張。頁面未提及 Gemma 基礎的家族就沒有 `based_on`。

## 什麼是產生的

這份清單中的東西絕不該手動編輯。每個都有產生器，當已提交的檔案與其來源不符時，`lint_docs.py --check generated` 就會失敗。

| 輸出 | 產生器 | 來源 |
|---|---|---|
| `overrides/partials/model-names.html` | `generate_model_names.py` | `models.yml` |
| `docs/javascripts/model-data.js` | `generate_model_names.py` | `models.yml`、`frameworks.yml` |
| `snippets/links.md`、`snippets/glossary.md` | `generate_snippets.py` | `links.yml`、`glossary.yml` |
| `docs/llms.txt` 與原始 `.md` 路由 | `agent_docs.py` | 頁面 frontmatter 與導覽 |
| `docs/notebooks/*.ipynb` | `notebooks.py` | 已註冊的教學頁面（可切換頁面的模型適配變體） |
| `overrides/.icons/**.svg` | `vendor_icons.py` | 頁面實際參照的圖示 |

圖示是唯一由 `--check icons` 而非 `--check generated` 檢查的項目，因為它們是從 Zensical 複製出來的，而不是從我們自己的資料渲染的。它們必須提交：lint 工作流程只安裝 PyYAML 與 Jinja2，所以只存在於 Zensical 套件內的圖示在那裡會遺失，連帶拖垮整個測試套件。參照新圖示在撰寫時立即生效，但在圖示被收錄（vendored）之前 lint 會失敗。

## 匯入的頁面已凍結

多數頁面從上游文件移植而來。移植只改格式，絕不改文字或程式碼。

- 每個匯入的頁面都在 frontmatter 記錄 `source_url`。
- `scripts/importers/sources.toml` 中的每個項目都是 `frozen = true`。凍結的頁面絕不會被大量重新轉換。
- 新頁面以未凍結狀態加入清單、匯入、審查**一次**，然後在任何文件專屬編輯開始前凍結。
- 轉換器的改進只影響未來的匯入。既有頁面用手修。

頁面退役或合併時，清單會保留註解說明它去了哪裡，讓上游來源仍然找得到。

## Frontmatter

`scripts/frontmatter_schema.py` 是唯一的真實來源，由 lint 強制執行。它劃分的區別很重要：

- **Authored** 鍵是只有人知道的事實——`title`、`description`、`models`、`frameworks`、`prerequisites`、`requires`（模態限制）、`colab`（筆記本直接連結）。
- **Computed** 鍵可從檔案衍生。手寫一個是*錯誤*，因為手寫的詞元計數在頁面被編輯的那一刻就錯了。

詞彙來自資料檔而非寫死的清單，因此在 `models.yml` 新增尺寸會自動擴充 `models:` 接受的值。

## 帶有模型名稱的連結

有些外部 URL 內嵌世代名稱，新一代發布後就會過時。它們會被標記以便日後找到，用情境允許的任何形式：

```yaml
# data/links.yml — preferred, declares it once beside the URL
hf-gemma4:
  url: https://huggingface.co/collections/google/gemma-4
  model: gemma-4
```

```markdown
[Kaggle](https://www.kaggle.com/models/google/gemma-4){ data-model="gemma-4" }
```

在卡片網格內（原始 HTML），`attr_list` 不適用——改用純屬性形式把 `data-model` 放在錨點上。

```bash
uv run python scripts/pinned_links.py --model gemma-4
```

列出所有這類連結。只標記真正版本釘選的 URL：專案首頁不是，標記它只會淹沒真正需要回顧的連結。

## Markdown 是來源，筆記本是輸出

`SOURCES` 中的十個教學頁面匯出到 Jupyter。之後的每次編輯都以 Markdown 為準；`.ipynb` 檔案是建置產物。

只有標記 `.notebook-cell` 的圍欄會成為可執行儲存格。當教學包含 `{ .model-swap }` 程式碼區塊時，`notebooks.py` 會為所有符合教學引擎與 `requires:` 模態限制的候選模型，匯出獨立的模型專用筆記本，命名為 `<stem>-gemma-4-<size>.ipynb`（例如 `thinking-gemma-4-e2b.ipynb`）。不可切換的教學只匯出單一的 `<stem>.ipynb`。

對於可切換的教學，frontmatter 在 `colab:` 指定預設模型筆記本，頁面資訊切換器會在讀者選擇不同模型時動態更新連結。

頁面的其餘部分都是網站外框，`notebooks.py` 會剝除 Jupyter 無法渲染的東西——`attr_list` 註記、卡片網格、`{ .output }` 區塊、片段引入、縮寫定義，以及跳過區塊（`<!-- notebook:skip-start -->` ... `<!-- notebook:skip-end -->`）。**加入撰寫詞彙的任何東西都需要對應的剝除**，否則它會以字面文字落入筆記本。

## 驗證

三個指令，CI 會全部執行：

```bash
uv run --frozen python -m unittest discover -s tests -p "test_*.py"
uv run --frozen python scripts/lint_docs.py --baseline
uv run --frozen python scripts/build_docs.py
```

`lint_docs.py --list` 顯示所有檢查。`.lintbaseline.json` 中的基準是**空的**，而且應該保持如此：它的存在是為了讓已知問題可以暫時帶著走，而不是藏起來。

另外兩個工具是回答問題用的，不做把關：

```bash
uv run python scripts/pinned_links.py      # what breaks when a new generation ships
uv run python scripts/check_contrast.py    # whether a new brand colour is legible
```

視覺工作在瀏覽器中量測過之前都不算完成。只看原始碼看不出卡片的懸停狀態會把深色圖示放在深色填色上，也看不出標題的特異性會輸給主題。
