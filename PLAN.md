# Gemma 文件路線圖

這個儲存庫以 Zensical 移植並重新構想了 Gemma 文件。目前目標是
一個打磨完善的概念驗證（proof of concept）：可安全合併，且易於持續維護。

這個檔案是活的路線圖，而非變更日誌。已完成的實作細節屬於 Git
歷史；長期有效的文件撰寫指示屬於 `README.md`、`docs/contributing/` 或
`skills/gemma-docs/SKILL.md`。

## 目前狀態

- Zensical 設計系統、導覽、頁面情境標頭、全域 Gemma 4 選擇器與 MathJax
  公式渲染都已運作。
- 面向代理的原始 Markdown 路由與 `llms.txt` 由建置產生（`docs/llms.txt`
  已提交並由 `lint_docs.py` 驗證）。
- 23 篇 Markdown 優先的教學匯出 66 個 Jupyter notebook（`docs/notebooks/*.ipynb`），
  含 Apache 2.0 授權標頭，包括 `.model-swap` 頁面的模型專屬變體，
  以及瀏覽器中的動態 Colab 連結切換。
- 兩個互動示範（FunctionGemma 物理遊樂場 HF Space 與 AIventure 網頁遊戲）、
  以任務為先的 Learn 中心、視覺指南（`tutorials/visual-guides/`），以及 6 篇的
  Gemma Explained 系列（`tutorials/gemma-explained/`），展示了更豐富的文件體驗。
- `scripts/importers/sources.toml` 中的全部 65 個匯入頁面皆已凍結（`frozen = true`），
  並在總計 97 個文件頁面中手動維護。
- lint 基準線（`.lintbaseline.json`）為空（全部 15 項檢查皆為
  `0 new, 0 known`）。
- 儲存庫治理（`CONTRIBUTING.md`、`CODEOWNERS`、Apache 2.0 `LICENSE`）、
  貢獻者指南（`docs/contributing/`），以及文件撰寫 skill
  （`skills/gemma-docs/SKILL.md`）皆已完成。

延期中的示範與互動工具不是 POC 合併的阻擋項。

## 正式上線切換與剩餘任務

保持這份清單精簡。這些任務實質影響正式上線切換或儲存庫維護；
更廣的產品想法屬於延期路線圖。

### 分析切換

- [ ] 在 GitHub Actions 中設定正式環境的 `GA_MEASUREMENT_ID` 儲存庫變數或 secret。
- [ ] 最終的量測 ID 就緒後，將 `tests/skip_test_analytics.py` 重新命名為
      `tests/test_analytics.py`，以在 CI 中啟用 GA4 與 cookie 同意驗證。

### 產生的 notebook（可選的 CI 發布）

目前 `docs/notebooks/*.ipynb` 已提交到 `main`，並由
`scripts/lint_docs.py --check generated` 嚴格驗證，因此 Colab 連結
（`blob/main/docs/notebooks/...`）可直接適用於所有模型規模。
如果未來想從 `main` 移除已提交的 `.ipynb` 檔案：

- [ ] 從 CI 發布 notebook 到專用的 `generated-notebooks` 分支。
- [ ] 將每個 Colab URL（以及 `model-switcher.js` URL 重寫器）指向該分支。
- [ ] 只有在發布的 URL 正常運作後，才從 `main` 移除已提交的 `.ipynb` 檔案。
- [ ] 僅給予建置後發布工作在 `main` 建置成功時的 `contents: write` 權限，
      使用 SHA 鎖定的 action 並保持 zizmor 檢查乾淨。

### 舊版上游頁面

仍有五個舊版上游頁面尚未移植（`/meme-generator` 已在 `link_map.toml` 中移除連結）：

- `/core/gemma_library`
- `/core/pytorch_gemma`
- `/core/deploy_to_cloud_run_from_ai_studio`
- `/gemma_chat`
- `/meme-generator`

合併後再決定是否永久退役或移植這些舊版指南。

## 長期有效的文件撰寫規則

### 匯入一次，然後手動維護

匯入的內容已在上游驗證。移植只改變格式，不改變文字或程式碼。

- 轉換期間不要改寫、重排或默默修正匯入的文字。
- 沒有明確的技術理由與審查，不要更動匯入的程式碼。
- 每個匯入的頁面都在 front matter 中記錄 `source_url`。
- 每個現有的 manifest 條目都已凍結。絕不要大量重新轉換已凍結的頁面。
- 新頁面以未凍結狀態加入 `scripts/importers/sources.toml`，匯入並審查一次，
  然後在開始文件專屬編輯前標記為 `frozen = true`。
- 轉換器的改進只影響未來的匯入；現有頁面刻意以手動方式修復。

官方文件匯入管線為：

```text
scripts/importers/port_docs.py
  -> scripts/importers/vendor_images.py  # only if the source needs local assets
  -> scripts/importers/rewrite_links.py
  -> scripts/lint_docs.py --baseline
```

`port_docs.py` 在寫入前執行 Markdown 轉換器與保真度檢查。視覺指南改用
`scripts/importers/import_newsletter.py`。暫時的示範/供應商圖片不值得重工，
因為它們將被替換為原始 PNG。

### Markdown 優先的 notebook

- 只在應於 Jupyter 中執行的圍欄加上 `.notebook-cell`。
- 一般文字成為 Markdown 儲存格。
- 標記 `.output` 的圍欄會被省略，並在 notebook 執行時重建。
- 僅限網站的區域使用 `docs/contributing/components.md` 中記載的
  notebook 跳過標記。
- 帶有 `.model-swap` 程式碼區塊的教學會為符合頁面引擎與 `requires:`
  模態限制的所有候選模型匯出各自的模型專屬 notebook
  （`<stem>-gemma-4-<size>.ipynb`）。不可切換的教學只匯出單一
  `<stem>.ipynb`。
- front matter 的 `colab:` 指定預設模型 notebook（`-gemma-4-e2b.ipynb`）；
  當讀者選擇其他模型時，`docs/javascripts/model-switcher.js`
  會動態重寫 Colab 連結。
- 匯出的 notebook 會自動包含 Google LLC Apache 2.0 授權標頭儲存格。
- 不要同時維護 Markdown 與 notebook JSON。

### 資料與渲染

- `data/models.yml`、`data/frameworks.yml`、`data/links.yml` 與 `data/glossary.yml`
  是模型中繼資料、引擎資訊、共用連結與詞彙表提示的單一事實來源。
  不要在頁面或 JavaScript 中手動維護副本。
- 引用的 Zensical 圖示透過 `scripts/vendor_icons.py` 供應到 `overrides/.icons/`，
  讓獨立的 CI lint 工作不需安裝 Zensical 也能執行。
- front matter 的鍵可能會遮蔽樣板資料。基於這個原因，
  樣板巨集使用 `modeldata.*`。
- 在瀏覽器中量測渲染版面。僅檢查原始碼不足以進行視覺工作。
- 保持 Zensical 版本鎖定。套件更新不得默默重新設計文件的樣式。
- 新的工作流程變更必須使用最小權限、SHA 鎖定的 action，並通過 zizmor。

## 維護中的工具

目錄結構將正式環境基礎設施、可選的匯入工具與測試分開。

### 正式的建置與驗證：`scripts/`

| 檔案 | 職責 |
|---|---|
| `build_docs.py` | 正式環境進入點：產生器、GA4 設定注入、Zensical 建置與原始 Markdown 發布。 |
| `agent_docs.py` | 建置 `llms.txt` 與原始 `.md` 路由。 |
| `lint_docs.py` | CI 守門員：15 項文件與儲存庫不變量檢查。 |
| `frontmatter_schema.py` | 共用的 front matter 結構描述與受控詞彙。 |
| `generate_model_names.py` | 為樣板與瀏覽器選擇器編譯模型/框架資料。 |
| `generate_snippets.py` | 將 `data/links.yml` 與 `data/glossary.yml` 渲染為 `snippets/links.md` 與 `snippets/glossary.md`。 |
| `notebooks.py` | 將註冊的 Markdown 教學匯出為單一或多模型的 `.ipynb` 檔案。 |
| `vendor_icons.py` | 將引用的 Zensical SVG 圖示供應到 `overrides/.icons/`。 |
| `pinned_links.py` | 掃描並報告鎖定到特定模型世代（`data-model`）的外部 URL。 |
| `check_contrast.py` | 驗證 CSS 品牌色彩階（`--gemma-blue-seed`）的 WCAG 對比度。 |

### 可重用的擷取工具：`scripts/importers/`

這些工具仍受支援，因為預期還會有更多官方 Markdown 與 notebook 內容。

| 檔案 | 職責 |
|---|---|
| `convert_notebook.py` | 將官方 `.md.txt`/notebook 匯出轉換為 Zensical Markdown。 |
| `port_docs.py` | 由 manifest 驅動、具凍結感知的匯入編排。 |
| `check_fidelity.py` | 偵測新轉換中遺失的文字。 |
| `rewrite_links.py`, `link_map.toml` | 在匯入後轉換上游連結。 |
| `vendor_images.py` | 新匯入的可選遠端資產步驟。 |
| `import_newsletter.py` | 匯入額外的第一方視覺指南；若該路線圖被捨棄則移除。 |
| `sources.toml` | 匯入來源與凍結頁面的安全邊界。 |

### 測試：`tests/`

回歸測試與轉換器測試夾具屬於 `tests/`，而不是放在可執行的維護指令旁。
CI 從這個目錄探索 `test_*.py`（`test_agent_docs.py`、
`test_convert_notebook.py`、`test_notebooks.py`；`skip_test_analytics.py`
等待正式環境的 `GA_MEASUREMENT_ID`）。

已從維護範圍中移除：

- `backfill_frontmatter.py`：已完成的一次性遷移。
- `PLAN-ARCHIVE.md`：重複了 Git 歷史，且讓支援的工作流程更難找到。

## 驗證

在交接或合併前執行：

```bash
uv run --frozen python -m unittest discover -s tests -p "test_*.py"
uv run --frozen python scripts/lint_docs.py --baseline
uv run --frozen python scripts/build_docs.py
```

對於新的官方文件匯入，還要以 `--dry-run` 模式執行相關的匯入器，
並在寫入檔案前審查其保真度報告。

## 延期路線圖

### 內容與 Learn

- 繼續 Markdown 優先的 notebook 擷取/匯出工作流程，但匯入後保持單向。
- 只有在原始的、正式品質的 PNG 就緒時，才新增視覺指南資產。
- 只有在新內容足以支撐另一個路由時，才擴展簡單的以任務為先的 Learn 頁面。

### 示範與嵌入

- 將 FunctionGemma HF Space 與 AIventure 示範保留為最初的互動範例。
- 將額外的 spaces、媒體嵌入與互動示範延期，直到核心內容確定。

### 互動工具

- 由 `data/models.yml` 驅動的 VRAM 與上下文計算器，含模型、量化、上下文、
  批次大小、KV-cache dtype 與 GPU 預設；在 URL 中編碼狀態。
- 瀏覽器內 E2B q4 遊樂場，透過 WebGPU/transformers.js，需明確同意大型下載。
- 「向 Gemma 詢問 Gemma」文件助手。
- 互動式 Gemma 4 架構與 MTP 說明。
- DiffusionGemma 去噪動畫。
- 社群吞吐量排行榜，只有在其維護成本合理時才做。
- 自行託管成本計算器。
