# Gemma 文件

Google DeepMind Gemma 文件，已移植到
[Zensical](https://github.com/zensical/zensical)。

## 本機開發

專案使用 [uv](https://docs.astral.sh/uv/)，並將文件工具鏈鎖定在
`uv.lock` 中。

```bash
uv sync --frozen
uv run --frozen python scripts/agent_docs.py
uv run --frozen zensical serve
```

預覽可在 <http://localhost:8000/> 開啟。
由於本機 MIME 映射的關係，Zensical 的預覽伺服器可能會下載原始的
`.md` 路由；GitHub Pages 則會以其 Markdown 內容類型提供這些靜態檔案。

## 驗證

在提交文件或樣式變更前，請執行以下檢查：

```bash
uv run --frozen python -m unittest discover -s tests -p "test_*.py"
uv run --frozen python scripts/lint_docs.py --baseline
uv run --frozen python scripts/build_docs.py
```

`build_docs.py` 會執行一般的 Zensical 建置，然後發布原始 Markdown 路由，
例如 `/get-started/index.md`。提交到版本庫的 `docs/llms.txt`
是從頁面 front matter 重新產生，並由 `generated` lint 檢查驗證。

遷移路線圖請參閱 `PLAN.md`；CSS tokens 與可重用元件請參閱
`docs/contributing/components.md`。
