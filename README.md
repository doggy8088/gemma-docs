# Gemma documentation

The Google DeepMind Gemma documentation, ported to
[Zensical](https://github.com/zensical/zensical).

## Local development

The project uses [uv](https://docs.astral.sh/uv/) and pins the documentation
toolchain in `uv.lock`.

```bash
uv sync --frozen
uv run --frozen zensical serve
```

The preview is available at <http://localhost:8000/>.

## Verification

Run these checks before committing documentation or styling changes:

```bash
uv run --frozen python scripts/test_convert_notebook.py
uv run --frozen python scripts/lint_docs.py --baseline
uv run --frozen zensical build
```

See `PLAN.md` for the migration roadmap and `docs/contributing/components.md`
for the CSS tokens and reusable components.
