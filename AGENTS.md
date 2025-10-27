# Repository Guidelines

## Project Structure & Module Organization

Authoring sources live in:

- `paper/` — manuscript (`paper.md`), wrapper (`index.qmd`), `bibliography.bib`, and `images/`
- `src/` — Python alt-text pipeline (`generate_alt_text.py`) writing timestamped outputs to `runs/`
- `abstract/`, `presentation/` — ancillary materials; published HTML in `_site/`

## Pipeline Workflow (src/)

1. `generate_alt_text.py` — fetch metadata, call configured models, and write timestamped runs under `runs/<timestamp>/` (raw JSON, prompt tables, manifests, cached images).
2. `clean_survey_data.py` — strip excluded submissions/emails from `data/raw/formspree_*_export.json`, saving `data/processed/processed_survey_submissions.json`.
3. `process_survey_rankings.py` — expand per-object rankings into `data/processed/survey_rankings.csv` with elapsed time per rater.
4. `process_best_answers.py` — derive consensus winners and export `data/processed/best_answers.csv` with exemplar texts.
5. `analyze_survey_time.py` — summarise completion durations and write `analysis/time_stats_by_{object,submission}.csv`.
6. `ranking_tests.py` — run Friedman/Wilcoxon tests, generate CSV summaries, and save plots (boxplots, heatmaps) in `analysis/`.
7. `viz_dataset.py` — build manuscript figures at `paper/images/fig_type_era_{full,subset}.png`.

See `README.md` for the mermaid workflow diagram and an artefact directory table.

## Paper Authoring Workflow

- Edit `paper/paper.md`; `paper/index.qmd` includes it for rendering.
- Preview: `quarto preview` (live reload).
- Citations: add BibTeX to `paper/bibliography.bib` and cite with `[@key]`. Example: “...explicit detail [@cecilia2023b]”.
- Figures: place under `paper/images/` and reference with labels. Example: `![Model comparison.](images/fig_models.png){#fig:models width=60%}`.
- Tables: use Markdown tables with captions. Example: `Table: Model costs {#tbl:models}` then reference as `[Table @tbl:models]`.
- From TODO to paper: items under `TODO.md` → “Paper/Literature and Context” and “Methodology Updates” should become citations and a maintained `@tbl:models` in `paper.md`. Keep model IDs and costs aligned with `src/generate_alt_text.py` and recent runs.

## Build, Test, and Development Commands

- `npm install && uv sync` — install Node and Python toolchains.
- `quarto preview` — preview the website; add `paper/` to focus on the manuscript.
- `uv run python src/generate_alt_text.py` — execute the batch job (requires `.env`). [DO NOT EXECUTE UNLESS EXPLICITLY AUTHORIZED]
- `npm run check` / `npm run format` — Prettier verify/fix.
- `uv run ruff check` / `uv run ruff fix` — Ruff verify/fix.

## Coding Style & Naming Conventions

- Prettier for Markdown/Quarto/JSON; Ruff for Python (PEP 8, typed).
- Use `snake_case` filenames; figures `fig_<short-key>.png`; labels `#fig:key`, `#tbl:key`.
- Commit notebooks with cleared outputs (`jupyter nbconvert --clear-output`).

## Commit & Pull Request Guidelines

- Use `npm run commit` (Conventional Commits); reference issues with `#NNN`.
- PRs: describe scope, note affected sections of `paper.md`, and attach before/after renders when changing figures/tables. Ensure `npm run check` passes.

## Reproducibility & Security

- Reference run artifacts via relative paths (e.g., `runs/2025.../manifest.json`); avoid committing large intermediates.
- Configure secrets in `.env` (e.g., `OPENROUTER_API_KEY`); never commit credentials.
