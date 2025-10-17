# Seeing History Unseen: CHR 2025 Conference Materials

This repository contains the abstract and presentation materials for the CHR 2025 conference paper "Seeing History Unseen: Evaluating Vision-Language Models for WCAG-Compliant Alt-Text in Digital Heritage Collections" by Moritz Mähr (University of Bern and Basel) and Moritz Twente (University of Basel).

[![GitHub issues](https://img.shields.io/github/issues/maehr/chr2025-seeing-history-unseen.svg)](https://github.com/maehr/chr2025-seeing-history-unseen/issues)
[![GitHub forks](https://img.shields.io/github/forks/maehr/chr2025-seeing-history-unseen.svg)](https://github.com/maehr/chr2025-seeing-history-unseen/network)
[![GitHub stars](https://img.shields.io/github/stars/maehr/chr2025-seeing-history-unseen.svg)](https://github.com/maehr/chr2025-seeing-history-unseen/stargazers)
[![Code license](https://img.shields.io/github/license/maehr/chr2025-seeing-history-unseen.svg)](https://github.com/maehr/chr2025-seeing-history-unseen/blob/main/LICENSE-AGPL.md)
[![Data license](https://img.shields.io/github/license/maehr/chr2025-seeing-history-unseen.svg)](https://github.com/maehr/chr2025-seeing-history-unseen/blob/main/LICENSE-CCBY.md)
[![DOI](https://zenodo.org/badge/614287827.svg)](https://zenodo.org/badge/latestdoi/ZENODO_RECORD)

## About This Repository

This repository hosts the conference materials for CHR 2025 (Conference on Computational Humanities Research) including:

- **Abstract**: A LaTeX document containing the complete abstract for our paper
- **Presentation Materials**: Slides and supporting materials for the conference presentation
- **Documentation**: Supporting documentation and setup instructions

## Research Overview

Our research explores the feasibility, accuracy, and ethics of using state-of-the-art vision-language models to generate WCAG-compliant alt-text for heterogeneous digital heritage collections. We combine computational experiments with qualitative evaluation to develop a framework for responsible AI-assisted accessibility in the humanities.

### Key Research Questions

1. **Feasibility**: Can current vision-language models produce useful, WCAG 2.2–compliant alt-text for complex historical images when provided with contextual metadata?
2. **Quality and Authenticity**: How do domain experts rate AI-generated image descriptions in terms of factual accuracy, completeness, and usefulness for understanding historical content?
3. **Ethics and Governance**: What are the ethical implications of using AI to generate alt-text in heritage collections, and what human oversight or policy safeguards are required for responsible use?

## Repository Structure

- `abstract/`: Contains the LaTeX source, class files, and bibliography for the conference abstract
- `presentation/`: Will contain presentation slides and supporting materials
- `documentation/`: Additional documentation and resources (including WCAG guidelines in `notes.md`)
- `src/`: Alt-text generation pipeline
  - `main.py`: Standalone script for generating WCAG-compliant alt-text using OpenRouter models
  - `playground.ipynb`: Interactive Jupyter notebook for experimenting with the pipeline
- `runs/`: Output directory for generated alt-text results, including raw API responses and CSV/JSONL/Parquet tables
- `data/`: Data directories for raw and cleaned datasets

## Installation

We recommend using **[GitHub Codespaces](https://github.com/features/codespaces)** for a reproducible setup.

## Getting Started

### For Most Users: Reproducible Setup with GitHub Codespaces

1. **[Use this template](https://github.com/new?template_name=open-research-data-template&template_owner=maehr)** for your project in a new repository on your GitHub account.

   <div align="center">
     <img src=".github/docs/assets/img_use.png" alt="Use the repository" style="width: 540px; margin: 1em 0;" />
   </div>

2. Click the green **`<> Code`** button at the top right of this repository.

3. Select the **“Codespaces”** tab and click **“Create codespace on `main`”**.
   GitHub will now build a container that includes:
   - ✅ Node.js (via `npm`)
   - ✅ Python with `uv`
   - ✅ R with `renv`
   - ✅ Quarto

   <div align="center">
     <img src=".github/docs/assets/img_codespace.png" alt="Create Codespace" style="width: 540px; margin: 1em 0;" />
   </div>

4. Once the Codespace is ready, open a terminal and preview the documentation:

   ```bash
   uv run quarto preview
   ```

   <div align="center">
     <img src=".github/docs/assets/img_terminal.png" alt="Terminal window showing command" style="width: 540px; margin: 1em 0;" />
   </div>

> **Note:** All dependencies (Node.js, Python, R, Quarto) are pre-installed in the Codespace.

<details>
<summary>👩‍💻 <strong>Advanced</strong> Local Installation</summary>

#### Prerequisites

- [Node.js](https://nodejs.org/en/download/)
- [R](https://cran.r-project.org/) and Rtools (on Windows)
- [uv (Python manager)](https://github.com/astral-sh/uv#installation)
- [Quarto](https://quarto.org/docs/get-started/)

> _Note: `uv` installs and manages the correct Python version automatically._

#### Local Setup Steps

```bash
# 1. Install Node.js dependencies
npm install
npm run prepare

# 2. Setup Python environment
uv sync

# 3. Setup R environment
Rscript -e 'install.packages("renv"); renv::restore()'

# 4. Preview documentation
uv run quarto preview
```

</details>

## Use

### Building the Abstract

To build the LaTeX abstract:

```bash
cd abstract
make paper
```

### Development Commands

Check that all files are properly formatted:

```bash
npm run check
```

Format all files:

```bash
npm run format
```

Run the wizard to write meaningful commit messages:

```bash
npm run commit
```

Generate a changelog:

```bash
npm run changelog
```

### Alt-Text Generation Pipeline

The repository includes a focused Python pipeline for generating WCAG-compliant alternative texts using OpenRouter-compatible vision-language models. The pipeline supports systematic evaluation of VLM performance on alt-text generation tasks for digital heritage collections.

#### Features

- **Automated alt-text generation** using multiple VLM models in parallel
- **WCAG 2.2 compliance** with structured prompts based on accessibility guidelines
- **Metadata integration** from remote sources with provenance tracking
- **Wide-format output** with model responses in CSV, JSONL, and Parquet formats
- **Raw API response storage** for reproducibility and analysis
- **Interactive playground** via Jupyter notebook for experimentation

#### Quick Start

1. Install Python dependencies with uv:

```bash
uv sync
```

2. Set up your OpenRouter API key in a `.env` file:

```bash
cp example.env .env
# Edit .env to add your OPENROUTER_API_KEY
```

3. Run the alt-text generation pipeline:

```bash
uv run python src/main.py
```

This will:

- Fetch metadata from the configured URL
- Generate alt-text for specified media IDs using all configured models
- Save results in `runs/YYYYmmdd_HHMMSS/` including:
  - `metadata.json`: Copy of fetched metadata for provenance
  - `alt_text_runs_*.csv`: Wide-format table with all model responses
  - `alt_text_runs_*.jsonl`: Same data in JSONL format
  - `alt_text_runs_*.parquet`: Same data in Parquet format (if available)
  - `raw/*.json`: Individual raw API responses from each model
  - `manifest.json`: Run metadata including configuration and file paths

4. Experiment interactively with the Jupyter notebook:

```bash
uv run jupyter notebook src/playground.ipynb
```

#### Configuration

Edit `src/main.py` to customize:

- `MODELS`: List of OpenRouter model identifiers to use
- `MEDIA_IDS`: List of media object IDs to process
- `METADATA_URL`: URL to fetch media metadata JSON

Current models configured:

- `google/gemini-2.5-flash-lite`
- `mistralai/pixtral-12b`
- `openai/gpt-4.1-nano`
- `allenai/molmo-7b-d`

For WCAG compliance guidelines and prompt engineering notes, see [documentation/notes.md](documentation/notes.md).

## Support

This project is maintained by [@maehr](https://github.com/maehr). Please understand that we can't provide individual support via email. We also believe that help is much more valuable when it's shared publicly, so more people can benefit from it.

| Type                                   | Platforms                                                                                |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| 🚨 **Bug Reports**                     | [GitHub Issue Tracker](https://github.com/maehr/chr2025-seeing-history-unseen/issues)    |
| 📚 **Docs Issue**                      | [GitHub Issue Tracker](https://github.com/maehr/chr2025-seeing-history-unseen/issues)    |
| 🎁 **Feature Requests**                | [GitHub Issue Tracker](https://github.com/maehr/chr2025-seeing-history-unseen/issues)    |
| 🛡 **Report a security vulnerability** | See [SECURITY.md](SECURITY.md)                                                           |
| 💬 **General Questions**               | [GitHub Discussions](https://github.com/maehr/chr2025-seeing-history-unseen/discussions) |

## Roadmap

- [ ] Complete the conference abstract and presentation preparation
- [ ] Create presentation slides for CHR 2025
- [ ] Finalize user study design and implementation
- [ ] Publish dataset and benchmark for future research

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Authors and credits

- **Moritz Mähr** - _University of Bern & Basel_ - [maehr](https://github.com/maehr)
- **Moritz Twente** - _University of Basel_ - [moritztwente](https://github.com/mtwente)

See also the list of [contributors](https://github.com/maehr/chr2025-seeing-history-unseen/graphs/contributors) who contributed to this project.

## License

The abstract, presentation materials, and documentation in this repository are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License - see the [LICENSE-CCBY](LICENSE-CCBY.md) file for details. By using these materials, you agree to give appropriate credit to the original author(s) and to indicate if any modifications have been made.

Any code in this repository is released under the GNU Affero General Public License v3.0 - see the [LICENSE-AGPL](LICENSE-AGPL.md) file for details.
