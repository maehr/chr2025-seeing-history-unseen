# CHR 2025 Project TODO

This file tracks setup, research, and publication tasks for the **CHR 2025: Seeing History Unseen** repository.

Completed items moved to `ARCHIVE.md`.

## Repository Setup /repo/

### Pending 🔄

- [ ] Connect repository to **Zenodo**
- [ ] Add Zenodo DOI badge and DOI to `README.md`
- [ ] (Optional) Add `.zenodo.json` metadata file
- [ ] (Optional) Generate and commit `CHANGELOG.md`
- [ ] (Optional) Enable and publish **GitHub Pages** with Quarto
- [ ] Test all build commands (`npm`, `make`)
- [ ] Update from latest base template

## Paper /paper/

### Literature and Context

- [ ] Incorporate model overview: [https://github.com/zli12321/Vision-Language-Models-Overview](https://github.com/zli12321/Vision-Language-Models-Overview)
- [ ] Add links to open-weights models on Hugging Face
- [ ] Cite and integrate:
  - [https://arxiv.org/abs/2403.09193](https://arxiv.org/abs/2403.09193)
  - [https://arxiv.org/html/2507.11543v1](https://arxiv.org/html/2507.11543v1)
  - [https://arxiv.org/html/2501.00113v1](https://arxiv.org/html/2501.00113v1)
  - [https://arxiv.org/html/2409.03054v1](https://arxiv.org/html/2409.03054v1)
  - [https://tealab.sites.northeastern.edu/generative-ai-and-accessibility](https://tealab.sites.northeastern.edu/generative-ai-and-accessibility)
  - [https://www.cni.org/topics/digital-libraries/beyond-this-image-may-contain-using-vision-language-models-to-improve-accessibility-for-digital-image-collections](https://www.cni.org/topics/digital-libraries/beyond-this-image-may-contain-using-vision-language-models-to-improve-accessibility-for-digital-image-collections)
- [ ] Ensure consistent vocabulary for disability justice framing
      Reference:
  - [NYC Disability-Inclusive Terminology Guide (2021)](https://www.nyc.gov/assets/mopd/downloads/pdf/Disability-Inclusive-Terminology-Guide-Dec-2021.pdf)
  - [UN Geneva Disability-Inclusive Language Guidelines](https://www.ungeneva.org/sites/default/files/2021-01/Disability-Inclusive-Language-Guidelines.pdf)
  - [Stanford Disability Language Guide](https://disability.stanford.edu/sites/g/files/sbiybj26391/files/media/file/disability-language-guide-stanford_1.pdf)

### Methodology Updates

- [ ] Document model selection and changes:
  - Initial use of `allenai/molmo-7b-d` abandoned due to inconsistency
  - Switched from `openai/gpt-4.1-nano` to `openai/gpt-4o-mini` for better performance
  - Evaluated models with comparable costs:

| Model Name & ID                                                                                                   | Input ($/1M) | Output ($/1M) | Context (tokens) |
| ----------------------------------------------------------------------------------------------------------------- | ------------ | ------------- | ---------------- |
| [Mistral: Pixtral 12B](https://openrouter.ai/mistralai/pixtral-12b)`mistralai/pixtral-12b`                        | 0.10         | 0.10          | 32 768           |
| [Google: Gemini 2.5 Flash Lite](https://openrouter.ai/google/gemini-2.5-flash-lite)`google/gemini-2.5-flash-lite` | 0.10         | 0.40          | 1 048 576        |
| [Meta: Llama 4 Maverick](https://openrouter.ai/meta-llama/llama-4-maverick)`meta-llama/llama-4-maverick`          | 0.15         | 0.60          | 1 048 576        |
| [OpenAI: GPT-4o-mini](https://openrouter.ai/openai/gpt-4o-mini)`openai/gpt-4o-mini`                               | 0.15         | 0.60          | 128 000          |

## Analysis /analysis/

- [ ] To be completed during paper writing

## Runs /runs/

- [ ] Final run with full dataset

## Survey /survey/

- [ ] `survey/questions.csv` — update after final run
- [ ] `survey/results.csv` — update after expert survey

## Source Code /src/

- [ ] `analysis.py` — write after final run and survey
- [ ] `playground.ipynb` — update after paper finalization

## Presentation /presentations/

- [ ] Align slides with final paper
- [ ] Add examples, visualizations, and demonstrations
- [ ] Final review and rehearsal

## Documentation /documentation/

- [ ] Update after final paper version and publication

## Testing /test/

- [ ] TBD after final version

## Research Tasks /research/

- [ ] Complete user study with blind/low-vision participants
- [ ] Finalize evaluation of AI models
- [ ] Release benchmark dataset
- [ ] Draft practical guidelines for heritage institutions

## Agents.md

- [ ] Update to support paper writing and model comparison

## Notes

- Keep this document current
- Use `[x]`, `[ ]`, and status emojis consistently
- Archive historical TODOs after publication in `ARCHIVE.md`
