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

- [ ] Incorporate summary of model cards:
  - [Mistral: Pixtral 12B](https://openrouter.ai/mistralai/pixtral-12b)
  - [Google: Gemini 2.5 Flash Lite](https://openrouter.ai/google/gemini-2.5-flash-lite)
  - [Meta: Llama 4 Maverick](https://openrouter.ai/meta-llama/llama-4-maverick)
  - [OpenAI: GPT-4o-mini](https://openrouter.ai/openai/gpt-4o-mini)
- [ ] Add links to open-weights models on Hugging Face for pixtral and llama-maverick
- [ ] Cite and integrate:
  - [https://arxiv.org/html/2403.09193v2](https://arxiv.org/html/2403.09193v2)

Question: Do LLM–vision fusion and prompts change what visual cues VLMs rely on?
arXiv

Method: Measure texture-vs-shape bias on cue-conflict images for VQA and captioning; run mechanistic checks across vision encoder vs LLM fusion; test prompt “steering” toward shape/texture and low/high spatial frequencies.
arXiv

Core findings:

VLMs inherit encoder biases only partly and are more shape-biased than standard vision-only models, though still below human levels.
arXiv

Multi-modal fusion suppresses or amplifies cues; the LLM can shift which visual evidence is used.
arXiv

Prompting can steer cue use without retraining. Steering toward texture is often easier than toward shape; achievable shape bias ranged roughly 49–72% with little accuracy loss.
arXiv

- [https://arxiv.org/html/2507.11543v1](https://arxiv.org/html/2507.11543v1)

The paper is a structured literature review on generative AI in computer science education. It synthesizes 52 studies (2019–2024) around three lenses—accuracy (hallucinations, bias, error propagation, metrics), authenticity (authorship, integrity, sociotechnical context), and assessment (AI-assisted grading, fairness, limits of current metrics). It argues for human-in-the-loop, hybrid assessment models, AI literacy, and bias-mitigation frameworks; it highlights gaps such as longitudinal effects and reliable accuracy measures beyond narrow correctness.

- [https://arxiv.org/html/2409.03054v1](https://arxiv.org/html/2409.03054v1)

The paper designs and tests a pipeline that injects webpage context into GPT-4V image descriptions and shows, with a 12-participant BLV user study, that context-aware descriptions are preferred and rated higher on quality, imaginability, relevance, and plausibility than context-free baselines. It also reports a technical audit for hallucinations, subjectivity, and irrelevance, and surfaces risks around trust, privacy, and person identification. This directly supports your claim that VLMs can act as accessibility assistants and offers concrete design, evaluation, and governance patterns you can adapt to GLAM workflows.

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
