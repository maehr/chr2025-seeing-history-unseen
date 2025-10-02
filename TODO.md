# CHR 2025 Project TODO List

This file tracks the setup and development tasks for the CHR 2025 conference materials repository.

## Repository Setup Tasks

### Completed ✅

- [x] **Update Project Details**: Replaced placeholders in `package.json`, `_quarto.yml`, and `index.qmd` with correct repository information
- [x] **Add Citation File**: Created `CITATION.cff` with complete metadata including author ORCID IDs and paper keywords
- [x] **Format Files**: Ran `npm run format` to ensure all files are properly formatted
- [x] **Commit Changes**: Using standardized commit messages
- [x] **Finalize README**: Updated `README.md` to reflect actual repository purpose (CHR 2025 conference materials)
- [x] **LaTeX Setup**: Verified LaTeX compilation works correctly for the abstract

### In Progress 🔄

- [ ] **Enable GitHub Security Alerts**: Go to repository "Security" tab on GitHub and enable security alerts
- [ ] **Protect the Main Branch**: In repository settings on GitHub (under "Branches"), protect the `main` branch
- [ ] **Set Up Zenodo Integration**: Follow the [Zenodo guide](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content) to connect repository to Zenodo
- [ ] **Set Up Zenodo DOI Badge**: Replace `GITHUB_REPO_ID` with `id` from `https://api.github.com/repos/maehr/chr2025-seeing-history-unseen`
- [ ] **Add Zenodo DOI to README**: Once Zenodo DOI is obtained, add it to `README.md` by replacing `ZENODO_RECORD`
- [ ] **Update Remaining Placeholders**: Replace `[INSERT CONTACT METHOD]` in remaining files

### Optional Tasks

- [ ] **Add Zenodo Metadata File**: Create `.zenodo.json` file for Zenodo metadata ([Zenodo developer docs](https://developers.zenodo.org/?python#add-metadata-to-your-github-repository-release))
- [ ] **Generate Changelog**: Run `npm run changelog` and copy output into `CHANGELOG.md` file
- [ ] **Customize Documentation**: Customize documentation using [Quarto's features](https://quarto.org/docs/websites/#workflow)
- [ ] **Enable GitHub Pages**: In repository settings on GitHub (under "Pages"), configure GitHub Pages to publish from the `gh-pages` branch
- [ ] **Publish Documentation**: Run `quarto publish gh-pages` to publish documentation website

## Conference Paper Tasks

### Abstract

- [x] **LaTeX Source**: Complete abstract written in LaTeX using `anthology-ch.cls` class
- [x] **Bibliography**: Bibliography entries in `bibliography.bib`
- [x] **Build System**: Makefile for building the paper
- [x] **PDF Generation**: Successfully generates 10-page PDF document

### Presentation

- [ ] **Create Initial Slide Outline**: Develop structure for conference presentation
- [ ] **Develop Main Content**: Create presentation slides with key findings
- [ ] **Add Examples and Demonstrations**: Include examples of AI-generated alt-text
- [ ] **Add Figures**: Create additional visualizations for presentation
- [ ] **Practice and Refine**: Rehearse presentation and refine content
- [ ] **Final Review**: Complete final review and formatting

### Research Tasks

- [ ] **Complete User Study**: Conduct user study with blind/low-vision individuals
- [ ] **Finalize Evaluation**: Complete systematic evaluation of AI models
- [ ] **Dataset Release**: Prepare and release dataset as benchmark for future research
- [ ] **Develop Guidelines**: Create practical guidelines for heritage institutions

## Development Tasks

- [x] **Clean LaTeX Build Files**: Remove intermediate `.aux`, `.bbl`, `.bcf`, etc. files from repository
- [ ] **Update Template Changes**: Incorporate latest updates from the base template
- [x] **Configure Gitignore**: Add LaTeX intermediate files to `.gitignore`
- [ ] **Test All Build Commands**: Verify all npm scripts and make targets work correctly

## Notes

- Keep this file updated as tasks are completed or new ones are identified
- Mark completed items with `[x]` and update status emojis (✅, 🔄, etc.)
- Add new sections as needed for different aspects of the project
