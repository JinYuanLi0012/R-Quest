# R-Quest paper homepage

A standalone HTML/CSS/JavaScript website, ready for GitHub Pages. All displayed figures are converted from the supplied paper PDF figures. No npm installation or build is required.

## Preview

Open `index.html` in a browser, or serve this folder with a local HTTP server.

## Publish in the R-Quest code repository

1. Copy this folder's contents into `docs/` in `https://github.com/JinYuanLi0012/R-Quest`.
2. Commit and push the files.
3. Open repository **Settings → Pages**.
4. Choose **Deploy from a branch**, branch **main**, folder **/docs**, then **Save**.
5. The expected project URL is `https://JinYuanLi0012.github.io/R-Quest/`.

All internal asset paths are relative, so the site works under the repository URL prefix. The `.nojekyll` file disables Jekyll processing for these static files.

## Paper and citation

The Code buttons point to the provided R-Quest repository. The Paper button is disabled and marked **Coming soon** because arXiv has not been published. Once the paper is available, replace the disabled button in `index.html` with an anchor to the arXiv URL. Update the BibTeX block with the final arXiv metadata at the same time. The current `@misc` entry is provisional and includes the code repository URL, not an invented arXiv identifier.

## Content provenance

Source: `rzero_iclr2027_overleaf.zip`. Authors and affiliations follow the order supplied in the conversation, including Jiaxin Huang as corresponding author.

- Hero summary: `sections/abstract.tex`; hero caption and performance: `sections/introduction.tex`.
- Invalid-question and repeated-task analysis: `sections/background.tex`.
- Method and novelty probability: `sections/method/overview.tex`, `validity.tex`, `novelty.tex`, and `training_loop.tex`.
- Benchmark scores, extended-run results, ablations and quality analysis: `sections/experiments.tex`.
- Figures in `assets/` preserve the paper originals; `.webp` files provide high-resolution browser images, and `.pdf` files preserve vector originals.

The website separates the five-round benchmark tables from the ten-round extended-run analysis. It does not claim that the 17.32-point difference is measured against R-Zero's best checkpoint: it is the same-round difference at round ten.

## Interactions

- Light/dark theme, saved locally.
- A centered, explicitly two-line title and linked authors, followed by six animated domain-average bar groups and an inspectable ten-round trajectory.
- Bar growth on entry and a Replay button; reduced-motion preferences show the completed chart immediately.
- Invalid-question trend and controlled-repair figures (Fig. 3a and the full Fig. 3b), followed by the task-repetition analysis.
- A round-by-round trajectory animation, Replay, hover inspection and a keyboard-accessible round slider.
- Separate validity and novelty comparisons, using the corresponding halves of the original overview figure.
- Interactive novelty probability curves using the paper's large-batch approximation `(1-p)^K`, with K selection, pass/rejection toggles, pointer tooltips and keyboard navigation.
- Backbone and domain selection with named, per-benchmark results across all twelve benchmarks.
- Four analysis figure tabs.
- Figure enlargement with a keyboard-accessible dialog.
- Copyable BibTeX citation.

The site uses no analytics, accounts, model APIs or external JavaScript dependencies.

The trajectory values in `trajectory-data.js` and `assets/trajectory-data.csv` were recovered from the vector paths in `assets/evolution.pdf`, then independently cross-checked against `assets/teaser.pdf`. They reproduce the plotted scores to two decimal places; they are not raw, unrounded experiment logs. Round zero is the base model. The robot illustrations are embedded images extracted from the original teaser PDF. The overview halves are mechanical crops of the original overview, without altering its contents.

Hero bars show Base, R-Zero and R-Quest in that order. Scores match the Average columns in `benchmark-data.js`. A blue training label and arrow connect math-focused self-evolution to purple code/general transfer labels. Each group uses a non-zero baseline to magnify differences; broken-axis marks and the caption disclose this zoom. The underlying scales are Qwen Math 44–54%, Code 56–64%, General 27–34%, and OctoThinker Math 22–32%, Code 11–20%, General 5–17%. Exact ranges remain in the charts’ accessible descriptions. Gain badges and visible axis-range labels are omitted.
