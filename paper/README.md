# Paper — CascadeBench (SoftwareX OSP)

SoftwareX Original Software Publication. Software described lives in the repository root
(`../`).

## Files
- `main.tex` — entry point (**elsarticle** class, SoftwareX template); `\input`s the sections.
- `abstract.tex · introduction.tex · related_work.tex · methodology.tex · results.tex · discussion.tex · conclusion.tex`
- `refs.bib` — references (domain literature only).

OSP structure: Motivation and significance · Software description · Illustrative examples ·
Impact · Conclusions. ~3000 words, 1 figure (TikZ architecture), code-metadata table C1–C8.

## Build
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
`elsarticle` ships with TeX Live. No local TeX? Build in Docker:
```bash
docker run --rm -v "$PWD":/w -w /w texlive/texlive:latest \
  sh -c "pdflatex main && bibtex main && pdflatex main && pdflatex main"
```

## Overleaf
Import the repository into Overleaf and set **`paper/main.tex`** as the main document
(compiler: pdfLaTeX). `\input` and `refs.bib` are relative to `paper/`, so it compiles as-is.

## Notes
- Result tables/figures use **real CascadeBench runs**, reproducible from `../results/*.csv`
  (cb_panel, cb_headline_ci, exp_cascade_enron, cb_robustness_curves, cb_transfer).
- C2 metadata points at the public repo; the Zenodo DOI is reserved on acceptance.
