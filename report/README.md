# Report Build Instructions

This report uses the professor-provided NeurIPS 2026 style files already located in `report/`.

## Compile

From the repository root:

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

This produces `report/main.pdf`.

## Files

- `main.tex`: report source
- `neurips_2026.sty`: professor-provided style file used by `main.tex`
- `checklist.tex`: NeurIPS checklist included at the end of the paper
- `figures/cumulative_rmse.pdf`: report-local experiment trajectory figure
- `figures/cumulative_rmse.png`: PNG version of the same figure

No bibliography tool is required because references are written directly in `main.tex`.
