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
- `checklist.tex`: separate NeurIPS checklist file retained alongside the report
- `figures/cumulative_rmse.pdf`: report-local experiment trajectory figure
- `figures/cumulative_rmse.png`: PNG version of the same figure
- `figures/performance_cumulative_clean.png`: cleaned cumulative RMSE trajectory figure for the report

No bibliography tool is required because references are written directly in `main.tex`.
