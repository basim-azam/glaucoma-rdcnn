# CD-Former paper draft

LaTeX source for the CD-Former manuscript. Built with `pdflatex` against `IEEEtran` class file (default for IEEE TMI / IEEE JBHI submissions). For MICCAI / Springer LNCS, swap `\documentclass{IEEEtran}` for `\documentclass{llncs}` and reformat the title block.

## Build

```bash
make            # produces main.pdf
make clean      # remove intermediate files
```

If `IEEEtran.cls` is missing locally, install via your TeX distribution:
- Ubuntu/Debian: `sudo apt install texlive-publishers`
- macOS (MacTeX): bundled
- Overleaf: select the `IEEE Transactions on Medical Imaging` template, paste in `main.tex`, `tables.tex`, `refs.bib`.

## File layout

```
main.tex      — paper body (~400 lines)
tables.tex    — 5 LaTeX tables (drop-in)
refs.bib      — bibliography (21 entries)
figures/      — placeholder; generate from notebooks/figures.py
```

## Placeholders to fill before submission

Search for `% PLACEHOLDER` in `main.tex`:
- Discussion section: 1-2 paragraphs
- Conclusion section: ~120 words
- Author list and affiliations (currently anonymous)
- Figure 1: architecture diagram (export `cd_former_architecture` widget as PDF)

Numerical results that may be updated as further experiments land:
- Table~\ref{tab:drishti}: "CD-Former (ours)" row. Currently single-seed best (90.57\% OC, 0.990 AUC). To be updated with 3-seed ensemble (expected +1-2 pp).
- Table~\ref{tab:rimone}: same.
- Table~\ref{tab:ablation}: multi-rater configuration row to be added after task #10 completes.

## Target venue

- **Primary**: IEEE Transactions on Medical Imaging (current template).
- **Backup**: Medical Image Analysis (Elsevier), MICCAI 2026, IEEE JBHI.
