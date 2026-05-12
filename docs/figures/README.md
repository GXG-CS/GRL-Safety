# Figures for the project site

Place exported plots and diagrams here. The main page references images with **relative** paths, for example `figures/overview.svg`.

## Naming

Use short, stable names so `index.html` does not need frequent edits:

- `overview.svg` or `overview.png` — benchmark / pipeline overview
- `results_<model>_<task>.png` — optional pattern for per-model plots

## Regeneration

Document the exact command or notebook cell used to produce each asset (paths relative to the repository root), for example:

- Script: `…`
- Command: `python …`

Update this file when you add new figures so the site stays reproducible.
