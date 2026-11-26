# Process scope tool

Demo tool for the two border algorithms of the thesis: **hierarchical merging** and **Infomap** on the
signature graph of an OCEL 2.0 log.

- **Learn how it works**: guided walkthrough on the official Logistics log (bundled), 9 steps,
  with a *free explore* switch on the signals, boundary score, merge and Infomap steps.
- **Run on my log**: upload an OCEL 2.0 JSON file → signal overview → merge (k or τ) → Infomap (Markov time)
  → compare → export Procellar rules + a Markdown report. Optional GT rules file for F1.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. Everything runs locally; uploaded logs do not leave the machine.
Logs up to 2 GB can be uploaded (`.streamlit/config.toml`). Large logs (BPIC 2017) take a while
on first load; results are cached while the app runs.

## Link to Procellar

The home page and the export tab link to the Procellar repository (https://github.com/hudsonjychen/procellar).
Procellar has no hosted version; its README explains how to run it locally and import the exported rules.

## Files

| file | what |
|---|---|
| `engine.py` | the notebook cells as functions: signatures, hand-overs, signals, b, merge, Infomap, stage 5 rules, GT F1 |
| `viz.py` | Plotly figures: signature graph (fixed layered layout), dendrogram, alluvial, signal bars, F1 curve |
| `common.py` | caching, glossary, comparisons (event-weighted ARI), report |
| `views/` | home, explainer, workbench pages |
| `data/` | official Logistics OCEL + fixed GT rules |
| `tests/check_logistics.py` | checks the engine against the notebook outputs (F1 curve, first merge, Infomap, rules) |

## Differences to the notebook

- Combine is the weighted mean only (no max switch).
- Defaults = thesis settings: equal weights, change of pace 0, θ 0, λ 0, Infomap `together`, seed 42, 20 trials.
- Logic of signals, merge, Infomap, rules and F1 is copied from the cells unchanged; ties in the merge are
  still broken by key order and are flagged in the UI.
- Only OCEL 2.0 **JSON** is read (same reader as the notebook).
