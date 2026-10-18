# US Phillips Curve Simulator (final presentation)

Streamlit app + Colab-trained models on nine FRED series (Review-1 scope). Monthly data, 2003+ for the full-variable models, 1960+ for the era tests.

## Work split (Person A and Person B work in parallel)

| | Person A: Data and models | Person B: App and deployment |
|---|---|---|
| Now | Run `notebooks/01_data_and_models.ipynb` in Colab (section by section) | `python scripts/make_stub_artifacts.py`, then `streamlit run app.py` (works on stub data immediately) |
| Build | Check the FRED pull, EDA plots, model metrics, HAC tables, era slopes (H1-H4) | Create the GitHub repo, tune the annotation offsets, layout, styling, slider behaviour, warnings |
| Hand-off | Download `phillips_monthly.csv`, `models.joblib`, `metrics.json`; put the printed sklearn version in `requirements.txt` | Drop the three files into `data/` and `artifacts/`, delete the stub banner logic only if you wish (it turns off by itself) |
| Then | Write the findings for each hypothesis (use numbers from `metrics.json`); results slides | Deploy on Streamlit Community Cloud, test on phone, write the live-demo script |
| Together | Integration check, rehearsal, final presentation. Never edit `src/config.py` alone: it is the contract. | |

## The contract (`src/config.py`)
- `data/phillips_monthly.csv`: index `date` (month start), columns `cpi_yoy, core_yoy, unrate, nrou, mich, t5yie, fedfunds, wti, m2_yoy`.
- `artifacts/models.joblib`: dict with keys `"<target>:<model>"`, target in `cpi_yoy|core_yoy`, model in `classic|augmented|full_ols|rf`; fitted on DataFrames with the feature names in `config.MODELS`.
- `artifacts/metrics.json`: schema as written by the notebook (see `scripts/make_stub_artifacts.py` for a small example).
- Prediction goes only through `src/predict.py`.

## Deploy
1. Push this folder to a GitHub repo (include `data/` and `artifacts/`).
2. share.streamlit.io, New app, pick the repo, main file `app.py`.
3. `requirements.txt` must pin the same scikit-learn version as Colab, or the joblib file will not load.

## What changed from the first version
- Monthly data (about 280 observations instead of about 23 annual points).
- Out-of-sample metrics with a time-ordered split; the old app only reported training R².
- M2 as YoY growth (the level is trending and not comparable across decades).
- Review-1 items added: core CPI toggle, NROU long-run curve, era stability test (H1), NROU drift (H2), oil (H3), expectations (H4), limits tab, breakeven cross-check.
- Page text is driven by the computed metrics, not by pre-written conclusions.
