# DiaPilot

DiaPilot is a diabetic diet and exercise recommendation system. It generates personalized 30-day, 90-meal diet plans (breakfast/lunch/dinner) constrained by patient clinical attributes (age, BMI, glucose level, risk mode), and pairs each plan with a rule-based exercise schedule (post-meal walking + a rotating 7-day evening workout).

This repository contains the data pipeline, four trained recommendation models (compared experimentally), and the exercise rule engine. Model-to-app integration (Flask API) is in progress — see [Status](#status) below.

---

## Repository Structure

```
DiaPilot_Workspace/
├── data/
│   └── processed/
│       └── DiaPilot_Combined_Data.parquet   # Already included — see Setup below
├── models/                # Trained model artifacts (committed — see Models below)
├── notebooks/             # The 4 experiments (KNN, XGBoost, Random Forest, DNN)
├── reports/               # Generated CSV/PDF outputs from each experiment
├── src/
│   ├── data_pipeline.py          # Cleans + preprocesses raw recipe data (optional, see note below)
│   └── generate_combined_pdf.py  # Adds exercise columns to a generated diet plan
├── .gitignore
└── readme.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Saad30775/Diapilot_Models_Experiments.git
cd Diapilot_Models_Experiments
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install pandas numpy scikit-learn xgboost tensorflow jupyter
```

(No `requirements.txt` is pinned yet — if you hit a version mismatch on any package, check the import cell at the top of the relevant notebook.)

### 3. Data — no download needed

`data/processed/DiaPilot_Combined_Data.parquet` is **already committed in this repository**. This is the cleaned, feature-engineered dataset that all four notebooks load directly — you do **not** need to download the raw Food.com CSV or run `data_pipeline.py` to get started.

You only need the raw dataset if you want to rebuild the processed file from scratch (e.g. to change cleaning/feature logic). In that case:

- Source: [Food.com Recipes and Interactions — Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- File needed: `RAW_recipes.csv`
- Place it at: `data/raw/RAW_recipes.csv` (this path is git-ignored — it will not be committed)
- Then run: `cd src && python data_pipeline.py`

---

## Models

Four recommendation approaches were implemented and compared on the same 90-meal generation task. Trained artifacts are committed in `models/` so the notebooks (and eventually the API) can run without retraining:

| File | Model | Notebook |
|---|---|---|
| `diapilot_dnn_best.keras` | Deep Neural Network | `Exp3_DNN_fixed.ipynb` |
| `diapilot_xgb_ranker.pkl` | XGBoost ranker | `Exp2_XGBoost_Ranker.ipynb` |
| `diapilot_rf_model.pkl` / `diapilot_rf_importance.pkl` | Random Forest | `Exp4_RandomForest.ipynb` |
| `diapilot_macro_scaler.pkl` | Feature scaler (shared preprocessing) | used across notebooks |
| `diapilot_tfidf_vectorizer.pkl` | TF-IDF vectorizer (recipe text features) | used across notebooks |

KNN (`Exp1_KNN_Baseline.ipynb`) is evaluated as a baseline and does not require a saved model file — it runs directly against the processed dataset.

A full comparative write-up of all four experiments (methodology, metrics, results) is available separately as a working document for the group's research paper.

> **Note:** The exact input feature names/order expected by each `.pkl`/`.keras` file have not yet been fully audited and documented here. Inspect each notebook's feature-preparation cell before wiring these models into a new script (e.g. the planned Flask API), to avoid shape-mismatch errors at inference time.

---

## Running an Experiment

Each notebook in `notebooks/` is self-contained and loads `data/processed/DiaPilot_Combined_Data.parquet` directly:

```bash
cd notebooks
jupyter notebook Exp3_DNN_fixed.ipynb
```

Run all cells in order. Each notebook loads the processed dataset, trains (or loads) its model, generates a 30-day plan, and saves results to `reports/` (CSV, and in some cases PDF). Sample outputs from prior runs are already present in `reports/` for reference.

---

## Exercise Module

Unlike the diet models above, exercise recommendations are **rule-based**, not learned. Given a generated diet plan CSV, `generate_combined_pdf.py` adds:

- `Post_Meal_Exercise` — a walk recommendation per individual meal, scaled to that meal's calories and capped/adjusted by the patient's age and current glucose reading.
- `Daily_Evening_Workout` — one workout per day, drawn from a rotating 7-day pool matched to the patient's age bracket and risk profile.

Run it against an already-generated diet plan:

```bash
cd src
python generate_combined_pdf.py
```

Safety overrides applied (in priority order): glucose > 10.0 mmol/L → gentle walk only; age ≥ 66 → slow walk, capped duration; BMI > 35 → slow walk, joint-friendly pace; age 56–65 → brisk walk, capped duration; all other adult patients → brisk walk, duration scaled to meal calories.

---

## Status

- [x] Data pipeline (Food.com dataset → cleaned processed dataset, already committed)
- [x] Four diet models trained and compared (KNN, XGBoost, Random Forest, DNN)
- [x] Rule-based exercise module implemented
- [ ] Flask API wrapping model inference for the mobile app (`app.py`) — **in progress, not yet in this repo**
- [ ] Mobile app (React Native + Appwrite) integration with the above API — **not yet started**

If you are picking this repo up to continue work: the next step is building `app.py`, a small Flask service exposing one endpoint (e.g. `POST /generate-plan`) that loads the models above and returns a generated plan as JSON for the mobile app to consume.