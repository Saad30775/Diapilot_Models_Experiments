# DiaPilot

DiaPilot is a diabetic prevention and management app with an AI-driven diet
and exercise recommendation engine. Patients answer a clinically-based risk
questionnaire (Diabetes UK Leicester score model), and — depending on
whether they are in **Prevention Mode** or **Management Mode** — receive a
personalized 30-day, 90-meal diet plan (breakfast/lunch/dinner), localized
to Pakistani cuisine, paired with a rule-based exercise schedule.

---

## How it actually works (current state)

```
React Native app
      |  POST /generate-plan  (patient profile JSON)
      v
Flask API (app.py)
      |
      +- Clinical rule engine
      |     Leicester score, medication status, glucose reading
      |     -> per-meal safe carb / sugar / calorie limits
      |
      +- Recipe pool (Food.com, 117,662 cleaned recipes)
      |     filtered down to recipes inside the safe limits above
      |
      +- DNN relevance model  (the production engine - see Models below)
      |     scores every filtered recipe for this specific patient
      |     in one batched call, picks the best match per meal slot,
      |     for all 30 days
      |
      +- Gemini API - Pakistani conversion
      |     rewrites recipe names to their closest Pakistani equivalent
      |     (best-effort: if this step fails, original names are returned
      |     instead of failing the whole request - see Models below)
      |
      +- Exercise rule engine
            post-meal walk + rotating 7-day evening workout,
            with age/BMI/glucose safety overrides
      |
      v
JSON response -> rendered in the app's diet plan screen
```

This README documents what is **actually built and tested** as of today.
See [Known Gaps](#known-gaps-vs-original-design-docs) for the places where
the original design documentation describes more than what's implemented.

---

## Repository Structure

```
DiaPilot_Workspace/
├── data/
│   └── processed/
│       ├── DiaPilot_Combined_Data.parquet   # Training dataset (already included)
│       └── recipes_clean.csv                # Recipe pool used by app.py at inference time
├── models/                # Trained model artifacts (committed - see Models below)
├── notebooks/             # The 4 experiments (KNN, XGBoost, Random Forest, DNN)
├── reports/               # Generated CSV outputs from each experiment
├── src/
│   ├── data_pipeline.py          # Cleans + preprocesses raw recipe data (optional, see note below)
│   ├── generate_combined_pdf.py  # Standalone script: adds exercise columns to a generated diet plan CSV
│   └── app.py                    # Flask API - the live backend the mobile app calls
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

### 2. Two Python environments are required

This project needs **two separate virtual environments** because the
notebooks (sklearn / XGBoost) and the Flask API (TensorFlow / Keras) were
developed against different Python versions due to a TensorFlow/Python 3.14
compatibility issue.

**Environment A - for notebooks (KNN, XGBoost, Random Forest):**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install pandas numpy scikit-learn xgboost jupyter
```

**Environment B - for the DNN notebook and the Flask API:**
```bash
py -3.12 -m venv .venv312
.venv312\Scripts\activate       # Windows
pip install pandas numpy scikit-learn tensorflow flask google-generativeai jupyter
```

(No `requirements.txt` is pinned yet - if you hit a version mismatch on any
package, check the import cell at the top of the relevant notebook.)

### 3. Data - no download needed for the training data

`data/processed/DiaPilot_Combined_Data.parquet` is **already committed**.
This is what the four training notebooks load directly - you do **not**
need to download the raw Food.com CSV to retrain the models.

You only need the raw dataset if you want to rebuild the processed file
from scratch:
- Source: [Food.com Recipes and Interactions - Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- File needed: `RAW_recipes.csv` -> place at `data/raw/RAW_recipes.csv` (git-ignored)
- Then run: `cd src && python data_pipeline.py`

### 4. Running the Flask API locally

```bash
cd src
.venv312\Scripts\activate
$env:GEMINI_API_KEY="your-gemini-api-key-here"     # PowerShell
python app.py
```

The server starts at `http://localhost:5000` (and on your machine's local
network IP - check the terminal output for the exact address). The mobile
app must be configured with this address; see `services/dietPlanApi.ts` in
the React Native project for where that's set.

If `GEMINI_API_KEY` is not set, the API still works - diet plans are
returned with original English recipe names instead of Pakistani names.

**Test it directly** (without the mobile app) using curl or Postman:
```bash
curl -X POST http://localhost:5000/generate-plan ^
  -H "Content-Type: application/json" ^
  -d "{\"age\": 47, \"weight_kg\": 86, \"height_cm\": 173, \"gender\": \"male\", \"activity\": \"sedentary\", \"leicester_score\": 16, \"on_medication\": false, \"current_glucose_high\": false}"
```

---

## Models

Four recommendation approaches were trained and compared on the same
90-meal generation task:

| File | Model | Notebook | Used by app.py? |
|---|---|---|---|
| `diapilot_dnn_best.keras` + `diapilot_dnn_scaler.pkl` | Deep Neural Network | `Exp3_DNN_fixed.ipynb` | **Yes - this is the production model** |
| `diapilot_xgb_ranker.pkl` | XGBoost ranker | `Exp2_XGBoost_Ranker.ipynb` | No - comparison only |
| `diapilot_rf_model.pkl` / `diapilot_rf_importance.pkl` | Random Forest | `Exp4_RandomForest.ipynb` | No - comparison only |
| `diapilot_macro_scaler.pkl`, `diapilot_tfidf_vectorizer.pkl` | Shared preprocessing | used across notebooks | No - notebook-only artifacts |

KNN (`Exp1_KNN_Baseline.ipynb`) is evaluated as a baseline and requires no
saved model file.

**The DNN was chosen as the production model** because, across testing on
multiple patient profiles (including profiles outside the original
training set), it produced the most clinically appropriate per-patient
variation in macros and meal selection. A full comparative write-up of all
four experiments is available separately as a working document for the
research paper.

**DNN architecture (confirmed via direct inspection):** two-branch network
— a patient branch taking 8 features (`p_age, p_bmi, p_leicester, p_tdee,
p_medication, p_carb_target, p_sugar_limit, p_cal_limit`) and a meal branch
taking 5 features (`m_calories, m_carbs, m_sugar, m_protein, m_fat`),
merged into dense layers, outputting a single relevance score per
patient-meal pair. `app.py` scores the entire filtered recipe pool against
a patient in **one batched call** (not one call per recipe) - this dropped
typical response time from over 11 minutes to under 1 second.

---

## Exercise Module

Exercise recommendations are **rule-based**, not learned by a model. This
is a deliberate design choice (not a placeholder) - see
[Known Gaps](#known-gaps-vs-original-design-docs) for the original
documentation's more elaborate dataset-lookup design, which was simplified
for time.

`app.py` generates two things per request:
- `post_meal_exercise` - a short walk recommendation per individual meal,
  with safety overrides: glucose currently high -> gentle walk only;
  age >= 66 -> slow walk; BMI > 35 -> slow walk (joint-friendly); age 56-65
  -> brisk walk, shorter duration; all other adults -> brisk walk.
- `daily_evening_workout` - one workout per day, cycling through a 7-item
  pool so the 30-day plan doesn't repeat the same workout every day.

A standalone script, `src/generate_combined_pdf.py`, applies the same
exercise logic to an already-generated diet plan CSV (used during
notebook-only testing, independent of the live API).

---

## Known Gaps vs. Original Design Docs

The project's workflow documentation (Module 3 and Module 5) describes a
more elaborate system than what is implemented in `app.py` today. Documented
here explicitly so this is a known, stated limitation rather than a
surprise:

- **Allergy / dietary preference filtering** (Module 3): documented as a
  planned input (Veg/Non-Veg/Vegan, allergen exclusion) but **not yet
  implemented** in `risk_assessments` or `app.py`. No allergy data is
  currently collected or filtered on.
- **Exact age vs. age bracket**: the risk assessment questionnaire
  intentionally collects age as a bracket (e.g. "50-59"), matching the
  Diabetes UK Leicester tool's own format. Since `app.py`'s calorie
  calculation (Mifflin-St Jeor) needs a single numeric age, the bracket's
  midpoint is used as a reasonable approximation (e.g. "50-59" -> 55).
- **Exercise dataset lookup** (Module 5): documented as a dynamic
  calorie-burn lookup against an exercise dataset by MET value. Implemented
  instead as a fixed rule-based table (see Exercise Module above) - chosen
  for reliability and speed under time constraints, not because the
  dataset-lookup approach was found to be worse.
- **Real-time medication / glucose check-in wiring**: the clinical logic
  for both already exists and is tested in `app.py` (`on_medication`,
  `current_glucose_high` parameters). What's not yet wired up is pulling
  these from the app's actual daily check-in / medication log screens into
  the API call automatically - currently the integration uses sensible
  defaults where this data isn't yet passed through from the app.

---

## Status

- [x] Data pipeline (Food.com dataset -> cleaned processed dataset)
- [x] Four diet models trained and compared (KNN, XGBoost, Random Forest, DNN)
- [x] DNN selected as production model, verified on multiple patient profiles
- [x] Rule-based exercise module implemented
- [x] Flask API (`app.py`) - built, batched for speed, tested end-to-end
- [x] Gemini Pakistani food-name conversion - implemented, fails gracefully
- [ ] React Native screens wired to call the live API - in progress
- [ ] Allergy / diet-type filtering - documented as planned, not yet built
- [ ] Real-time medication/glucose data passed automatically from app screens to API - partial (logic exists, wiring from UI in progress)