"""
app.py
DiaPilot Flask API — generates a full 30-day diet + exercise plan
for a given patient profile, using the trained DNN relevance model.

Run with:
    cd src
    python app.py

Then call:
    POST http://localhost:5000/generate-plan
    Body (JSON):
    {
        "age": 47,
        "weight_kg": 86,
        "height_cm": 173,
        "gender": "male",
        "activity": "sedentary",
        "leicester_score": 16,
        "on_medication": false,
        "current_glucose_high": false
    }
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow import keras

app = Flask(__name__)

# ────────────────────────────────────────────────────────────
# Load model + scaler ONCE at startup (not per request — slow otherwise)
# ────────────────────────────────────────────────────────────
MODELS_DIR = "../models"

dnn_model = keras.models.load_model(
    os.path.join(MODELS_DIR, "diapilot_dnn_best.keras")
)
scaler_X = joblib.load(
    os.path.join(MODELS_DIR, "diapilot_dnn_scaler.pkl")
)

# ────────────────────────────────────────────────────────────
# TODO (Saad): point this at your actual cleaned recipe dataset
# — the same one used in the notebook to build `df` before
#   training (the one with columns: Recipe_Name, Calories,
#   Carbs_g, Sugar_g, Protein_g, Fat_g, Meal_Type).
# Easiest: export that dataframe from the notebook once with
#   df.to_csv("../data/processed/recipes_clean.csv", index=False)
# then point the path below at it.
# ────────────────────────────────────────────────────────────
RECIPE_DATA_PATH = "../data/processed/recipes_clean.csv"

if os.path.exists(RECIPE_DATA_PATH):
    recipes_df = pd.read_csv(RECIPE_DATA_PATH)
else:
    recipes_df = None
    print(f"WARNING: {RECIPE_DATA_PATH} not found.")
    print("Set RECIPE_DATA_PATH in app.py to your cleaned recipe CSV"
          " before this API can generate real plans.")

PATIENT_FEATURE_ORDER = [
    "p_age", "p_bmi", "p_leicester", "p_tdee",
    "p_medication", "p_carb_target", "p_sugar_limit", "p_cal_limit",
]
MEAL_FEATURE_ORDER = [
    "m_calories", "m_carbs", "m_sugar", "m_protein", "m_fat",
]

DAYS_IN_PLAN = 30
MEAL_SLOTS = ["Breakfast", "Lunch", "Dinner"]


# ────────────────────────────────────────────────────────────
# Clinical engine — identical logic to the training notebook
# ────────────────────────────────────────────────────────────
def calculate_tdee(age, weight_kg, height_cm, gender, activity="sedentary"):
    """Mifflin-St Jeor Equation."""
    if gender.lower() == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
    return round(bmr * multipliers.get(activity, 1.2))


def get_dynamic_macros(tdee, leicester_score, on_medication=False,
                        current_glucose_high=False):
    """ADA 2025-aligned per-meal constraints (same as training notebook)."""
    if leicester_score <= 10:
        carb_pct, sugar_limit = 0.50, 15.0
    elif leicester_score <= 15:
        carb_pct, sugar_limit = 0.40, 10.0
    else:
        carb_pct, sugar_limit = 0.30, 8.0

    if current_glucose_high:
        carb_pct -= 0.10
        sugar_limit = 3.0

    per_meal_carbs_max = round(((tdee * carb_pct) / 4) / 3)
    per_meal_carbs_min = 25 if on_medication else 0
    if on_medication and per_meal_carbs_max < 25:
        per_meal_carbs_max = 25

    return {
        "per_meal_calories": round(tdee / 3),
        "per_meal_max_carbs": per_meal_carbs_max,
        "per_meal_min_carbs": per_meal_carbs_min,
        "per_meal_sugar": sugar_limit,
    }


# ────────────────────────────────────────────────────────────
# Exercise rule engine — same logic as generate_combined_pdf.py
# ────────────────────────────────────────────────────────────
def get_post_meal_exercise(age, current_glucose_high, bmi):
    """Per-meal walk recommendation, with clinical safety overrides."""
    if current_glucose_high:
        return "Gentle Walk 10 mins"
    if age >= 66:
        return "Slow Walk 10 mins"
    if bmi > 35:
        return "Slow Walk 10 mins (joint-friendly pace)"
    if age >= 56:
        return "Brisk Walk 15 mins"
    return "Brisk Walk 20 mins"


WORKOUT_POOL = [
    "Bodyweight circuit: squats, lunges, wall push-ups (15 min)",
    "Resistance band: rows, chest press, bicep curls (15 min)",
    "Stationary cycling or brisk treadmill walk (20 min)",
    "Light dumbbell strength: shoulder press, deadlift (15 min)",
    "Active rest day: gentle stretching + mobility work (10 min)",
    "Bodyweight circuit: step-ups, glute bridges, plank holds (15 min)",
    "Resistance band: lateral raises, leg press, tricep extensions (15 min)",
]


def get_daily_evening_workout(day_index):
    """Cycles through a 7-day rotating workout pool."""
    return WORKOUT_POOL[day_index % len(WORKOUT_POOL)]


# ────────────────────────────────────────────────────────────
# DNN scoring — BATCHED: scores an entire pool of candidate
# meals for ONE patient in a single model.predict() call.
#
# Why this matters: calling dnn_model.predict() one row at a
# time (the old approach) pays TensorFlow's per-call overhead
# thousands of times over. Stacking all candidate meals into
# one matrix and predicting once is dramatically faster — the
# same batching pattern already used successfully in the
# training notebook's own inference step.
# ────────────────────────────────────────────────────────────
def score_meal_pool(patient_vector_8, pool_df):
    """
    patient_vector_8: list of 8 floats, in PATIENT_FEATURE_ORDER
    pool_df: DataFrame with Calories, Carbs_g, Sugar_g, Protein_g, Fat_g
             (one row per candidate meal)
    Returns: numpy array of relevance scores, one per row in pool_df,
             in the same order as pool_df.
    """
    n_rows = len(pool_df)

    # Repeat the same patient vector once per candidate meal,
    # so every meal gets paired with the patient for scoring.
    patient_block = np.tile(
        np.array(patient_vector_8, dtype=np.float32), (n_rows, 1)
    )

    meal_block = pool_df[
        ["Calories", "Carbs_g", "Sugar_g", "Protein_g", "Fat_g"]
    ].to_numpy(dtype=np.float32)

    combined = np.hstack([patient_block, meal_block])
    combined_scaled = scaler_X.transform(combined)

    n_p = len(PATIENT_FEATURE_ORDER)
    patient_part = combined_scaled[:, :n_p]
    meal_part = combined_scaled[:, n_p:]

    scores = dnn_model.predict(
        [patient_part, meal_part], batch_size=1024, verbose=0
    )
    return scores.flatten()


def pick_best_meals_for_slot(meal_type, patient_vector_8, macros, n_needed):
    """
    Filters the recipe pool to this meal slot + hard medical constraints,
    scores the ENTIRE filtered pool in one batched DNN call, returns the
    top n_needed, sorted by score descending.
    """
    pool = recipes_df[recipes_df["Meal_Type"] == meal_type].copy()

    pool = pool[
        (pool["Calories"] <= macros["per_meal_calories"]) &
        (pool["Carbs_g"] <= macros["per_meal_max_carbs"]) &
        (pool["Carbs_g"] >= macros["per_meal_min_carbs"]) &
        (pool["Sugar_g"] <= macros["per_meal_sugar"])
    ]

    if pool.empty:
        return []

    pool["relevance_score"] = score_meal_pool(patient_vector_8, pool)

    pool = pool.sort_values("relevance_score", ascending=False)

    # If fewer unique meals pass filters than days needed, cycle through them
    selected = []
    pool_records = pool.to_dict("records")
    for i in range(n_needed):
        selected.append(pool_records[i % len(pool_records)])
    return selected


# ────────────────────────────────────────────────────────────
# Main endpoint
# ────────────────────────────────────────────────────────────
@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    if recipes_df is None:
        return jsonify({
            "error": "Recipe dataset not loaded on the server. "
                     "Set RECIPE_DATA_PATH in app.py first."
        }), 500

    data = request.get_json(force=True)

    try:
        age = float(data["age"])
        weight_kg = float(data["weight_kg"])
        height_cm = float(data["height_cm"])
        gender = data["gender"]
        activity = data.get("activity", "sedentary")
        leicester_score = float(data["leicester_score"])
        on_medication = bool(data.get("on_medication", False))
        current_glucose_high = bool(data.get("current_glucose_high", False))
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    # ── Clinical engine ──────────────────────────────────
    tdee = calculate_tdee(age, weight_kg, height_cm, gender, activity)
    macros = get_dynamic_macros(tdee, leicester_score, on_medication, current_glucose_high)
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

    # Build the 8-value patient feature vector, in the EXACT order
    # the model was trained on (confirmed via inspect_pkls.py)
    carb_target = (macros["per_meal_max_carbs"] + macros["per_meal_min_carbs"]) / 2
    patient_vector_8 = [
        age, bmi, leicester_score, tdee,
        int(on_medication), carb_target,
        macros["per_meal_sugar"], macros["per_meal_calories"],
    ]

    # ── Pick best meals per slot, for all 30 days ────────
    plan_by_slot = {}
    for slot in MEAL_SLOTS:
        plan_by_slot[slot] = pick_best_meals_for_slot(
            slot, patient_vector_8, macros, DAYS_IN_PLAN
        )
        if not plan_by_slot[slot]:
            return jsonify({
                "error": f"No recipes found for {slot} matching this "
                         f"patient's clinical constraints."
            }), 422

    # ── Assemble the full 30-day plan + exercise ─────────
    plan = []
    for day in range(DAYS_IN_PLAN):
        day_entry = {"day": day + 1, "meals": []}
        for slot in MEAL_SLOTS:
            meal = plan_by_slot[slot][day]
            day_entry["meals"].append({
                "meal_type": slot,
                "recipe_name": meal.get("Recipe_Name", "Unknown"),
                "calories": meal["Calories"],
                "carbs_g": meal["Carbs_g"],
                "sugar_g": meal["Sugar_g"],
                "protein_g": meal["Protein_g"],
                "fat_g": meal["Fat_g"],
                "relevance_score": round(meal["relevance_score"], 4),
                "post_meal_exercise": get_post_meal_exercise(
                    age, current_glucose_high, bmi
                ),
            })
        day_entry["daily_evening_workout"] = get_daily_evening_workout(day)
        plan.append(day_entry)

    return jsonify({
        "patient_summary": {
            "age": age, "bmi": bmi, "tdee": tdee,
            "leicester_score": leicester_score,
            "on_medication": on_medication,
            "per_meal_macros": macros,
        },
        "plan": plan,
    })


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "DiaPilot API is running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)