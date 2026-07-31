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
        "medication_risk_level": "none",
        "current_glucose_high": false
    }

    medication_risk_level must be one of:
      "high" — insulin or other medications with genuine hypoglycemia
               risk (e.g. sulfonylureas)
      "low"  — medications like metformin with no meaningful
               hypoglycemia risk on their own
      "none" — not on any diabetes medication
    (Any other/missing value defaults to "high", the safer assumption.)
"""

import os
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from tensorflow import keras
from dotenv import load_dotenv

# Load environment variables from a .env file in this same folder
# (src/.env), the same way the React Native project's .env works.
# This means GEMINI_API_KEY no longer needs to be set manually with
# $env:GEMINI_API_KEY every time a new terminal is opened — it's read
# automatically from .env on every run.
load_dotenv()

app = Flask(__name__)

# ────────────────────────────────────────────────────────────
# Gemini setup — converts recipe names to Pakistani equivalents
# before the plan is sent to the app.
#
# IMPORTANT: this calls Gemini's REST API directly with `requests`
# instead of the `google-generativeai` SDK. The SDK depends on a
# protobuf version that conflicts with the protobuf version
# TensorFlow needs in this environment (confirmed: SDK wants
# protobuf <6.0, TensorFlow needs protobuf >=6.31.1 — no single
# version satisfies both). Calling the REST endpoint directly
# avoids this conflict entirely, since `requests` has no protobuf
# dependency at all.
#
# This is a SEPARATE Gemini key/config from the chatbot's Gemini
# integration (that one lives in the React Native/JS side and is
# unrelated to this).
#
# If GEMINI_API_KEY is not set, the API still works correctly and
# returns the plan with original (English) recipe names — Pakistani
# conversion is a best-effort enhancement, never a hard requirement
# for the endpoint to function.
# ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY)

# ────────────────────────────────────────────────────────────
# Model fallback chain — tried in order, top to bottom.
#
# Why this exists: during testing we found individual Gemini
# models can independently fail even when the API key and
# billing are both fine — e.g. a model being deprecated
# (gemini-2.0-flash, shut down June 2026), a model being
# temporarily overloaded (gemini-flash-latest -> 503 "high
# demand"), or a specific model having zero free-tier quota on
# a given project while another model on the SAME key works
# fine. Trying several models in sequence makes the conversion
# step resilient to any ONE of these failing, without needing
# manual intervention each time.
#
# Ordered cheapest/lightest first (cheaper models tend to have
# more free-tier headroom), falling back to heavier models only
# if the lighter ones are unavailable.
# ────────────────────────────────────────────────────────────
GEMINI_MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

GEMINI_REST_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

if not GEMINI_ENABLED:
    print("WARNING: GEMINI_API_KEY environment variable not set.")
    print("Diet plans will be returned with original (English) recipe"
          " names. Set GEMINI_API_KEY to enable Pakistani conversion.")

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


def get_dynamic_macros(tdee, leicester_score, medication_risk_level="none",
                        current_glucose_high=False):
    """
    ADA 2025-aligned per-meal constraints (same as training notebook).

    medication_risk_level: one of "high", "low", "none".
      - "high": insulin or other medications with genuine hypoglycemia
        risk (e.g. sulfonylureas) — applies the 25g carb safety floor.
      - "low": medications like metformin that do not typically cause
        hypoglycemia on their own — no carb floor applied.
      - "none": not on any diabetes medication — no carb floor applied.
      - If an unrecognized value is passed (e.g. "unsure"), defaults to
        the SAFER "high" behavior — better to apply an unnecessary
        floor than to miss a genuine hypoglycemia risk.
    """
    if leicester_score <= 10:
        carb_pct, sugar_limit = 0.50, 15.0
    elif leicester_score <= 15:
        carb_pct, sugar_limit = 0.40, 10.0
    else:
        carb_pct, sugar_limit = 0.30, 8.0

    if current_glucose_high:
        carb_pct -= 0.10
        sugar_limit = 3.0

    # Safe default: anything other than explicit "low" or "none" is
    # treated as high-risk, so an unrecognized/missing value never
    # silently skips the safety floor.
    is_high_hypoglycemia_risk = medication_risk_level not in ("low", "none")

    per_meal_carbs_max = round(((tdee * carb_pct) / 4) / 3)
    per_meal_carbs_min = 25 if is_high_hypoglycemia_risk else 0
    if is_high_hypoglycemia_risk and per_meal_carbs_max < 25:
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
# Gemini step — Pakistani food conversion
#
# Takes the full English-named 30-day plan and asks Gemini to
# rewrite recipe names (and a short ingredient hint) as their
# closest common Pakistani equivalent, while leaving every
# numeric/clinical field (calories, carbs, sugar, protein, fat,
# scores, exercise text) completely untouched.
#
# Batched as ONE request for the whole 90-meal plan (not one
# request per meal) to keep this fast and to keep Gemini API
# usage low. If anything goes wrong — network issue, malformed
# response, missing key — the ORIGINAL plan is returned
# unchanged rather than failing the whole request.
# ────────────────────────────────────────────────────────────
def convert_plan_to_pakistani(plan):
    if not GEMINI_ENABLED:
        return plan

    # Build a numbered list that includes each meal's TYPE alongside its
    # name (e.g. "1. [Breakfast] beer batter halibut") — without this,
    # the model has no way to know a dish is meant for breakfast vs.
    # dinner, and will happily suggest dinner-style dishes (fried fish,
    # heavy curries) for a breakfast slot. Including meal type lets the
    # model pick genuinely meal-appropriate Pakistani equivalents.
    flat_meals = []
    for day_entry in plan:
        for meal in day_entry["meals"]:
            flat_meals.append((meal["meal_type"], meal["recipe_name"]))

    numbered_list = "\n".join(
        f"{i+1}. [{meal_type}] {name}"
        for i, (meal_type, name) in enumerate(flat_meals)
    )

    prompt = (
        "You are helping localize a diabetic meal plan for a Pakistani "
        "household. Below is a numbered list of dish names from a "
        "Western recipe dataset, each tagged with its meal slot "
        "(Breakfast, Lunch, or Dinner).\n\n"
        "For each one, reply with a SIMPLE, EVERYDAY Pakistani dish "
        "name that an ordinary Pakistani household actually eats and "
        "cooks regularly — not a fancy, invented, or overly literal "
        "translation. Avoid names that sound like translated diet "
        "products (e.g. do not say things like 'sugar-free diet "
        "jelly' — instead pick a real dish, like a specific daal or "
        "sabzi, that naturally fits the calorie/carb range).\n\n"
        "Meal-appropriateness matters:\n"
        "- Breakfast items should be genuine Pakistani breakfast foods "
        "(e.g. anda paratha, plain omelette, daliya, low-fat lassi, "
        "besan cheela) — NOT heavy dinner dishes like fried fish or "
        "karahi, even if the original Western dish was a 'breakfast' "
        "entry in the dataset.\n"
        "- Lunch and Dinner items should be common Pakistani main "
        "dishes (e.g. daal, sabzi, grilled or roasted chicken/fish, "
        "karahi, khichdi) that fit the same general calorie/protein "
        "range as the original.\n\n"
        "IMPORTANT — fully commit to the substitution, do not blend "
        "English words onto a Pakistani dish name, and never keep "
        "non-Pakistani ingredients in the name even if the original "
        "recipe used them. If the original dish contains an ingredient "
        "uncommon in Pakistani cooking (e.g. quinoa, broccoli, kale), "
        "replace it entirely with a common Pakistani ingredient that "
        "fits a similar role (e.g. quinoa -> replace with roti/rice as "
        "part of a normal Pakistani dish; broccoli -> replace with "
        "aloo, palak, bhindi, or gobi).\n\n"
        "Examples of what NOT to do, and the correct alternative:\n"
        "- WRONG: 'Grilled Chicken with Quinoa Salad' (keeps a non-"
        "Pakistani ingredient) -> RIGHT: 'Grilled Chicken with Roti'\n"
        "- WRONG: 'Steamed Prawns with Herbs' (Western-style phrasing) "
        "-> RIGHT: 'Jhinga Masala' or 'Prawn Curry'\n"
        "- WRONG: 'Creamy Fish Handi' (English adjective bolted onto a "
        "Pakistani noun) -> RIGHT: 'Fish Handi' or 'Malai Fish Curry'\n\n"
        "Do NOT change the nutritional meaning — only replace the name "
        "with a realistic, commonly recognized Pakistani equivalent "
        "that fits both the meal slot and the approximate nutrition "
        "profile. Reply with ONLY a numbered list in the exact same "
        "order, one Pakistani dish name per line, no extra "
        "commentary, no meal-type tags in your reply.\n\n"
        f"{numbered_list}"
    )

    # Try each model in the fallback chain, in order, until one
    # succeeds with a correctly-shaped response. This makes the
    # conversion step resilient to any single model being deprecated,
    # overloaded, or out of quota — without needing to know in
    # advance which model will actually work today.
    for model_name in GEMINI_MODEL_FALLBACK_CHAIN:
        url = GEMINI_REST_URL_TEMPLATE.format(model=model_name)

        try:
            response = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ]
                },
                timeout=30,
            )

            if response.status_code != 200:
                print(f"INFO: Gemini model '{model_name}' returned "
                      f"status {response.status_code}, trying next "
                      f"model in fallback chain.")
                continue

            result = response.json()

            # Gemini REST response shape:
            # { "candidates": [ { "content": { "parts": [ {"text": "..."} ] } } ] }
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"]

            lines = [
                line.strip() for line in raw_text.strip().split("\n")
                if line.strip()
            ]

            # Parse "1. Dish Name" -> "Dish Name", keep order strict
            converted_names = []
            for line in lines:
                if ". " in line:
                    converted_names.append(line.split(". ", 1)[1].strip())
                else:
                    converted_names.append(line.strip())

            # Safety check: only apply if Gemini returned exactly as
            # many names as we sent. If the count doesn't match,
            # something went wrong with parsing for THIS model —
            # try the next one in the chain rather than giving up.
            if len(converted_names) != len(flat_meals):
                print(f"INFO: Gemini model '{model_name}' returned "
                      f"{len(converted_names)} names, expected "
                      f"{len(flat_meals)}. Trying next model in "
                      f"fallback chain.")
                continue

            # Success — apply the converted names back onto the plan,
            # same order, then stop trying further models.
            idx = 0
            for day_entry in plan:
                for meal in day_entry["meals"]:
                    meal["recipe_name"] = converted_names[idx]
                    idx += 1

            print(f"INFO: Pakistani conversion succeeded using model "
                  f"'{model_name}'.")
            return plan

        except Exception as e:
            print(f"INFO: Gemini model '{model_name}' failed ({e}), "
                  f"trying next model in fallback chain.")
            continue

    # Every model in the chain failed — return the original
    # (English) plan rather than failing the whole request.
    print("WARNING: All models in the Gemini fallback chain failed. "
          "Returning original (English) recipe names.")
    return plan


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

        # medication_risk_level: "high" (insulin/sulfonylureas — genuine
        # hypoglycemia risk), "low" (e.g. metformin — no meaningful
        # hypoglycemia risk on its own), or "none" (not on medication).
        #
        # Backward compatibility: if an older client still sends the
        # original on_medication boolean instead, translate it safely
        # (True -> "high", the safer assumption; False -> "none").
        if "medication_risk_level" in data:
            medication_risk_level = data["medication_risk_level"]
        elif "on_medication" in data:
            medication_risk_level = "high" if bool(data["on_medication"]) else "none"
        else:
            medication_risk_level = "high"  # safest default if omitted entirely

        current_glucose_high = bool(data.get("current_glucose_high", False))
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    # Separate, binary signal for the DNN's trained "p_medication"
    # feature — the model was trained on a simple on/off medication
    # flag and was not retrained for risk-level granularity, so this
    # stays binary: on if the patient takes ANY diabetes medication
    # (whether high or low hypoglycemia risk), off if none.
    is_on_any_medication = medication_risk_level in ("high", "low")

    # ── Clinical engine ──────────────────────────────────
    tdee = calculate_tdee(age, weight_kg, height_cm, gender, activity)
    macros = get_dynamic_macros(tdee, leicester_score, medication_risk_level, current_glucose_high)
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

    # Build the 8-value patient feature vector, in the EXACT order
    # the model was trained on (confirmed via inspect_pkls.py)
    carb_target = (macros["per_meal_max_carbs"] + macros["per_meal_min_carbs"]) / 2
    patient_vector_8 = [
        age, bmi, leicester_score, tdee,
        int(is_on_any_medication), carb_target,
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

    # ── Gemini step: convert recipe names to Pakistani equivalents ──
    # Safe no-op if GEMINI_API_KEY isn't set or the call fails —
    # see convert_plan_to_pakistani() for the fallback behavior.
    plan = convert_plan_to_pakistani(plan)

    return jsonify({
        "patient_summary": {
            "age": age, "bmi": bmi, "tdee": tdee,
            "leicester_score": leicester_score,
            "medication_risk_level": medication_risk_level,
            "per_meal_macros": macros,
        },
        "plan": plan,
    })


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "DiaPilot API is running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)